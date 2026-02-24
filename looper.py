"""
ComfyUI custom node to detect and extract a loop cycle from an IMAGE batch.

Input batch shape: [frames, height, width, channels]
Output: a new IMAGE batch containing one detected loop cycle.

Behavior:
- If a confident loop is found -> returns exactly one full loop cycle.
- If no confident loop is found:
  - fallback_strategy="original"    -> returns original batch.
  - fallback_strategy="best_effort" -> returns closest cycle candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass
class LoopDetectionResult:
    period_frames: int
    start_frame: int
    score: float
    baseline_score: float
    confidence: float
    estimated_loops: float


@dataclass
class SeamCheckResult:
    seam_distance: float
    adjacent_mean: float
    adjacent_p90: float
    seam_ratio: float
    ok: bool


def _as_bhwc(images: torch.Tensor) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B,H,W,C], got {tuple(images.shape)}")
    if images.shape[-1] not in (1, 3, 4):
        raise ValueError("images last dim must be 1, 3, or 4 channels")
    return images


def _to_gray(images_bhwc: torch.Tensor) -> torch.Tensor:
    x = images_bhwc[..., :3].float().clamp(0.0, 1.0)
    if x.shape[-1] == 1:
        return x[..., 0]
    return 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]


def _resize_gray_batch(gray_bhw: torch.Tensor, size: int) -> torch.Tensor:
    g = gray_bhw.unsqueeze(1)
    g = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    return g[:, 0]


def _build_dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    mat = torch.cos(math.pi / float(n) * (i + 0.5) * k)
    mat[0] *= 1.0 / math.sqrt(2.0)
    mat *= math.sqrt(2.0 / float(n))
    return mat


def _extract_fast_features(images_bhwc: torch.Tensor) -> tuple[torch.Tensor, None]:
    gray = _to_gray(images_bhwc)
    small = _resize_gray_batch(gray, size=16)
    flat = small.reshape(small.shape[0], -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = flat / (flat.std(dim=1, keepdim=True) + EPS)

    rgb = images_bhwc[..., :3].float().clamp(0.0, 1.0)
    rgb_mean = rgb.mean(dim=(1, 2))
    features = torch.cat([flat, rgb_mean], dim=1)
    return features, None


def _extract_pro_features(images_bhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = _to_gray(images_bhwc)
    small = _resize_gray_batch(gray, size=32)

    vec = small.reshape(small.shape[0], -1)
    vec = vec - vec.mean(dim=1, keepdim=True)
    vec = vec / (vec.std(dim=1, keepdim=True) + EPS)

    dct = _build_dct_matrix(32, device=small.device, dtype=small.dtype)
    tmp = torch.einsum("ij,bjk->bik", dct, small)
    coeff = torch.einsum("bij,jk->bik", tmp, dct.t())

    block = coeff[:, :8, :8].clone()
    block[:, 0, 0] = 0.0
    med = block.reshape(block.shape[0], -1).median(dim=1, keepdim=True).values
    bits = (block.reshape(block.shape[0], -1) > med).to(torch.bool)

    gx = small[:, :, 1:] - small[:, :, :-1]
    gy = small[:, 1:, :] - small[:, :-1, :]
    edge_energy = (gx.abs().mean(dim=(1, 2)) + gy.abs().mean(dim=(1, 2))).unsqueeze(1)

    features = torch.cat([vec, edge_energy], dim=1)
    return features, bits


def _sample_indices(n_pairs: int, max_samples: int, device: torch.device) -> torch.Tensor:
    if n_pairs <= max_samples:
        return torch.arange(n_pairs, device=device)
    return torch.linspace(0, n_pairs - 1, max_samples, device=device).long()


def _lag_score(
    features: torch.Tensor,
    lag: int,
    bits: Optional[torch.Tensor],
    max_samples: int,
) -> float:
    n = features.shape[0]
    n_pairs = n - lag
    if n_pairs < 4:
        return float("inf")

    idx = _sample_indices(n_pairs, max_samples=max_samples, device=features.device)
    a = features[idx]
    b = features[idx + lag]
    dist = (a - b).abs().mean(dim=1)
    score = float(dist.mean().item())

    if bits is not None:
        ba = bits[idx]
        bb = bits[idx + lag]
        h = (ba != bb).float().mean(dim=1)
        score = 0.65 * score + 0.35 * float(h.mean().item())

    return score


def _phase_alignment_score(
    features: torch.Tensor,
    period: int,
    start: int,
    bits: Optional[torch.Tensor],
    max_samples: int,
) -> float:
    n = features.shape[0]
    n_pairs = min(period, n - start - period)
    if n_pairs < 4:
        return float("inf")

    idx = _sample_indices(n_pairs, max_samples=max_samples, device=features.device)
    a = features[start + idx]
    b = features[start + period + idx]
    dist = (a - b).abs().mean(dim=1)
    score = float(dist.mean().item())

    if bits is not None:
        ba = bits[start + idx]
        bb = bits[start + period + idx]
        h = (ba != bb).float().mean(dim=1)
        score = 0.65 * score + 0.35 * float(h.mean().item())

    return score


def _feature_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    bits_a: Optional[torch.Tensor] = None,
    bits_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    score = (a - b).abs().mean(dim=1)
    if bits_a is not None and bits_b is not None:
        h = (bits_a != bits_b).float().mean(dim=1)
        score = 0.65 * score + 0.35 * h
    return score


def _divisors(value: int) -> list[int]:
    out: list[int] = []
    for d in range(2, int(math.sqrt(value)) + 1):
        if value % d == 0:
            out.append(d)
            other = value // d
            if other != d:
                out.append(other)
    out.append(value)
    out = sorted(set(out))
    return out


def _find_fundamental_period(
    best_period: int,
    min_period: int,
    score_fn,
) -> int:
    candidate_score = score_fn(best_period)
    threshold = candidate_score * 1.2 + 0.01

    factors = _divisors(best_period)
    for f in factors:
        if f < min_period:
            continue
        if f >= best_period:
            break
        s = score_fn(f)
        if s <= threshold:
            return f
    return best_period


def _detect_loop_period_and_phase(
    features: torch.Tensor,
    bits: Optional[torch.Tensor],
    min_period: int,
    max_period: int,
    mode: str,
) -> Optional[LoopDetectionResult]:
    n = features.shape[0]
    if n < 8:
        return None

    max_lag = min(max_period, n - 1)
    min_lag = max(2, min_period)
    if max_lag <= min_lag:
        return None

    score_cache: dict[int, float] = {}

    if mode == "fast":
        lag_samples = 240
        phase_samples = 120
        refine_window = 2
    else:
        lag_samples = 420
        phase_samples = 240
        refine_window = 8

    def score_lag(lag: int) -> float:
        if lag not in score_cache:
            score_cache[lag] = _lag_score(features, lag, bits=bits, max_samples=lag_samples)
        return score_cache[lag]

    best_lag = None
    best_score = float("inf")
    for lag in range(min_lag, max_lag + 1):
        s = score_lag(lag)
        if s < best_score:
            best_score = s
            best_lag = lag

    if best_lag is None:
        return None

    best_lag = _find_fundamental_period(best_lag, min_period=min_lag, score_fn=score_lag)

    lo = max(min_lag, best_lag - refine_window)
    hi = min(max_lag, best_lag + refine_window)
    for lag in range(lo, hi + 1):
        s = score_lag(lag)
        if s < best_score:
            best_score = s
            best_lag = lag

    max_start = min(best_lag - 1, n - (2 * best_lag))
    if max_start < 0:
        phase = 0
        phase_score = best_score
    else:
        phase = 0
        phase_score = float("inf")
        for s in range(0, max_start + 1):
            p_score = _phase_alignment_score(
                features,
                period=best_lag,
                start=s,
                bits=bits,
                max_samples=phase_samples,
            )
            if p_score < phase_score:
                phase_score = p_score
                phase = s

    probe_lags = sorted(set([
        min_lag,
        (min_lag + max_lag) // 2,
        max_lag,
        max(min_lag, int(round(max_lag * 0.75))),
    ]))
    baseline_vals = [score_lag(l) for l in probe_lags if l != best_lag]
    baseline = float(sum(baseline_vals) / len(baseline_vals)) if baseline_vals else (best_score + 1e-3)

    periodic_gain = max(0.0, 1.0 - (best_score / (baseline + EPS)))
    phase_gain = max(0.0, 1.0 - (phase_score / (baseline + EPS)))
    confidence = max(0.0, min(1.0, 0.7 * periodic_gain + 0.3 * phase_gain))

    estimated_loops = float(n) / float(best_lag)
    return LoopDetectionResult(
        period_frames=int(best_lag),
        start_frame=int(phase),
        score=float(best_score),
        baseline_score=float(baseline),
        confidence=float(confidence),
        estimated_loops=float(estimated_loops),
    )


def _check_cycle_seam(
    features: torch.Tensor,
    bits: Optional[torch.Tensor],
    start: int,
    period: int,
    mode: str,
) -> SeamCheckResult:
    seg = features[start : start + period]
    if seg.shape[0] < 3:
        return SeamCheckResult(
            seam_distance=float("inf"),
            adjacent_mean=float("inf"),
            adjacent_p90=float("inf"),
            seam_ratio=float("inf"),
            ok=False,
        )

    if mode == "strict":
        max_samples = 320
    elif mode == "pro":
        max_samples = 220
    else:
        max_samples = 120

    idx = _sample_indices(seg.shape[0] - 1, max_samples=max_samples, device=seg.device)
    a = seg[idx]
    b = seg[idx + 1]

    if bits is not None:
        bseg = bits[start : start + period]
        badj = bseg[idx]
        bnext = bseg[idx + 1]
    else:
        badj = None
        bnext = None

    adjacent = _feature_distance(a, b, badj, bnext)
    adjacent_mean = float(adjacent.mean().item())
    adjacent_p90 = float(torch.quantile(adjacent, 0.90).item())

    if bits is not None:
        first_bits = bits[start : start + 1]
        last_bits = bits[start + period - 1 : start + period]
    else:
        first_bits = None
        last_bits = None

    seam = _feature_distance(
        seg[period - 1 : period],
        seg[0:1],
        last_bits,
        first_bits,
    )
    seam_distance = float(seam[0].item())
    seam_ratio = seam_distance / (adjacent_mean + EPS)

    allowed = max(adjacent_p90 * 1.08, adjacent_mean * 1.35)
    if mode == "strict":
        allowed = min(allowed, adjacent_mean * 1.18 + 0.005)
    ok = seam_distance <= allowed

    return SeamCheckResult(
        seam_distance=seam_distance,
        adjacent_mean=adjacent_mean,
        adjacent_p90=adjacent_p90,
        seam_ratio=seam_ratio,
        ok=ok,
    )


def _auto_threshold(mode: str) -> float:
    if mode == "fast":
        return 0.34
    if mode == "pro":
        return 0.44
    return 0.56


def _slice_cycle(images: torch.Tensor, start: int, period: int) -> Optional[torch.Tensor]:
    if period <= 1:
        return None
    if start < 0:
        return None
    end = start + period
    if end > images.shape[0]:
        return None
    return images[start:end]


class LoppinerLoopExtractor:
    """Detect and extract one full loop cycle from an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["fast", "pro", "strict"], {"default": "pro"}),
                "fallback_strategy": (["original", "best_effort"], {"default": "original"}),
                "min_period_frames": ("INT", {"default": 6, "min": 2, "max": 100000}),
                "max_period_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "confidence_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "images",
        "period_frames",
        "start_frame",
        "confidence",
        "estimated_loops",
        "status",
    )
    FUNCTION = "extract_loop"
    CATEGORY = "loppiner/video"

    def extract_loop(
        self,
        images: torch.Tensor,
        mode: str,
        fallback_strategy: str,
        min_period_frames: int,
        max_period_frames: int,
        confidence_threshold: float,
    ):
        images = _as_bhwc(images)
        total_frames = int(images.shape[0])

        if total_frames < 8:
            return (
                images,
                0,
                0,
                0.0,
                0.0,
                "NO_LOOP_TOO_SHORT_RETURNED_ORIGINAL",
            )

        max_period = max_period_frames if max_period_frames > 0 else (total_frames // 2)
        max_period = max(2, min(max_period, total_frames - 1))
        min_period = max(2, min(min_period_frames, max_period))

        if mode == "fast":
            features, bits = _extract_fast_features(images)
        else:
            features, bits = _extract_pro_features(images)

        result = _detect_loop_period_and_phase(
            features=features,
            bits=bits,
            min_period=min_period,
            max_period=max_period,
            mode=mode,
        )

        if result is None:
            return (
                images,
                0,
                0,
                0.0,
                0.0,
                "NO_LOOP_DETECTED_RETURNED_ORIGINAL",
            )

        threshold = confidence_threshold if confidence_threshold > 0.0 else _auto_threshold(mode)
        has_confident_loop = result.confidence >= threshold

        selected = _slice_cycle(images, result.start_frame, result.period_frames)
        if selected is None:
            return (
                images,
                0,
                0,
                float(result.confidence),
                float(result.estimated_loops),
                "INVALID_SLICE_RETURNED_ORIGINAL",
            )

        seam = _check_cycle_seam(
            features=features,
            bits=bits,
            start=result.start_frame,
            period=result.period_frames,
            mode=mode,
        )
        strict_ok = True if mode != "strict" else seam.ok

        if has_confident_loop and strict_ok:
            if mode == "strict":
                status = f"LOOP_FOUND_STRICT_SEAM_OK_{seam.seam_ratio:.3f}"
            else:
                status = "LOOP_FOUND"
            return (
                selected,
                int(result.period_frames),
                int(result.start_frame),
                float(result.confidence),
                float(result.estimated_loops),
                status,
            )

        if fallback_strategy == "best_effort":
            if mode == "strict" and not seam.ok:
                status = f"STRICT_SEAM_FAIL_BEST_EFFORT_{seam.seam_ratio:.3f}"
            else:
                status = "NO_CONFIDENT_LOOP_BEST_EFFORT"
            return (
                selected,
                int(result.period_frames),
                int(result.start_frame),
                float(result.confidence),
                float(result.estimated_loops),
                status,
            )

        if mode == "strict" and not seam.ok:
            status = f"STRICT_SEAM_FAIL_RETURNED_ORIGINAL_{seam.seam_ratio:.3f}"
        else:
            status = "NO_CONFIDENT_LOOP_RETURNED_ORIGINAL"

        return (
            images,
            0,
            0,
            float(result.confidence),
            float(result.estimated_loops),
            status,
        )


NODE_CLASS_MAPPINGS = {
    "LoppinerLoopExtractor": LoppinerLoopExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoppinerLoopExtractor": "Loppiner Loop Extractor",
}
