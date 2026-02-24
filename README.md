# Loppiner ComfyUI Loop Extractor

Custom node para ComfyUI que detecta e extrai 1 ciclo de loop de um batch de imagens (`IMAGE`).

## Node

- Classe: `LoppinerLoopExtractor`
- Categoria: `loppiner/video`

## Inputs

- `images`: batch `IMAGE` (frames do video)
- `mode`: `fast` | `pro` | `strict`
- `fallback_strategy`: `original` | `best_effort`
- `min_period_frames`: periodo minimo buscado (frames)
- `max_period_frames`: periodo maximo buscado (0 = automatico)
- `confidence_threshold`: 0.0 usa threshold automatico por modo

## Outputs

- `images`: batch retornado
- `period_frames`: periodo detectado
- `start_frame`: offset de inicio
- `confidence`: confianca da deteccao
- `estimated_loops`: loops aproximados no batch de entrada
- `status`: status textual da decisao

## Modos

- `fast`: mais rapido, menor custo computacional.
- `pro`: mais robusto e preciso.
- `strict`: igual ao `pro`, mas com validacao extra de costura (ultimo->primeiro frame). Se a costura falhar, o loop nao e aceito.

## Fallback

- `original`: se nao houver loop confiavel, retorna batch original.
- `best_effort`: retorna o ciclo mais proximo de um loop, mesmo sem confianca ideal.

## Instalacao

1. Copie a pasta deste projeto para `ComfyUI/custom_nodes/`.
2. Reinicie o ComfyUI.
3. Procure por `Loppiner Loop Extractor`.
