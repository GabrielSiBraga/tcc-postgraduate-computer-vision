# Módulo 1 — Detecção Mask R-CNN

Parte do TCC [`TrabalhoDetecaoObjetos`](../README.md): localiza e segmenta hidrômetros nas fotos de campo.

## Resultado

| Métrica | Valor | Motivo |
|---------|-------|--------|
| Acurácia de máscara (treino final) | **~98,2%** | Detecção confiável do visor para o pipeline VLM |

Curvas completas: [`output/metrics.json`](output/metrics.json).

---

## Pipeline desta fase

| Passo | O que | Por quê |
|-------|-------|---------|
| 1 | Anotações COCO em `dataset-hidrometro/` | bbox + máscara por instância |
| 2 | Treino Mask R-CNN R50-FPN | Generaliza em fotos de campo |
| 3 | Export `model_final.pth` | Pesos congelados para inferência |
| 4 | Classe `display` (id=1) | Visor usado pelo módulo 2 para crop |

O módulo 2 **não retreina** o Detectron2 — consome `model_final.pth` via [`vit-tcc-qlora-hidrometro/configs/paths.yaml`](../vit-tcc-qlora-hidrometro/configs/paths.yaml).

---

## Artefatos

| Arquivo | Descrição |
|---------|-----------|
| `output/model_final.pth` | Pesos treinados (~335 MB) |
| `output/metrics.json` | Logs de treino (acurácia máscara) |
| `dataset-hidrometro/` | Anotações COCO (train/valid/test) |
| `detectron2_repo/` | Biblioteca Detectron2 (vendored) |

---

## Notebook

[`00_mask_rcnn_hidrometro.ipynb`](00_mask_rcnn_hidrometro.ipynb) — treino, avaliação COCO e visualizações.

Kernel sugerido: **Python 3.14 (detectron2)** ou ambiente com Detectron2 compilado.

---

## Uso no pipeline completo

O módulo 2 ([`vit-tcc-qlora-hidrometro/`](../vit-tcc-qlora-hidrometro/)) consome:

- `output/model_final.pth` — detecção congelada
- `detectron2_repo/` — via `pip install -e` ou `sys.path`

Scripts de crop: `vit-tcc-qlora-hidrometro/scripts/01_generate_crops.py`.

---

## Instalação Detectron2

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e detectron2_repo --no-build-isolation
```

Ambiente completo (Detectron2 + VLM): [`vit-tcc-qlora-hidrometro/scripts/setup_env.sh`](../vit-tcc-qlora-hidrometro/scripts/setup_env.sh).

Troubleshooting: [`vit-tcc-qlora-hidrometro/README.md`](../vit-tcc-qlora-hidrometro/README.md).
