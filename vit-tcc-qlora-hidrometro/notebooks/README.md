# Notebooks — Documentação do TCC

Ordem recomendada para o avaliador. Kernel: **Python 3 (vit-tcc-qlora)** (`.vscode/settings.json` do módulo).

| # | Notebook | Fase | O que demonstra | Por quê |
|---|----------|------|-----------------|---------|
| 0 | [00_mask_rcnn_hidrometro.ipynb](../../mask-rcnn/00_mask_rcnn_hidrometro.ipynb) | Detecção | Treino Mask R-CNN, curvas, métricas COCO | Fase 1 — localizar hidrômetro |
| 1 | [01_preprocessamento.ipynb](01_preprocessamento.ipynb) | Pré-processamento | Calibração crop, CLAHE, exemplos | Contraste e recorte para o VLM |
| 2 | [02_dataset_e_labels.ipynb](02_dataset_e_labels.ipynb) | Dados | Stats SFT, schema JSON, auditoria | Ground truth e consistência |
| 3 | [03_treino_qlora.ipynb](03_treino_qlora.ipynb) | Treino | Config QLoRA, loss, early stopping | Fine-tuning com ~500 amostras |
| 4 | [04_resultados_comparativos.ipynb](04_resultados_comparativos.ipynb) | Avaliação | Baseline vs QLoRA, conclusão dinâmica | Métricas finais documentadas |

## Labels

Não recrie labels. Notebooks 2–4 usam `data/sft/`, `data/autolabel/validated/` e `data/label_studio/export/`.

## Retreino

Por padrão `RUN_TRAINING = False` nos notebooks 3 e 4. Para retreinar: `RUN_TRAINING = True` ou `PYTHONPATH=src python scripts/04_train_qlora.py`.

## Resultados esperados (test, n=72)

Alinhados com [`reports/evaluation_test.json`](../reports/evaluation_test.json): parse **100%**, exact match **38,9%**, acurácia por caractere **79,0%**, fabricante/estado **97,2%**.
