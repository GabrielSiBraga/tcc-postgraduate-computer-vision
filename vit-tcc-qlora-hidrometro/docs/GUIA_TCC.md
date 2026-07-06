# Guia TCC - Pipeline Híbrido de Leitura de Hidrômetros

Documento de referência para reproduzir e avaliar o projeto entregue neste repositório.

## 1. Objetivo

Automatizar a leitura de hidrômetros analógicos a partir de fotos de campo, combinando detecção de objetos (Mask R-CNN) e visão-linguagem (Florence-2 QLoRA).

## 2. Fluxo completo

| # | Script / etapa | Entrada | Saída | Motivo |
|---|----------------|---------|-------|--------|
| 0a | `00_calibrate_crop.py` | Imagens + bbox | Parâmetros de expansão | Crop inclui visor completo |
| 0b | `01_generate_crops.py` | Dataset COCO | `data/crops/` | CLAHE melhora roletes |
| 1a | `00_purge_labels.py` | Labels legados | - | Remove schema antigo (só se reproduzir do zero) |
| 1b | *(histórico)* Label Studio | Crops | Export anotado | Autolabel inicial + revisão humana |
| 1c | `03_export_label_studio.py --mode convert` | `data/label_studio/export/` | `data/autolabel/validated/` | Parse do export LS |
| 1d | `03_export_label_studio.py --mode export` | Validated | `data/sft/*.jsonl` | Dataset SFT |
| 1e | `06_audit_labels.py` | SFT + validated | `reports/label_audit.json` | Consistência inteiro vs completo |
| 2 | `04_train_qlora.py` | SFT train/val | `artifacts/lora_adapter/` | Fine-tuning LoRA |
| 3 | `05_evaluate.py` | Crops test + GT | `reports/evaluation_test.json` | Métricas finais |
| 4 | API + Streamlit | Foto bruta | JSON + overlay | Demonstração |

**Nota:** Para a entrega, labels e LoRA **já estão prontos**. Não rode autolabel ou Label Studio novamente.

## 3. Dataset

| Split | Amostras SFT | Uso |
|-------|--------------|-----|
| train | 503 | Treino LoRA |
| val | 144 | Early stopping / eval_loss |
| test | 72 | Métricas finais (nunca treinar) |

Ground truth: `data/autolabel/validated/{split}/` (derivado de `data/label_studio/export/`).

Auditoria atual: **100% consistência** (719 registros, 0 issues).

## 4. Treino QLoRA - decisões de design

### Modelo
- **Florence-2-large**: encoder DaViT + decoder seq2seq
- **4-bit NF4**: cabe em 16 GB VRAM

### LoRA
- **r=16, alpha=32**: ~3,5M parâmetros treináveis (0,8%)
- **Targets q/k/v/o_proj**: atenção do language model

### Hiperparâmetros (`configs/qlora.yaml`)

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| lr | 1e-4 | Dataset pequeno; evita overfit |
| épocas | 8 | Early stopping seleciona melhor checkpoint |
| eval_steps | 50 | Seleção fina por eval_loss |
| load_best_model_at_end | true | Salva melhor val, não última época |

## 5. Métricas - como interpretar

### Leitura (split test, n=72)

| Métrica | Baseline | Pós-treino | Interpretação |
|---------|----------|------------|---------------|
| Parse JSON | 0% | **100%** | Resposta estruturada válida |
| Exact match | 0% | **38,9%** | String `completo` idêntica ao GT |
| Acurácia por caractere | 0% | **79,0%** | 1 − CER; mais permissiva que exact match |
| CER completo | 1,0 | **0,21** | Taxa de erro por caractere |

Exact match ≠ acurácia por caractere: a primeira exige vírgula e zeros corretos; a segunda tolera edições parciais.

### Classificação

| Métrica | Valor | Nota |
|---------|-------|------|
| Acurácia fabricante | **97,2%** | Acertos por amostra |
| Acurácia estado | **97,2%** | Acertos por amostra |
| F1 fabricante (macro) | **73,8%** | Penaliza classes raras |
| F1 estado (macro) | **49,3%** | *embacado* tem 2 amostras no teste |

## 6. Engenharia

| Métrica | Valor |
|---------|-------|
| VRAM peak inferência | ~746 MB |
| Adaptador LoRA | ~81 MB |
| Throughput | ~0,54 img/s |

## 7. Demo (FastAPI + Streamlit)

1. Subir API: `PYTHONPATH=src uvicorn hidrometro.api.main:app --port 8000`
2. Subir UI: `streamlit run src/hidrometro/ui/streamlit_app.py`
3. Enviar **foto completa** do hidrômetro (não crop)
4. Resposta: leitura JSON, fabricante, estado, overlay e crop CLAHE

## 8. Limitações

1. Dependência do Detectron2 para localizar hidrômetro
2. Exact match 38,9% - transcrição literal é exigente
3. F1 macro baixo em classes raras (*embacado*, marcas minoritárias)
4. Dataset ~500 amostras de treino
5. Primeira inferência requer download do Florence-2 (~1,5 GB)

## 9. Arquivos principais da entrega

- `reports/evaluation_test.json`
- `reports/notebook_comparison_test.json`
- `reports/label_audit.json`
- `notebooks/04_resultados_comparativos.ipynb`
- `artifacts/lora_adapter/`
- `../mask-rcnn/output/model_final.pth`
