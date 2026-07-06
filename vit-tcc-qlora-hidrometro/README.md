# Módulo 2 - Pipeline VLM + QLoRA

Parte do TCC [`TrabalhoDetecaoObjetos`](../README.md): após o Mask R-CNN localizar o hidrômetro, **Florence-2 QLoRA** lê o visor e retorna JSON estruturado.

## Resultados (split test, 72 amostras)

| Métrica | Baseline | Pós-treino QLoRA |
|---------|----------|------------------|
| Parse JSON | 0% | **100%** |
| Exact match | 0% | **38,9%** |
| Acurácia por caractere (1 − CER) | 0% | **79,0%** |
| Acurácia fabricante | 0% | **97,2%** |
| Acurácia estado | 0% | **97,2%** |
| F1 fabricante (macro) | 0% | **73,8%** |
| F1 estado (macro) | 0% | **49,3%** |

Ver [`reports/evaluation_test.json`](reports/evaluation_test.json). Comparativo baseline vs QLoRA: [`reports/notebook_comparison_test.json`](reports/notebook_comparison_test.json).

---

## Pipeline - scripts e motivos

| Script | Entrada | Saída | Motivo |
|--------|---------|-------|--------|
| `00_calibrate_crop.py` | Bboxes COCO | Parâmetros de expansão | Crop inclui visor completo |
| `01_generate_crops.py` | Fotos + detecção | `data/crops/` | CLAHE melhora legibilidade dos roletes |
| `03_export_label_studio.py --mode convert` | Export LS | `data/autolabel/validated/` | Parse das anotações finais |
| `03_export_label_studio.py --mode export` | Validated | `data/sft/*.jsonl` | Dataset para treino |
| `04_train_qlora.py` | SFT train/val | `artifacts/lora_adapter/` | Fine-tuning LoRA (~40 min GPU) |
| `05_evaluate.py` | Crops test + GT | `reports/evaluation_test.json` | Métricas finais |
| `06_audit_labels.py` | SFT + validated | `reports/label_audit.json` | Consistência inteiro vs completo |
| `07_apply_manual_review.py` | CSV revisão | validated + SFT | Correções pontuais de labels |

**Entrega TCC:** labels e LoRA **já prontos** - não é necessário rodar autolabel ou Label Studio novamente.

---

## Labels

**Não recrie labels validados.** Use `data/sft/*.jsonl` (503/144/72) e `artifacts/lora_adapter/`.

### Anotação no Label Studio

1. Auto-labeling inicial (GPT-4o) importado no Label Studio para acelerar
2. Correção manual campo a campo
3. Export → `validated/` → `sft/`

Artefatos intermediários (prelabels e import) **não estão versionados**. Registro mantido: [`data/label_studio/export/`](data/label_studio/export/) (719 tasks).

| Pasta | Conteúdo |
|-------|----------|
| `data/label_studio/export/` | Export JSON final do Label Studio |
| `data/autolabel/validated/` | Ground truth parseado |
| `data/sft/` | Dataset SFT para treino |

---

## Notebooks

Índice: [`notebooks/README.md`](notebooks/README.md)

| Notebook | Conteúdo |
|----------|----------|
| `01_preprocessamento.ipynb` | Crops + CLAHE |
| `02_dataset_e_labels.ipynb` | SFT, schema JSON, auditoria |
| `03_treino_qlora.ipynb` | Treino e curvas de loss |
| `04_resultados_comparativos.ipynb` | Baseline vs QLoRA (conclusão dinâmica) |

Kernel: **Python 3 (vit-tcc-qlora)** - ver `.vscode/settings.json`.

---

## Instalação

```bash
cd vit-tcc-qlora-hidrometro
bash scripts/setup_env.sh
source .venv/Scripts/activate
pip install -e ".[dev]"
```

Depende de [`../mask-rcnn/`](../mask-rcnn/) para Detectron2 e `model_final.pth`.

---

## Deploy (demo)

**1. API** - carrega modelos no startup:

```bash
source .venv/Scripts/activate
PYTHONPATH=src uvicorn hidrometro.api.main:app --host 0.0.0.0 --port 8000
```

**2. Streamlit** - upload de foto → chama `POST /predict`:

```bash
streamlit run src/hidrometro/ui/streamlit_app.py
```

Envie **fotos completas** do hidrômetro. Primeira inferência pode baixar Florence-2 do HuggingFace (~1,5 GB).

---

## Documentação

- [`docs/GUIA_TCC.md`](docs/GUIA_TCC.md) - fluxo, métricas e limitações
- [`docs/MODELOS_E_ARTEFATOS.md`](docs/MODELOS_E_ARTEFATOS.md) - artefatos no repo
- [`reports/README.md`](reports/README.md) - JSONs de métricas

## Estrutura

```
vit-tcc-qlora-hidrometro/
├── configs/           # qlora.yaml, paths.yaml, autolabel.yaml
├── scripts/           # Pipeline 00–07
├── src/hidrometro/    # API, treino, inferência, pipeline híbrido
├── notebooks/
├── artifacts/lora_adapter/
├── data/
│   ├── autolabel/validated/
│   ├── label_studio/export/
│   ├── crops/
│   └── sft/
└── reports/
```

## Troubleshooting

| Erro | Solução |
|------|---------|
| Kernel errado no Jupyter | Selecione `Python 3 (vit-tcc-qlora)` |
| `WinError 5` cv2 | Feche Jupyter antes de instalar |
| `No module named peft` | Use o `.venv` deste projeto |
| transformers 5.x | Use 4.56.x (pyproject.toml) |
| Streamlit erro na API | Suba uvicorn na porta 8000 primeiro |
| 422 "Nenhum visor detectado" | Use foto completa, não crop |
