# Modelos e Artefatos - O que está no repositório

## Incluídos no Git (entrega TCC)

| Artefato | Caminho | Tamanho aprox. | Descrição |
|----------|---------|----------------|-----------|
| Mask R-CNN | `mask-rcnn/output/model_final.pth` | ~335 MB | Detecção/segmentação (~98,2% acurácia máscara) |
| Adaptador LoRA | `vit-tcc-qlora-hidrometro/artifacts/lora_adapter/` | ~81 MB | Fine-tuning Florence-2 para JSON |
| Dataset SFT | `vit-tcc-qlora-hidrometro/data/sft/*.jsonl` | <1 MB | 503/144/72 amostras |
| Export Label Studio | `vit-tcc-qlora-hidrometro/data/label_studio/export/` | ~1 MB | Anotações finais (719 tasks) |
| Labels validados | `vit-tcc-qlora-hidrometro/data/autolabel/validated/` | variável | Ground truth parseado do export LS |
| Métricas | `vit-tcc-qlora-hidrometro/reports/*.json` | <1 MB | Avaliação split test (72 amostras) |
| Crops | `vit-tcc-qlora-hidrometro/data/crops/` | variável | Imagens pós-Detectron2 + CLAHE |

### Métricas de referência (`evaluation_test.json`)

Parse **100%** · Exact match **38,9%** · Acurácia por caractere **79,0%** · Fabricante/estado **97,2%**

## Baixados automaticamente (não versionados)

| Modelo | Origem | Quando |
|--------|--------|--------|
| Florence-2-large (base) | HuggingFace `microsoft/Florence-2-large` | 1ª inferência ou treino |
| Cache transformers | `~/.cache/huggingface/` | Download automático |

Na primeira execução com internet, o Florence-2-base é baixado (~1,5 GB). O adaptador LoRA sobrescreve apenas os pesos de atenção.

## Checkpoints intermediários (ignorados)

- `artifacts/lora_adapter/checkpoint-*` - excluídos do Git; só o adaptador final na raiz de `lora_adapter/` é entregue

## Reproduzir sem retreinar

```bash
cd vit-tcc-qlora-hidrometro
source .venv/Scripts/activate
PYTHONPATH=src python scripts/05_evaluate.py
PYTHONPATH=src python scripts/06_audit_labels.py
```

Usa `artifacts/lora_adapter/` + `mask-rcnn/output/model_final.pth` já presentes.

## Reproduzir treino (opcional)

```bash
PYTHONPATH=src python scripts/04_train_qlora.py
```

Requer GPU CUDA, ~40 min para 8 épocas. Labels em `data/sft/` - **não recrie labels**.

## Demo

```bash
PYTHONPATH=src uvicorn hidrometro.api.main:app --host 0.0.0.0 --port 8000
streamlit run src/hidrometro/ui/streamlit_app.py
```

## Git LFS

Se `model_final.pth` ou LoRA exceder 100 MB por arquivo no GitHub, configure Git LFS:

```bash
git lfs track "*.pth"
git lfs track "artifacts/lora_adapter/**"
```

Ou documente link externo (Drive/Zenodo) no README raiz.
