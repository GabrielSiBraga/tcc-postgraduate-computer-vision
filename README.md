# TCC — Leitura Automática de Hidrômetros com Visão Computacional

**Repositório:** `tcc-postgraduate-computer-vision` — Trabalho de Conclusão de Curso (Pós-graduação em Visão Computacional).

Pipeline híbrido que combina **detecção de objetos** (Mask R-CNN) e **visão-linguagem** (Florence-2 QLoRA) para ler hidrômetros analógicos a partir de fotos de campo.

A entrega é **este repositório**: código, modelos treinados, notebooks explicativos e métricas documentadas.

---

## Resultados finais

### Fase 1 — Detecção (Mask R-CNN)

| Métrica | Valor | Motivo de reportar |
|---------|-------|--------------------|
| Acurácia de máscara (treino final) | **~98,2%** | Garante localização confiável do visor antes do VLM |

Ver [`mask-rcnn/output/metrics.json`](mask-rcnn/output/metrics.json) e [`mask-rcnn/00_mask_rcnn_hidrometro.ipynb`](mask-rcnn/00_mask_rcnn_hidrometro.ipynb).

### Fase 2 — Leitura VLM (Florence-2 QLoRA, split test n=72)

| Métrica | Baseline (sem LoRA) | Pós-treino QLoRA |
|---------|---------------------|------------------|
| Parse JSON válido | 0% | **100%** |
| Exact match (leitura) | 0% | **38,9%** |
| Acurácia por caractere (1 − CER) | 0% | **79,0%** |
| Acurácia fabricante | 0% | **97,2%** |
| Acurácia estado | 0% | **97,2%** |
| F1 fabricante (macro) | 0% | **73,8%** |
| F1 estado (macro) | 0% | **49,3%** |

Exact match exige string idêntica (zeros à esquerda e vírgula); acurácia por caractere mede proximidade parcial. F1 macro penaliza classes raras (*embacado*: 2 amostras no teste).

Fonte: [`vit-tcc-qlora-hidrometro/reports/evaluation_test.json`](vit-tcc-qlora-hidrometro/reports/evaluation_test.json) (regenerado com `scripts/05_evaluate.py`).

---

## Pipeline — passo a passo

| # | Fase | O que faz | Por quê |
|---|------|-----------|---------|
| 1 | **Mask R-CNN** | Detecta e segmenta hidrômetro na foto bruta | Isola a região de interesse antes da leitura |
| 2 | **Crop + CLAHE** | Expande bbox e aplica contraste adaptativo | Roletes têm baixo contraste e reflexos |
| 3 | **Label Studio** | Revisão humana dos labels JSON | Ground truth confiável para treino |
| 4 | **QLoRA Florence-2** | Fine-tuning com ~500 amostras SFT | Modelo base não gera JSON válido nesta tarefa |
| 5 | **Avaliação** | Métricas no split test (72 crops) | Mede leitura, fabricante e estado |
| 6 | **Demo** | FastAPI + Streamlit | Inferência em foto de campo (opcional) |

Detalhes de scripts, hiperparâmetros e decisões: [`vit-tcc-qlora-hidrometro/docs/GUIA_TCC.md`](vit-tcc-qlora-hidrometro/docs/GUIA_TCC.md).

---

## Mapa do repositório

```
TrabalhoDetecaoObjetos/
├── mask-rcnn/                         # Fase 1 — Mask R-CNN + dataset COCO
│   ├── dataset-hidrometro/
│   ├── output/model_final.pth         # Pesos treinados
│   └── 00_mask_rcnn_hidrometro.ipynb  # Notebook fase 1
│
└── vit-tcc-qlora-hidrometro/          # Fases 2–6 — crops, labels, QLoRA, deploy
    ├── scripts/                       # Pipeline 00–07
    ├── notebooks/                     # Documentação interativa
    ├── artifacts/lora_adapter/        # Adaptador LoRA treinado
    ├── data/sft/                      # Dataset SFT (503/144/72)
    ├── data/label_studio/export/      # Export final Label Studio
    └── reports/                       # Métricas JSON
```

---

## Ordem de leitura (avaliador)

1. **Este README** — visão geral, resultados e pipeline
2. **Notebooks** — processo fase a fase ([índice](vit-tcc-qlora-hidrometro/notebooks/README.md))
3. **[vit-tcc-qlora-hidrometro/README.md](vit-tcc-qlora-hidrometro/README.md)** — instalação, scripts e demo
4. **[docs/GUIA_TCC.md](vit-tcc-qlora-hidrometro/docs/GUIA_TCC.md)** — fluxo completo e decisões técnicas
5. **[docs/MODELOS_E_ARTEFATOS.md](vit-tcc-qlora-hidrometro/docs/MODELOS_E_ARTEFATOS.md)** — o que está no repo vs HuggingFace

### Notebooks

| # | Notebook | Conteúdo |
|---|----------|----------|
| 0 | [00_mask_rcnn_hidrometro.ipynb](mask-rcnn/00_mask_rcnn_hidrometro.ipynb) | Treino e métricas Mask R-CNN |
| 1 | [01_preprocessamento.ipynb](vit-tcc-qlora-hidrometro/notebooks/01_preprocessamento.ipynb) | Crops + CLAHE |
| 2 | [02_dataset_e_labels.ipynb](vit-tcc-qlora-hidrometro/notebooks/02_dataset_e_labels.ipynb) | Dataset SFT e auditoria |
| 3 | [03_treino_qlora.ipynb](vit-tcc-qlora-hidrometro/notebooks/03_treino_qlora.ipynb) | Treino QLoRA |
| 4 | [04_resultados_comparativos.ipynb](vit-tcc-qlora-hidrometro/notebooks/04_resultados_comparativos.ipynb) | Baseline vs QLoRA (%) |

---

## Labels — não é necessário recriar

Anotação feita no **Label Studio** (com autolabeling inicial para acelerar, revisão manual e export). Ground truth em:

| Split | Amostras | Uso |
|-------|----------|-----|
| train | 503 | Treino LoRA |
| val | 144 | Early stopping |
| test | 72 | Métricas finais |

Arquivos: `vit-tcc-qlora-hidrometro/data/sft/*.jsonl`. Auditoria: **100% consistência** (719 registros) — [`reports/label_audit.json`](vit-tcc-qlora-hidrometro/reports/label_audit.json).

---

## Instalação rápida

```bash
cd vit-tcc-qlora-hidrometro
bash scripts/setup_env.sh
source .venv/Scripts/activate   # Windows Git Bash
pip install -e ".[dev]"
```

Detectron2 e pesos Mask R-CNN: [`mask-rcnn/README.md`](mask-rcnn/README.md).

> **Git LFS:** `model_final.pth` (~335 MB) pode exigir Git LFS no GitHub. Ver [`.gitattributes`](.gitattributes).

---

## Reproduzir métricas (sem retreinar)

```bash
cd vit-tcc-qlora-hidrometro
source .venv/Scripts/activate
PYTHONPATH=src python scripts/05_evaluate.py
PYTHONPATH=src python scripts/06_audit_labels.py
```

Ou abrir `04_resultados_comparativos.ipynb` (kernel `Python 3 (vit-tcc-qlora)`).

---

## Demo — inferência em foto de campo

**Terminal 1 — API** (carrega Detectron2 + Florence-2; ~20–30 s no primeiro start):

```bash
cd vit-tcc-qlora-hidrometro
source .venv/Scripts/activate
PYTHONPATH=src uvicorn hidrometro.api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit** (interface; depende da API em `localhost:8000`):

```bash
cd vit-tcc-qlora-hidrometro
source .venv/Scripts/activate
streamlit run src/hidrometro/ui/streamlit_app.py
```

Use **fotos completas** do hidrômetro (não crops). Resposta: JSON com leitura, fabricante, estado, overlay e crop CLAHE.

Teste via curl: `curl -X POST http://localhost:8000/predict -F "file=@foto.jpg"`

---

## Checklist de entrega

- [x] README raiz com objetivo, resultados % e pipeline
- [x] 5 notebooks documentando cada fase
- [x] `model_final.pth` (Detectron2) incluído
- [x] `lora_adapter/` (Florence-2 QLoRA) incluído
- [x] `reports/*.json` com métricas atualizadas
- [x] Scripts 00–07 reproduzíveis
- [x] FastAPI / Streamlit documentados
