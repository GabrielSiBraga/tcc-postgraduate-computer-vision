# TCC - Leitura Automática de Hidrômetros com Visão Computacional

#### Alun(o/a): [Gabriel Silva Braga](https://github.com/GabrielSiBraga).
#### Orientador(/a/es/as): [Vitor Bento de Sousa](https://github.com/link_do_github).

---
Trabalho apresentado ao curso [CV Master](https://ccec.puc-rio.br/site/Folder?nCurso=visao-computacional%3A-interpretando-o-mundo-atraves-de-imagens-(traco)-computer-vision-master&nInst=CCE) como pré-requisito para conclusão de curso.

**Repositório:** `tcc-postgraduate-computer-vision` - Trabalho de Conclusão de Curso (Pós-graduação em Visão Computacional).
- [Link para o código](https://github.com/GabrielSiBraga/tcc-postgraduate-computer-vision).

---

## Resumo do projeto

Este trabalho propõe a **automatização da leitura de hidrômetros analógicos** a partir de fotos capturadas em campo, problema tradicionalmente resolvido por inspeção manual, lenta e sujeita a erros de transcrição. A solução adotada é um **pipeline híbrido** que combina detecção de objetos com Mask R-CNN (localização e segmentação do visor) e um modelo de visão-linguagem Florence-2 fine-tuned via QLoRA (extração estruturada da leitura em JSON).

Os resultados demonstram viabilidade técnica em ambas as etapas: o detector atinge **~98,2%** de acurácia de máscara, garantindo isolamento confiável da região de interesse; o VLM, avaliado no split de teste (n=72), alcança **100%** de parse JSON válido, **38,9%** de exact match na leitura, **79,0%** de acurácia por caractere e **97,2%** de acurácia em fabricante e estado. A entrega consiste neste repositório com código reproduzível, modelos treinados, notebooks explicativos, métricas documentadas e demo opcional (FastAPI + Streamlit).

---

## Abstract

This work proposes the **automation of analog water meter reading** from field-captured photographs, a task traditionally performed through manual inspection that is slow and prone to transcription errors. The adopted solution is a **hybrid pipeline** combining object detection with Mask R-CNN (localization and segmentation of the display) and a vision-language model, Florence-2, fine-tuned via QLoRA (structured extraction of the reading as JSON).

Results demonstrate technical feasibility at both stages: the detector achieves **~98.2%** mask accuracy, ensuring reliable isolation of the region of interest; the VLM, evaluated on the test split (n=72), reaches **100%** valid JSON parsing, **38.9%** exact match on the reading, **79.0%** character-level accuracy, and **97.2%** accuracy on manufacturer and physical state. The deliverable consists of this repository with reproducible code, trained models, explanatory notebooks, documented metrics, and an optional demo (FastAPI + Streamlit).

---

## Introdução

Hidrômetros analógicos permanecem amplamente utilizados em redes de abastecimento de água. A leitura periódica dos medidores é essencial para faturamento e gestão de consumo, mas o processo manual - fotografar o equipamento, identificar o visor e transcrever os dígitos - é trabalhoso e propenso a inconsistências.

As imagens de campo apresentam desafios visuais significativos: **reflexos** na superfície do visor, **baixo contraste** entre roletes pretos e vermelhos, variação de **fabricantes** (placas e tipografias distintas) e diferentes **estados físicos** do equipamento (*normal*, *embacado*, *trincado*, *sujo*, *anomalia*). Essas condições dificultam a aplicação direta de OCR genérico ou de modelos de visão-linguagem sem adaptação à tarefa.

Para contornar essas limitações, adotou-se uma abordagem em **duas etapas**: primeiro, um modelo de detecção e segmentação (Mask R-CNN) localiza o visor na foto completa; em seguida, um VLM (Florence-2) especializado via QLoRA interpreta os roletes e retorna um JSON estruturado com leitura numérica, fabricante e estado. Este documento descreve o projeto entregue como Trabalho de Conclusão de Curso em Visão Computacional.

---

## Objetivo

Os objetivos deste trabalho são:

1. **Automatizar a leitura de hidrômetros analógicos** a partir de fotografias de campo, eliminando a transcrição manual.
2. **Detectar e segmentar a região do visor** utilizando Mask R-CNN (Detectron2, backbone R50-FPN), isolando a área de interesse antes da interpretação.
3. **Extrair leitura numérica, fabricante e estado** por meio de fine-tuning QLoRA do Florence-2-large, produzindo saída JSON estruturada.
4. **Avaliar quantitativamente** o desempenho no split de teste (72 amostras, nunca utilizadas no treino) e disponibilizar uma **demo reproduzível** para inferência em fotos completas.

---

## Resultados finais

### Fase 1 - Detecção (Mask R-CNN)

| Métrica | Valor | Motivo de reportar |
|---------|-------|--------------------|
| Acurácia de máscara (treino final) | **~98,2%** | Garante localização confiável do visor antes do VLM |

Ver [`mask-rcnn/output/metrics.json`](mask-rcnn/output/metrics.json) e [`mask-rcnn/00_mask_rcnn_hidrometro.ipynb`](mask-rcnn/00_mask_rcnn_hidrometro.ipynb).

### Fase 2 - Leitura VLM (Florence-2 QLoRA, split test n=72)

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

## Pipeline - passo a passo

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

## Banco de dados

O projeto utiliza **dois conjuntos de dados complementares**: um para detecção (formato COCO) e outro para leitura via VLM (formato SFT/JSONL).

### Camada 1 - Detecção (COCO)

| Atributo | Detalhe |
|----------|---------|
| Raiz | [`mask-rcnn/dataset-hidrometro/`](mask-rcnn/dataset-hidrometro/) |
| Configuração | [`vit-tcc-qlora-hidrometro/configs/paths.yaml`](vit-tcc-qlora-hidrometro/configs/paths.yaml) |
| Formato | COCO (`_annotations.coco.json`) |
| Classe | `display` (id=1) - visor do hidrômetro |
| Splits | `train`, `valid`, `test` |
| Origem | Export Roboflow (CC BY 4.0) |
| No repositório | Anotações de treino presentes; pesos finais em `mask-rcnn/output/model_final.pth` |

O Mask R-CNN foi treinado sobre este dataset para localizar e segmentar o visor nas fotos brutas. Os pesos treinados (`model_final.pth`, ~335 MB) estão incluídos na entrega.

### Camada 2 - Leitura VLM (SFT)

Após a detecção, cada imagem passa por expansão de bounding box e realce de contraste (CLAHE), gerando crops em `data/crops/{split}/`. Sobre esses crops foram criadas **719 anotações** revisadas manualmente, divididas nos splits abaixo:

| Split | Amostras | Proporção | Uso |
|-------|----------|-----------|-----|
| **Treino** | 503 | ~70% | Fine-tuning LoRA |
| **Validação** | 144 | ~20% | Early stopping / `eval_loss` |
| **Teste** | 72 | ~10% | Métricas finais (nunca usado no treino) |

Fonte dos splits: [`vit-tcc-qlora-hidrometro/reports/label_studio_export_audit.json`](vit-tcc-qlora-hidrometro/reports/label_studio_export_audit.json).

**Arquivos do dataset SFT:**

| Caminho | Conteúdo |
|---------|----------|
| `vit-tcc-qlora-hidrometro/data/sft/{train,val,test}.jsonl` | Dataset no formato SFT para treino |
| `vit-tcc-qlora-hidrometro/data/autolabel/validated/{split}/` | Ground truth validado |
| `vit-tcc-qlora-hidrometro/data/label_studio/export/` | Export final do Label Studio |
| `vit-tcc-qlora-hidrometro/data/crops/{split}/` | Imagens recortadas com CLAHE |

**Schema de cada amostra** (definido em [`configs/autolabel.yaml`](vit-tcc-qlora-hidrometro/configs/autolabel.yaml)):

```json
{
  "leitura": { "inteiro": 302, "decimal": 21, "completo": "0302,21" },
  "fabricante": "SAGA",
  "estado": "normal|embacado|trincado|sujo|anomalia"
}
```

- `inteiro`: valor numérico dos roletes pretos (sem zeros à esquerda)
- `decimal`: valor numérico dos roletes vermelhos
- `completo`: transcrição literal do visor (ex.: `"0302,21"`)
- `fabricante`: marca/placa identificada no hidrômetro
- `estado`: condição física do visor

**Fluxo de preparação dos dados:**

1. Foto bruta de campo → Mask R-CNN → bounding box do visor
2. Expansão de crop + CLAHE → `data/crops/{split}/`
3. Autolabel inicial (GPT-4o, etapa histórica) + revisão humana no Label Studio
4. Export → validação → conversão para `data/sft/*.jsonl`

**Qualidade:** auditoria com **100% de consistência** (719 registros, 0 issues) - [`vit-tcc-qlora-hidrometro/reports/label_audit.json`](vit-tcc-qlora-hidrometro/reports/label_audit.json).

> **Nota de entrega:** as anotações já estão prontas e validadas. Não é necessário recriar labels nem executar novamente o fluxo de Label Studio.

---

## Mapa do repositório

```
TrabalhoDetecaoObjetos/
├── mask-rcnn/                         # Fase 1 - Mask R-CNN + dataset COCO
│   ├── dataset-hidrometro/
│   ├── output/model_final.pth         # Pesos treinados
│   └── 00_mask_rcnn_hidrometro.ipynb  # Notebook fase 1
│
└── vit-tcc-qlora-hidrometro/          # Fases 2–6 - crops, labels, QLoRA, deploy
    ├── scripts/                       # Pipeline 00–07
    ├── notebooks/                     # Documentação interativa
    ├── artifacts/lora_adapter/        # Adaptador LoRA treinado
    ├── data/sft/                      # Dataset SFT (503/144/72)
    ├── data/label_studio/export/      # Export final Label Studio
    └── reports/                       # Métricas JSON
```

---

## Ordem de leitura (avaliador)

1. **Este README** - visão geral, resultados e pipeline
2. **Notebooks** - processo fase a fase ([índice](vit-tcc-qlora-hidrometro/notebooks/README.md))
3. **[vit-tcc-qlora-hidrometro/README.md](vit-tcc-qlora-hidrometro/README.md)** - instalação, scripts e demo
4. **[docs/GUIA_TCC.md](vit-tcc-qlora-hidrometro/docs/GUIA_TCC.md)** - fluxo completo e decisões técnicas
5. **[docs/MODELOS_E_ARTEFATOS.md](vit-tcc-qlora-hidrometro/docs/MODELOS_E_ARTEFATOS.md)** - o que está no repo vs HuggingFace

### Notebooks

| # | Notebook | Conteúdo |
|---|----------|----------|
| 0 | [00_mask_rcnn_hidrometro.ipynb](mask-rcnn/00_mask_rcnn_hidrometro.ipynb) | Treino e métricas Mask R-CNN |
| 1 | [01_preprocessamento.ipynb](vit-tcc-qlora-hidrometro/notebooks/01_preprocessamento.ipynb) | Crops + CLAHE |
| 2 | [02_dataset_e_labels.ipynb](vit-tcc-qlora-hidrometro/notebooks/02_dataset_e_labels.ipynb) | Dataset SFT e auditoria |
| 3 | [03_treino_qlora.ipynb](vit-tcc-qlora-hidrometro/notebooks/03_treino_qlora.ipynb) | Treino QLoRA |
| 4 | [04_resultados_comparativos.ipynb](vit-tcc-qlora-hidrometro/notebooks/04_resultados_comparativos.ipynb) | Baseline vs QLoRA (%) |

---

## Conclusão e próximos passos

### Conclusão

O pipeline híbrido proposto demonstrou ser **tecnicamente viável** para leitura automática de hidrômetros analógicos. A etapa de detecção (Mask R-CNN) alcança acurácia de máscara de ~98,2%, garantindo que o VLM receba crops adequados do visor. O fine-tuning QLoRA do Florence-2 transformou um modelo base incapaz de gerar JSON válido (0%) em um sistema que produz respostas estruturadas corretas em **100%** das amostras de teste.

A transcrição literal da leitura permanece desafiadora: o exact match de 38,9% reflete a exigência de correspondência exata (zeros à esquerda, vírgula), enquanto a acurácia por caractere de 79,0% indica que a maioria das predições está parcialmente correta. A classificação de fabricante e estado é robusta (97,2% de acurácia), embora o F1 macro seja limitado por classes raras no conjunto de teste.

### Próximos passos

1. **Ampliar o dataset**, com foco em classes sub-representadas (*embacado*, marcas minoritárias) para melhorar generalização.
2. **Melhorar o exact match** da leitura via pós-processamento de dígitos, data augmentation direcionada ou modelo OCR dedicado para os roletes.
3. **Reduzir a dependência do Detectron2**, explorando detectores mais leves ou abordagens end-to-end que eliminem a etapa intermediária.
4. **Otimizar a inferência para produção**, reduzindo latência e viabilizando deploy em dispositivos edge.

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

## Demo - inferência em foto de campo

**Terminal 1 - API** (carrega Detectron2 + Florence-2; ~20–30 s no primeiro start):

```bash
cd vit-tcc-qlora-hidrometro
source .venv/Scripts/activate
PYTHONPATH=src uvicorn hidrometro.api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Streamlit** (interface; depende da API em `localhost:8000`):

```bash
cd vit-tcc-qlora-hidrometro
source .venv/Scripts/activate
streamlit run src/hidrometro/ui/streamlit_app.py
```

Use **fotos completas** do hidrômetro (não crops). Resposta: JSON com leitura, fabricante, estado, overlay e crop CLAHE.

Teste via curl: `curl -X POST http://localhost:8000/predict -F "file=@foto.jpg"`

---

## Checklist de entrega

- [x] README raiz com resumo, abstract, introdução, objetivo, banco de dados, conclusão, resultados % e pipeline
- [x] 5 notebooks documentando cada fase
- [x] `model_final.pth` (Detectron2) incluído
- [x] `lora_adapter/` (Florence-2 QLoRA) incluído
- [x] `reports/*.json` com métricas atualizadas
- [x] Scripts 00–07 reproduzíveis
- [x] FastAPI / Streamlit documentados

---

Matrícula: 232.100.436

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Computer Vision Master*