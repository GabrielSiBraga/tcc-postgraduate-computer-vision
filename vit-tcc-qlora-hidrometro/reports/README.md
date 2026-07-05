# Reports — Métricas pré-calculadas

Arquivos JSON para consulta **sem GPU**. Regenerados em jul/2026 com adaptador LoRA atual.

| Arquivo | Conteúdo |
|---------|----------|
| `evaluation_test.json` | Métricas completas no split test (**72 amostras**): CER, exact match, acurácia, F1, VRAM, throughput |
| `notebook_comparison_test.json` | Baseline Florence-2 vs QLoRA (**72 amostras**), tabela comparativa em % |
| `label_audit.json` | Consistência inteiro/completo nos labels (**719 registros, 100%**) |
| `label_studio_export_audit.json` | Auditoria do export Label Studio |

## Principais números (`evaluation_test.json`)

| Métrica | Valor |
|---------|-------|
| Parse JSON | 100,0% |
| Exact match | 38,9% |
| Acurácia por caractere | 79,0% |
| Acurácia fabricante | 97,2% |
| Acurácia estado | 97,2% |
| F1 fabricante (macro) | 73,8% |
| F1 estado (macro) | 49,3% |

## Regenerar

```bash
cd vit-tcc-qlora-hidrometro
source .venv/Scripts/activate
PYTHONPATH=src python scripts/05_evaluate.py      # → evaluation_test.json
PYTHONPATH=src python scripts/06_audit_labels.py  # → label_audit.json
```

O notebook `04_resultados_comparativos.ipynb` gera `notebook_comparison_test.json` (seção 9).
