#!/usr/bin/env bash
# Instalação do ambiente vit-tcc-qlora-hidrometro (Linux/macOS/Git Bash)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Criando venv em .venv"
python -m venv .venv
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

echo "==> Atualizando pip"
python -m pip install --upgrade pip wheel setuptools

echo "==> Instalando dependencias do Detectron2"
python -m pip install fvcore iopath pycocotools omegaconf hydra-core cloudpickle matplotlib

echo "==> Instalando PyTorch CUDA (cu128 para Python 3.14)"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==> Instalando Detectron2 (feche Jupyter antes)"
python -m pip install -e ../mask-rcnn/detectron2_repo --no-build-isolation || echo "AVISO: detectron2 falhou - feche notebook e repita"

echo "==> Instalando projeto hidrometro-vlm"
python -m pip install -e ".[dev]"

echo "==> Verificando imports"
python -c "import torch; import detectron2; import cv2; print('OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ""
echo "Instalação concluída. Ative o venv:"
echo "  source .venv/bin/activate   # ou .venv/Scripts/activate no Windows"
