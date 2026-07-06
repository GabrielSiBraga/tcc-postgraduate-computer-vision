# Instalação do ambiente vit-tcc-qlora-hidrometro (Windows PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Criando venv em .venv"
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

Write-Host "==> Atualizando pip"
python -m pip install --upgrade pip wheel setuptools

Write-Host "==> Instalando dependencias do Detectron2"
python -m pip install fvcore iopath pycocotools omegaconf hydra-core cloudpickle matplotlib

Write-Host "==> Instalando PyTorch CUDA (cu128 para Python 3.14)"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "==> Instalando Detectron2 (feche Jupyter/notebooks antes - evita Acesso negado no .pyd)"
python -m pip install -e ..\mask-rcnn\detectron2_repo --no-build-isolation
if ($LASTEXITCODE -ne 0) {
    Write-Host "AVISO: pip install detectron2 falhou (arquivo .pyd bloqueado?)."
    Write-Host "Feche o kernel do notebook mask-rcnn e rode novamente este script."
    Write-Host "Alternativa: o projeto usa sys.path para detectron2_repo - deps acima podem bastar."
}

Write-Host "==> Instalando projeto hidrometro-vlm"
python -m pip install -e ".[dev]"

Write-Host "==> Verificando imports"
python -c "import torch; import detectron2; import cv2; print('OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"

Write-Host ""
Write-Host "Instalação concluída. Ative o venv:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
