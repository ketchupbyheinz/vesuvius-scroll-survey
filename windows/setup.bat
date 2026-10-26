@echo off
rem One-time setup on a Windows PC with an NVIDIA GPU (tested: RTX 3060 Laptop, 6 GB).
rem Installs: Python env with PyTorch (CUDA), villa's vesuvius package, VC3D Windows release.
setlocal enabledelayedexpansion
cd /d "%~dp0\.."
set ROOT=%CD%
if not exist logs mkdir logs
echo Setting up (10-20 minutes). Log: %ROOT%\logs\setup.txt
call :main > "%ROOT%\logs\setup.txt" 2>&1
echo Done. See logs\setup.txt
pause
exit /b

:main
where py || (echo Python launcher not found. Install Python 3.14 from python.org first. & exit /b 1)
if not exist "%ROOT%\villa" git clone --depth 1 https://github.com/ScrollPrize/villa.git "%ROOT%\villa"
if not exist "%ROOT%\env" py -3.14 -m venv "%ROOT%\env"
set PY=%ROOT%\env\Scripts\python.exe
"%PY%" -m pip install --upgrade pip uv
"%PY%" -m uv pip install --python "%PY%" torch torchvision --torch-backend=auto
"%PY%" -m uv pip install --python "%PY%" numpy pillow pynrrd requests pyyaml fsspec "numcodecs<0.17" "zarr>=3" s3fs aiohttp cachetools tifffile imagecodecs einops fvcore timm huggingface_hub tqdm scipy opencv-python-headless safetensors
"%PY%" -m uv pip install --python "%PY%" --no-deps -e "%ROOT%\villa\vesuvius"
"%PY%" -m uv pip install --python "%PY%" --no-deps -e "%ROOT%"
if not exist "%ROOT%\apps\VC3D" (
  mkdir "%ROOT%\apps" 2>nul
  rem newest Windows build of VC3D from the villa "latest" release (tested with 322cb08)
  "%PY%" -c "import json,urllib.request as u;r=json.load(u.urlopen('https://api.github.com/repos/ScrollPrize/villa/releases/tags/latest'));print([a['browser_download_url'] for a in r['assets'] if a['name'].endswith('-win64.zip')][0])" > "%ROOT%\apps\vc3d_url.txt"
  set /p VC3D_URL=<"%ROOT%\apps\vc3d_url.txt"
  curl -L -o "%ROOT%\apps\VC3D-win64.zip" "!VC3D_URL!"
  mkdir "%ROOT%\apps\VC3D"
  tar -xf "%ROOT%\apps\VC3D-win64.zip" -C "%ROOT%\apps\VC3D"
)
"%PY%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"%PY%" -c "import vesuvius.ink_detection.inference.infer, vesuvius_survey.survey; print('imports OK')"
exit /b 0
