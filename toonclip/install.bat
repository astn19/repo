@echo off
rem One-time setup for Windows + NVIDIA GPU. Needs Python 3.10-3.13 from python.org.
cd /d "%~dp0"
python -m venv venv || goto :err
venv\Scripts\python -m pip install --upgrade pip || goto :err
rem NVIDIA (CUDA) build of PyTorch. If this line fails, get the current command from
rem https://pytorch.org/get-started/locally/ and run it with venv\Scripts\python -m pip ...
venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :err
venv\Scripts\python -m pip install -r requirements.txt || goto :err
venv\Scripts\python cli.py doctor
echo.
echo Setup finished. Double-click run.bat to start.
pause
exit /b 0
:err
echo.
echo Setup failed. Read the message above.
pause
exit /b 1
