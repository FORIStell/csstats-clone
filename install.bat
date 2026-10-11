@echo off
REM One-time setup: creates a venv and installs PyTorch (CUDA) + dependencies.
python -m venv venv || goto :err
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121 || goto :err
pip install -r requirements.txt || goto :err
echo.
echo Done. Run run.bat to start.
pause
exit /b 0
:err
echo Install failed. Make sure Python 3.10-3.12 is installed and on PATH.
pause
exit /b 1
