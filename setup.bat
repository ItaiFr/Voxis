@echo off
echo ============================================
echo   Voxis - Setup
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Download from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo [3/3] Installing browser engine...
python -m playwright install chromium

echo.
echo ============================================
echo   Setup complete! Run "run.bat" to start.
echo ============================================
pause
