@echo off
echo ============================================
echo   Voxis - Diagnostics
echo ============================================
echo.

call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
    echo [FAIL] Virtual environment not found. Run setup.bat first.
    goto end
)
echo [OK] Virtual environment activated
echo.

echo Checking Python packages...
python -c "import numpy; print('[OK] numpy', numpy.__version__)" 2>nul || echo [FAIL] numpy not installed
python -c "import sounddevice; print('[OK] sounddevice', sounddevice.__version__)" 2>nul || echo [FAIL] sounddevice not installed
python -c "import keyboard; print('[OK] keyboard')" 2>nul || echo [FAIL] keyboard not installed
python -c "import pyperclip; print('[OK] pyperclip')" 2>nul || echo [FAIL] pyperclip not installed
python -c "import pystray; print('[OK] pystray')" 2>nul || echo [FAIL] pystray not installed
python -c "from PIL import Image; print('[OK] Pillow')" 2>nul || echo [FAIL] Pillow not installed
python -c "from plyer import notification; print('[OK] plyer')" 2>nul || echo [FAIL] plyer not installed
python -c "import faster_whisper; print('[OK] faster-whisper')" 2>nul || echo [FAIL] faster-whisper not installed
python -c "import playwright; print('[OK] playwright')" 2>nul || echo [FAIL] playwright not installed
echo.

echo Checking microphone...
python -c "import sounddevice; devs = sounddevice.query_devices(); inp = [d for d in devs if d['max_input_channels'] > 0]; print(f'[OK] Found {len(inp)} input device(s):'); [print(f'     - {d[\"name\"]}') for d in inp[:5]]" 2>nul || echo [FAIL] No microphone found or sounddevice error
echo.

echo Checking Playwright browser...
python -c "from playwright.sync_api import sync_playwright; print('[OK] Playwright ready')" 2>nul || echo [FAIL] Playwright not working - run: python -m playwright install chromium
echo.

echo Checking admin privileges (needed for global hotkeys)...
net session >nul 2>&1
if errorlevel 1 (
    echo [WARN] NOT running as administrator - global hotkeys may not work!
    echo        Right-click run.bat ^> Show more options ^> Run as administrator
) else (
    echo [OK] Running as administrator
)
echo.

echo Trying to start Voxis...
echo ============================================
python voxis.py

:end
echo.
pause
