@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [setup] creating venv...
    python -m venv .venv || (echo Python not found. Install from https://www.python.org - check Add to PATH. & pause & exit /b 1)
)

.venv\Scripts\python -c "import pandas, numpy, yfinance, requests, matplotlib, openpyxl" 2>nul || (
    echo [setup] installing packages - this can take a few minutes...
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt || (echo. & echo [error] package install failed. Copy the messages above and send them. & pause & exit /b 1)
)

.venv\Scripts\python main.py %*
echo.
for /f "delims=" %%d in ('dir /b /ad /o-n output\run_* 2^>nul') do (
    start "" "output\%%d\report.html"
    goto :done
)
:done
pause
