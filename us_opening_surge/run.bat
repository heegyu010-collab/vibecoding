@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist .venv (
    echo [setup] creating venv and installing packages...
    python -m venv .venv || (echo Python not found. Install from https://www.python.org - check Add to PATH. & pause & exit /b 1)
    .venv\Scripts\python -m pip install --upgrade pip > nul
    .venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
)

.venv\Scripts\python main.py %*
echo.
for /f "delims=" %%d in ('dir /b /ad /o-n output\run_* 2^>nul') do (
    start "" "output\%%d\report.html"
    goto :done
)
:done
pause
