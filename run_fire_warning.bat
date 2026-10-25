@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
set "RUNTIME=%PROJECT_ROOT%.venv\Scripts\python.exe"
if defined FIRE_WARNING_PYTHON set "RUNTIME=%FIRE_WARNING_PYTHON%"
set "PORT=8502"
set "URL=http://127.0.0.1:%PORT%"
set "LOG_DIR=%PROJECT_ROOT%.cache"
set "LOG_FILE=%LOG_DIR%\fire_warning_streamlit.log"
set "ERR_FILE=%LOG_DIR%\fire_warning_streamlit_error.log"

if not exist "%RUNTIME%" (
    echo Fire Early Warning could not start.
    echo Working Python runtime was not found:
    echo   %RUNTIME%
    echo.
    echo Run powershell -ExecutionPolicy Bypass -File setup.ps1
    echo Or set FIRE_WARNING_PYTHON to a configured Python executable.
    pause
    exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

"%RUNTIME%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Python or Streamlit is unavailable. Run setup.ps1 to repair the environment.
    pause
    exit /b 1
)
set "PYTHONUTF8=1"

echo Starting Fire Early Warning at %URL% ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%scripts\start_demo.ps1" -Python "%RUNTIME%" -Port %PORT%
if errorlevel 1 exit /b 1

set "READY=0"
for /l %%I in (1,1,30) do (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ok = $false; try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%URL%'; $ok = ($r.StatusCode -eq 200) } catch {}; if ($ok) { exit 0 } else { exit 1 }"
    if not errorlevel 1 (
        set "READY=1"
        goto :ready
    )
    timeout /t 1 /nobreak >nul
)

:ready
if "%READY%"=="1" (
    start "" "%URL%"
    echo Fire Early Warning is running at %URL%
    exit /b 0
)

echo.
echo Fire Early Warning failed to start or did not answer on port %PORT%.
echo Diagnostic log: %LOG_FILE%
if exist "%LOG_FILE%" type "%LOG_FILE%"
if exist "%ERR_FILE%" type "%ERR_FILE%"
pause
exit /b 1
