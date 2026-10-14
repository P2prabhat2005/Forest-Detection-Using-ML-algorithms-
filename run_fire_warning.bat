@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
set "RUNTIME=C:\Users\prabh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
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
    echo Update RUNTIME in this launcher after installing or configuring the project runtime.
    pause
    exit /b 1
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

set "PYTHONPATH=%PROJECT_ROOT%.venv\Lib\site-packages;%PROJECT_ROOT%;%PYTHONPATH%"
set "PYTHONUTF8=1"

echo Starting Fire Early Warning at %URL% ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$args=@('-m','streamlit','run','%PROJECT_ROOT%app.py','--server.headless','true','--server.address','127.0.0.1','--server.port','%PORT%','--browser.gatherUsageStats','false'); Start-Process -FilePath '%RUNTIME%' -ArgumentList $args -WorkingDirectory '%PROJECT_ROOT%' -RedirectStandardOutput '%LOG_FILE%' -RedirectStandardError '%ERR_FILE%' -WindowStyle Hidden | Out-Null"

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
