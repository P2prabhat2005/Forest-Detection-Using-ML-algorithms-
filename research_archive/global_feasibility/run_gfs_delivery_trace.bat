@echo off
setlocal EnableExtensions
set "TRACE_DIR=%~dp0"
set "TRACE_PYTHON=C:\Users\prabh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%TRACE_PYTHON%" (
  echo Python runtime not found: %TRACE_PYTHON%
  echo Update TRACE_PYTHON in this launcher to a Python 3.12 installation.
  pause
  exit /b 1
)
cd /d "%TRACE_DIR%"
if /I "%~1"=="--self-test" (
  "%TRACE_PYTHON%" "%TRACE_DIR%gfs_delivery_trace.py" --self-test
  if errorlevel 1 exit /b 1
  exit /b 0
)
echo Starting the next eligible GFS 06Z delivery trace.
echo Keep this window open until the report appears. Press Ctrl+C to stop early.
"%TRACE_PYTHON%" "%TRACE_DIR%gfs_delivery_trace.py" --next-eligible
set "TRACE_EXIT=%ERRORLEVEL%"
echo.
echo Trace finished with exit code %TRACE_EXIT%.
echo Reports are under %TRACE_DIR%gfs_delivery_traces\
pause
exit /b %TRACE_EXIT%
