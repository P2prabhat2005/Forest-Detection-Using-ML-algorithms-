param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 for the saved model environment'"
if ($LASTEXITCODE -ne 0) { throw 'A working Python 3.12 executable is required. Pass -Python with its full path.' }
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $venvPython -m pip install -r requirements-dev.txt -c requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed.' }
& $venvPython -m src.check_setup
if ($LASTEXITCODE -ne 0) { throw 'Required prediction assets are missing; see the checks above.' }
Write-Host 'Setup complete. Run run_fire_warning.bat to open the demo.'
