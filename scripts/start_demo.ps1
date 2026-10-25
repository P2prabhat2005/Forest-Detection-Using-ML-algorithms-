param([Parameter(Mandatory=$true)][string]$Python, [int]$Port = 8502)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectRoot '.cache'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
# Pass the application path as a quoted argument, including on paths with spaces.
$launchArgs = '-m streamlit run "' + (Join-Path $projectRoot 'app.py') + '" --server.headless true --server.address 127.0.0.1 --server.port ' + $Port + ' --browser.gatherUsageStats false'
$listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($listeners) {
    Write-Error "Port $Port is already in use. Close the existing server before launching again."
    exit 1
}
Start-Process -FilePath $Python -ArgumentList $launchArgs -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $logDir 'fire_warning_streamlit.log') -RedirectStandardError (Join-Path $logDir 'fire_warning_streamlit_error.log') -WindowStyle Hidden | Out-Null
