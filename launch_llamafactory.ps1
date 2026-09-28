# LLaMA Board - LLaMA-Factory Web UI Launcher
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$ScriptDir\LlamaFactory"
$env:DISABLE_VERSION_CHECK = "1"
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Starting LLaMA Board (LLaMA-Factory Web UI)..." -ForegroundColor Green
Write-Host "  URL: http://127.0.0.1:7860" -ForegroundColor Yellow
Write-Host "===================================================" -ForegroundColor Cyan
& "C:\Users\sanath\AppData\Local\Programs\Python\Python311\python.exe" -m llamafactory.cli webui
