@echo off
title LLaMA Board - LLaMA-Factory Web UI
cd /d "%~dp0LlamaFactory"
set DISABLE_VERSION_CHECK=1
echo ===================================================
echo   Starting LLaMA Board (LLaMA-Factory Web UI)...
echo   URL: http://127.0.0.1:7860
echo ===================================================
"C:\Users\sanath\AppData\Local\Programs\Python\Python311\python.exe" -m llamafactory.cli webui
pause
