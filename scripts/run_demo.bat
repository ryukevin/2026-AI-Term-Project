@echo off
rem 13 team - Gradio demo. Double-click, then use the browser page that opens (http://127.0.0.1:7860).
cd /d "%~dp0.."
set PYTHONUTF8=1
python -c "import gradio" 2>nul || pip install gradio
python app.py %*
pause
