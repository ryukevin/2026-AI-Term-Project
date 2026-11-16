@echo off
rem 13 team - Whisper CER runner. Double-click or run from any folder.
cd /d "%~dp0.."
set PYTHONUTF8=1
python scripts\run_whisper.py %*
pause
