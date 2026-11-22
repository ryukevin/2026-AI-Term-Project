@echo off
rem 13 team - week 2: train normalizer + test-split scores. Double-click or run from any folder.
cd /d "%~dp0.."
set PYTHONUTF8=1
python scripts\run_week2.py %*
pause
