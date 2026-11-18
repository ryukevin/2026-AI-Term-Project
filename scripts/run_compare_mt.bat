@echo off
rem 13 team - MT model comparison (NLLB vs MADLAD). Double-click or run from any folder.
cd /d "%~dp0.."
set PYTHONUTF8=1
git checkout main
if errorlevel 1 goto git_failed
git pull
if errorlevel 1 goto git_failed
python scripts\run_compare_mt.py %*
goto end

:git_failed
echo.
echo [X] git step failed. Nothing was run. Copy the message above and send it.

:end
pause
