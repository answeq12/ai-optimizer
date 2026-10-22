@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Step 6: re-run static INT8 limited to weight MatMuls (same targets as A1). Do NOT run while DFC compile is running.
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash setup_and_run.sh a2fix
echo.
echo Finished. Logs are saved in the logs folder. You can close this window.
pause
