@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Step 8: layer sensitivity probe on A2m (8 runs: 5 by kind + 3 by depth). About 40-50 min. Do NOT run while DFC compile is running.
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash setup_and_run.sh sens
echo.
echo Finished. Logs are saved in the logs folder. You can close this window.
pause
