@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Run everything in order: pilot, probes, DFC smoke test. Finished steps are skipped.
echo Do not use this PC while it measures. The DFC step may ask for your WSL password.
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash run_everything.sh
echo.
echo Finished. Logs are saved in the logs folder. You can close this window.
pause
