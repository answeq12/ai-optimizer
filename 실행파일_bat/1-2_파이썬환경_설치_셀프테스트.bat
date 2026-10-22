@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Step 1: install the Python environment in WSL and run a self-test (no downloads, about 5 minutes).
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash setup_and_run.sh selftest
echo.
echo Finished. Logs are saved in the logs folder. You can close this window.
pause
