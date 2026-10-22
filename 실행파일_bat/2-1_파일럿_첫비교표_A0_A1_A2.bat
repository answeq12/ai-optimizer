@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Step 2: run the pilot (downloads about 1.3 GB, then measures). Close other programs first.
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash setup_and_run.sh pilot
echo.
echo Finished. Logs are saved in the logs folder. You can close this window.
pause
