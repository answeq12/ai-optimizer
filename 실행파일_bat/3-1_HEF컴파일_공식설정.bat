@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Step 5: same as step 4 plus compiling a Hailo-10H HEF (can take a long time).
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash dfc/setup_and_smoke.sh compile
echo.
echo Finished. Logs are saved in the logs folder. You can close this window.
pause
