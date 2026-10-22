@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Step 4: install Hailo DFC in WSL and run the smoke test. Run step 2 first. May ask for your WSL password once.
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash dfc/setup_and_smoke.sh smoke
echo.
echo Finished. Logs are saved in the logs folder. You can close this window.
pause
