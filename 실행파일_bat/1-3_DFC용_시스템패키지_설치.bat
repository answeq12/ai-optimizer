@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Install the system packages that Hailo DFC needs, as the WSL root user (no password needed).
echo Packages: graphviz libgraphviz-dev build-essential python3-dev
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -u root -- bash -c "apt-get update && apt-get install -y graphviz libgraphviz-dev build-essential python3-dev && echo && echo == DONE: system packages installed"
echo.
echo Finished. You can close this window.
pause
