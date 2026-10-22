@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Folder cleanup: moves files only, deletes nothing. See tools\organize.sh
echo.
wsl.exe %DISTRO_ARG% -e bash tools/organize.sh
echo.
pause
