@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo Step 7: compile a Hailo-10H HEF with CIFAR-100 train calibration (64 images, s0). Official alls otherwise unchanged. About 20-25 min. Do NOT run together with other steps.
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash dfc/setup_and_smoke.sh compile --calib-list splits/calib_cifar_64_s0.txt --tag cifar64_s0
echo.
echo Finished. Logs are saved in the logs folder. You can close this window.
pause
