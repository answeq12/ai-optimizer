@echo off
chcp 65001 >nul
title halio job runner - KEEP THIS OPEN
cd /d "%~dp0"
set "DISTRO_ARG="
if exist wsl_distro.txt (
  set /p DISTRO=<wsl_distro.txt
  call set "DISTRO_ARG=-d %%DISTRO%%"
)
echo This window lets Claude run experiments without you clicking anything.
echo Keep it open. Minimizing is fine. Jobs run one at a time.
if defined DISTRO_ARG echo Using WSL distro: %DISTRO%
echo.
wsl.exe %DISTRO_ARG% -e bash job_runner.sh
echo.
echo Job runner stopped.
pause
