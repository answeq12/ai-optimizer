@echo off
cd /d "%~dp0.."
if not exist logs mkdir logs
wsl.exe -l -v > logs\wsl_list.txt 2>&1
type logs\wsl_list.txt
echo.
echo Saved to logs\wsl_list.txt. You can close this window.
pause
