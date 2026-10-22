@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Shows the delete candidates in tools\delete_list.txt and moves them to the Recycle Bin after you type y.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\delete_to_recycle.ps1"
echo.
pause
