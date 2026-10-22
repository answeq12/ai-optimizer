@echo off
chcp 65001 >nul
echo Shut down WSL completely (stops every running experiment and job runner window).
echo Use this when WSL is frozen. After this, start 00_작업대기소_켜기.bat again.
echo.
pause
wsl.exe --shutdown
echo.
echo WSL stopped.
pause
