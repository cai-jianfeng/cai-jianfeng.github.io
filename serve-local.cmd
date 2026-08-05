@echo off
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve-local.ps1"
exit /b %errorlevel%
