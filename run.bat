@echo off
cd /d "%~dp0"
call run_factory.bat %*
if %errorlevel% neq 0 pause
