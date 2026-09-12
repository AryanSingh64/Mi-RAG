@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Mi:RAG Engine Factory

echo =========================================================
echo   Mi:RAG Engine Factory
echo =========================================================

if exist ".venv\Scripts\python.exe" (
    echo [*] Launching using local environment (.venv)...
    .venv\Scripts\python.exe run_factory.py %*
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        python run_factory.py %*
    ) else (
        where py >nul 2>&1
        if !errorlevel! equ 0 (
            py run_factory.py %*
        ) else (
            echo [!] Python 3.10+ was not found on your system PATH.
        )
    )
)

echo.
echo =========================================================
echo  [i] Mi:RAG Factory process finished.
echo =========================================================
pause
