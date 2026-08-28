# ==============================================================================
#  Mi:RAG — Mission RAG Factory Automated Installer
#  Repository: https://github.com/AryanSingh64/Mi-RAG
# ==============================================================================

$Host.UI.RawUI.WindowTitle = "Mi:RAG - Autonomous Multimodal RAG Engine"
Clear-Host

# 1. High-Impact Bold Red Pure ASCII Logo (Universal Compatibility)
Write-Host ""
Write-Host "  __  __ _       ____      _    ____  " -ForegroundColor Red
Write-Host " |  \/  (_)     |  _ \    / \  / ___| " -ForegroundColor Red
Write-Host " | |\/| | |  _  | |_) |  / _ \| |  _  " -ForegroundColor Red
Write-Host " | |  | | | (_) |  _ <  / ___ \ |_| | " -ForegroundColor Red
Write-Host " |_|  |_|_|     |_| \_\/_/   \_\____| " -ForegroundColor Red
Write-Host ""
Write-Host " ===========================================================" -ForegroundColor DarkGray
Write-Host "   [ MISSION RAG ] - Autonomous Multimodal RAG Engine       " -ForegroundColor Yellow
Write-Host " ===========================================================" -ForegroundColor DarkGray
Write-Host ""

function Print-Step($msg) {
    Write-Host -NoNewline " [*] $msg " -ForegroundColor Cyan
    Start-Sleep -Milliseconds 150
    Write-Host "[ OK ]" -ForegroundColor Green
}

# 2. Prerequisites Verification
Print-Step "Checking Python 3.10+ installation..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host " [!] Python not detected! Please install Python 3.10+ from python.org" -ForegroundColor Red
    Exit 1
}

# 3. Repository Discovery / Offline Safe Check
$targetDir = "$HOME\Mi-RAG"
if (Test-Path "$targetDir\run_factory.py") {
    Write-Host -NoNewline " [*] Checking for updates... " -ForegroundColor Cyan
    Set-Location $targetDir
    try {
        $gitOutput = git pull --quiet 2>&1
        Write-Host "[ UP TO DATE ]" -ForegroundColor Green
    } catch {
        Write-Host "[ OFFLINE MODE ]" -ForegroundColor Yellow
    }
} elseif (Test-Path ".\run_factory.py") {
    $targetDir = (Get-Location).Path
    Write-Host " [*] Running from local repository at $targetDir" -ForegroundColor Cyan
} else {
    Write-Host -NoNewline " [*] Cloning repository to $targetDir... " -ForegroundColor Cyan
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host " [!] Git not detected! Please install Git from git-scm.com" -ForegroundColor Red
        Exit 1
    }
    git clone --quiet https://github.com/AryanSingh64/Mi-RAG.git $targetDir
    Set-Location $targetDir
    Write-Host "[ CLONED ]" -ForegroundColor Green
}

# 4. Virtual Environment & Accelerated Parallel Dependencies Check
if (-not (Test-Path "$targetDir\.venv\Scripts\python.exe")) {
    Write-Host -NoNewline " [*] Creating virtual environment (.venv)... " -ForegroundColor Cyan
    python -m venv "$targetDir\.venv"
    Write-Host "[ CREATED ]" -ForegroundColor Green
}

# Verify packages exist in .venv
$hasUvicorn = & "$targetDir\.venv\Scripts\python.exe" -c "import uvicorn, fastapi; print('OK')" 2>$null
if ($hasUvicorn -ne "OK") {
    Write-Host ""
    Write-Host " [*] Accelerating installer with parallel package engine (uv)..." -ForegroundColor Cyan
    & "$targetDir\.venv\Scripts\python.exe" -m pip install --quiet uv 2>$null
    
    Write-Host " [*] Installing packages with parallel streaming & live progress:" -ForegroundColor Yellow
    Write-Host " ------------------------------------------------------------" -ForegroundColor DarkGray
    
    if (Test-Path "$targetDir\.venv\Scripts\uv.exe") {
        & "$targetDir\.venv\Scripts\uv.exe" pip install -r "$targetDir\requirements.txt"
    } else {
        & "$targetDir\.venv\Scripts\python.exe" -m pip install --progress-bar on -r "$targetDir\requirements.txt"
    }
    
    Write-Host " ------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host " [OK] All dependencies installed successfully!" -ForegroundColor Green
}

# 5. Launch Banner & Automatic Port Freeing
try {
    $connections = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        if ($conn.OwningProcess -and $conn.OwningProcess -ne $PID) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
} catch {}

Write-Host ""
Write-Host " +---------------------------------------------------------+" -ForegroundColor Red
Write-Host " |  Mi:RAG Studio is launching on http://localhost:8000    |" -ForegroundColor Yellow
Write-Host " |  100% Private  |  Zero API Costs  |  Hardware Accelerated|" -ForegroundColor White
Write-Host " +---------------------------------------------------------+" -ForegroundColor Red
Write-Host ""

# 6. Direct Native Launch via Python
Set-Location $targetDir
& "$targetDir\.venv\Scripts\python.exe" "$targetDir\run_factory.py"
