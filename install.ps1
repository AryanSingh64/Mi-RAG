# ==============================================================================
#  Mi:RAG — Mission RAG Factory Automated Installer
#  Repository: https://github.com/AryanSingh64/Mi-RAG
# ==============================================================================

$Host.UI.RawUI.WindowTitle = "Mi:RAG - Autonomous Multimodal RAG Engine"
Clear-Host

# 1. High-Impact Bold Red Pure ASCII Logo
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
    Start-Sleep -Milliseconds 120
    Write-Host "[ OK ]" -ForegroundColor Green
}

# 2. Prerequisites Verification: Python 3.10+
Print-Step "Checking Python 3.10+ installation..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host " +-------------------------------------------------------------------------+" -ForegroundColor Red
    Write-Host " |  [!] PREREQUISITE ERROR: Python is NOT detected in your PATH.           |" -ForegroundColor Red
    Write-Host " |  Please download and install Python 3.10+ from https://www.python.org/  |" -ForegroundColor Yellow
    Write-Host " |  (Make sure to check 'Add Python to PATH' during installation)          |" -ForegroundColor White
    Write-Host " +-------------------------------------------------------------------------+" -ForegroundColor Red
    Write-Host ""
    Exit 1
}

# 3. Prerequisites Verification: Ollama AI Engine
Print-Step "Checking Ollama AI Engine installation..."
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host " +-------------------------------------------------------------------------+" -ForegroundColor Red
    Write-Host " |  [!] PREREQUISITE ERROR: Ollama is NOT installed on your machine.       |" -ForegroundColor Red
    Write-Host " |                                                                         |" -ForegroundColor Red
    Write-Host " |  Mi:RAG requires Ollama to run local offline models with zero cost.     |" -ForegroundColor Yellow
    Write-Host " |  1. Download and install Ollama from: https://ollama.com/download       |" -ForegroundColor Cyan
    Write-Host " |  2. After installing, run this install command again in terminal!       |" -ForegroundColor White
    Write-Host " +-------------------------------------------------------------------------+" -ForegroundColor Red
    Write-Host ""
    Exit 1
}

# 4. Check / Auto-Start Ollama Service
Write-Host -NoNewline " [*] Checking Ollama local service... " -ForegroundColor Cyan
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "[ ACTIVE ]" -ForegroundColor Green
} catch {
    Write-Host "[ STARTING SERVICE ]" -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    try {
        $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction SilentlyContinue
        Write-Host " [*] Ollama service connected successfully!" -ForegroundColor Green
    } catch {
        Write-Host " [*] Ollama process launched in background." -ForegroundColor Yellow
    }
}

# 5. Repository Discovery / Directory Setup
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
        Write-Host ""
        Write-Host " [!] Git not detected! Please install Git from https://git-scm.com/" -ForegroundColor Red
        Exit 1
    }
    git clone --quiet https://github.com/AryanSingh64/Mi-RAG.git $targetDir
    Set-Location $targetDir
    Write-Host "[ CLONED ]" -ForegroundColor Green
}

# 6. Virtual Environment Setup
if (-not (Test-Path "$targetDir\.venv\Scripts\python.exe")) {
    Write-Host -NoNewline " [*] Creating virtual environment (.venv)... " -ForegroundColor Cyan
    python -m venv "$targetDir\.venv"
    Write-Host "[ CREATED ]" -ForegroundColor Green
}

# 7. Dependencies Verification & Visual Live Progress Installation
$hasDeps = & "$targetDir\.venv\Scripts\python.exe" -c "import uvicorn, fastapi, fitz, chromadb; print('OK')" 2>$null
if ($hasDeps -ne "OK") {
    Write-Host ""
    Write-Host " [*] Downloading & installing dependencies with live streaming progress:" -ForegroundColor Yellow
    Write-Host " -----------------------------------------------------------------------" -ForegroundColor DarkGray
    
    # Try high-speed UV installer or standard pip with visual progress bar
    & "$targetDir\.venv\Scripts\python.exe" -m pip install --quiet uv 2>$null
    if (Test-Path "$targetDir\.venv\Scripts\uv.exe") {
        & "$targetDir\.venv\Scripts\uv.exe" pip install -r "$targetDir\requirements.txt"
    } else {
        & "$targetDir\.venv\Scripts\python.exe" -m pip install --progress-bar on -r "$targetDir\requirements.txt"
    }
    
    Write-Host " -----------------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host " [OK] All dependencies installed successfully!" -ForegroundColor Green
}

# 8. Automatic Port Freeing & Launch Banner
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

# 9. Direct Native Launch
Set-Location $targetDir
& "$targetDir\.venv\Scripts\python.exe" "$targetDir\run_factory.py"

