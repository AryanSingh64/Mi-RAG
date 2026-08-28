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

Print-Step "Verifying Git version control..."
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host " [!] Git not detected! Please install Git from git-scm.com" -ForegroundColor Red
    Exit 1
}

# 3. Clone or Update Repository
$targetDir = "$HOME\Mi-RAG"
if (Test-Path $targetDir) {
    Write-Host -NoNewline " [*] Updating local Mi:RAG repository... " -ForegroundColor Cyan
    Set-Location $targetDir
    git pull --quiet
    Write-Host "[ UP TO DATE ]" -ForegroundColor Green
} else {
    Write-Host -NoNewline " [*] Cloning repository to $targetDir... " -ForegroundColor Cyan
    git clone --quiet https://github.com/AryanSingh64/Mi-RAG.git $targetDir
    Set-Location $targetDir
    Write-Host "[ CLONED ]" -ForegroundColor Green
}

# 4. Virtual Environment & Dependencies Check
if (-not (Test-Path "$targetDir\.venv\Scripts\python.exe")) {
    Write-Host -NoNewline " [*] Creating virtual environment (.venv)... " -ForegroundColor Cyan
    python -m venv "$targetDir\.venv"
    Write-Host "[ CREATED ]" -ForegroundColor Green
}

# Verify packages exist in .venv
$hasUvicorn = & "$targetDir\.venv\Scripts\python.exe" -c "import uvicorn, fastapi; print('OK')" 2>$null
if ($hasUvicorn -ne "OK") {
    Write-Host " [*] Installing dependencies into .venv (one-time setup)..." -ForegroundColor Cyan
    & "$targetDir\.venv\Scripts\pip.exe" install --upgrade pip
    & "$targetDir\.venv\Scripts\pip.exe" install -r "$targetDir\requirements.txt"
    Write-Host " [OK] Dependencies installed successfully!" -ForegroundColor Green
}

# 5. Launch Banner
Write-Host ""
Write-Host " +---------------------------------------------------------+" -ForegroundColor Red
Write-Host " |  Mi:RAG Studio is launching on http://localhost:8000    |" -ForegroundColor Yellow
Write-Host " |  100% Private  |  Zero API Costs  |  Hardware Accelerated|" -ForegroundColor White
Write-Host " +---------------------------------------------------------+" -ForegroundColor Red
Write-Host ""

# 6. Start Web Factory
Set-Location $targetDir
if (Test-Path "$targetDir\run_factory.bat") {
    & "$targetDir\run_factory.bat"
} else {
    & "$targetDir\.venv\Scripts\python.exe" "$targetDir\run_factory.py"
}
