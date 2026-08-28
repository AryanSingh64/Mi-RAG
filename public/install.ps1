# install.ps1 — 1-Click Automated Installer for Mi:RAG
Write-Host "=========================================" -ForegroundColor Magenta
Write-Host "  Mi:RAG — Mission RAG Factory Installer " -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Magenta

# 1. Check for Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
    Exit
}

# 2. Check for Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git not found. Please install Git." -ForegroundColor Red
    Exit
}

# 3. Clone or Update
$targetDir = "$HOME\MiRAG"
if (Test-Path $targetDir) {
    Write-Host "Mi:RAG already exists at $targetDir. Updating..." -ForegroundColor Cyan
    cd $targetDir
    git pull
} else {
    Write-Host "Cloning Mi:RAG repository to $targetDir..." -ForegroundColor Cyan
    git clone https://github.com/AryanSingh64/MiRAG.git $targetDir
    cd $targetDir
}

# 4. Launch Factory
Write-Host "Starting Mi:RAG Factory..." -ForegroundColor Green
.\run_factory.bat
