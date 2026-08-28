# ==============================================================================
#  Mi:RAG — Mission RAG Factory Automated Installer
#  Repository: https://github.com/AryanSingh64/Mi-RAG
# ==============================================================================

$Host.UI.RawUI.WindowTitle = "Mi:RAG — Autonomous Multimodal RAG Engine"
Clear-Host

# ANSI Color Codes
$esc = [char]27
$cReset  = "$esc[0m"
$cBold   = "$esc[1m"
$cPink   = "$esc[38;2;255;45;135m"
$cYellow = "$esc[38;2;255;232;20m"
$cGreen  = "$esc[38;2;43;226;108m"
$cCyan   = "$esc[38;2;0;210;255m"
$cDim    = "$esc[38;2;120;120;140m"

# 1. ASCII Art Logo inspired by Mi:RAG Logo
Write-Host ""
Write-Host "$cPink   ███╗   ███╗██╗██╗   ██╗██████╗  █████╗  ██████╗ $cReset"
Write-Host "$cPink   ████╗ ████║██║██║   ██║██╔══██╗██╔══██╗██╔════╝ $cReset"
Write-Host "$cYellow   ██╔████╔██║██║██║   ██║██████╔╝███████║██║  ███╗$cReset"
Write-Host "$cYellow   ██║╚██╔╝██║██║██║   ██║██╔══██╗██╔══██║██║   ██║$cReset"
Write-Host "$cCyan   ██║ ╚═╝ ██║██║╚██████╔╝██║  ██║██║  ██║╚██████╔╝$cReset"
Write-Host "$cCyan   ╚═╝     ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ $cReset"
Write-Host "$cDim   ------------------------------------------------------------$cReset"
Write-Host "$cBold$cYellow   [ Mission RAG ]$cReset — Autonomous Multimodal RAG-in-a-Box Engine"
Write-Host "$cDim   ------------------------------------------------------------$cReset"
Write-Host ""

function Start-StepAnimation($message) {
    Write-Host -NoNewline "$cCyan[ ⚡ ] $cReset$message "
    Start-Sleep -Milliseconds 250
    Write-Host "$cGreen[ OK ]$cReset"
}

# 2. Prerequisites Verification
Start-StepAnimation "Checking Python 3.10+ installation..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "$cPink[ ✕ ] Python not detected! Please install Python 3.10+ from python.org$cReset" -ForegroundColor Red
    Exit 1
}

Start-StepAnimation "Verifying Git version control..."
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "$cPink[ ✕ ] Git not detected! Please install Git from git-scm.com$cReset" -ForegroundColor Red
    Exit 1
}

# 3. Clone or Update Repository
$targetDir = "$HOME\Mi-RAG"
if (Test-Path $targetDir) {
    Write-Host -NoNewline "$cCyan[ ⚡ ] Updating local Mi:RAG repository... $cReset"
    Set-Location $targetDir
    git pull --quiet
    Write-Host "$cGreen[ UP TO DATE ]$cReset"
} else {
    Write-Host -NoNewline "$cCyan[ ⚡ ] Cloning repository to $targetDir... $cReset"
    git clone --quiet https://github.com/AryanSingh64/Mi-RAG.git $targetDir
    Set-Location $targetDir
    Write-Host "$cGreen[ CLONED ]$cReset"
}

# 4. Virtual Environment & Dependencies Check
if (-not (Test-Path "$targetDir\.venv")) {
    Write-Host -NoNewline "$cCyan[ ⚡ ] Initializing isolated virtual environment (.venv)... $cReset"
    python -m venv .venv
    Write-Host "$cGreen[ CREATED ]$cReset"
    
    Write-Host "$cCyan[ ⚡ ] Installing lightweight factory dependencies (one-time setup)...$cReset"
    & "$targetDir\.venv\Scripts\pip" install --quiet --upgrade pip
    & "$targetDir\.venv\Scripts\pip" install --quiet -r requirements.txt
    Write-Host "$cGreen[ ✓ ] All dependencies installed successfully!$cReset"
}

# 5. Launch Banner
Write-Host ""
Write-Host "$cPink ╔═══════════════════════════════════════════════════════════════════╗$cReset"
Write-Host "$cPink ║$cBold$cYellow   Mi:RAG Studio is launching on http://localhost:8000           $cPink║$cReset"
Write-Host "$cPink ║$cGreen   ✓ 100% Private  •  ✓ Zero API Costs  •  ✓ RTX Accelerated     $cPink║$cReset"
Write-Host "$cPink ╚═══════════════════════════════════════════════════════════════════╝$cReset"
Write-Host ""

# 6. Start Web Factory
if (Test-Path "$targetDir\run_factory.bat") {
    & "$targetDir\run_factory.bat"
} else {
    & "$targetDir\.venv\Scripts\python" "$targetDir\run_factory.py"
}
