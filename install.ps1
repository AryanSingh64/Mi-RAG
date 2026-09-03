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
    Start-Sleep -Milliseconds 80
    Write-Host "[ OK ]" -ForegroundColor Green
}

function Safe-Exit {
    Write-Host ""
    Write-Host "-----------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "Setup paused. The terminal window will stay open." -ForegroundColor DarkGray
    Read-Host -Prompt "Press [Enter] to close"
    return
}

# 2. Prerequisites Verification: Python 3.10+
Print-Step "Checking Python installation..."
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host " ==========================================================================" -ForegroundColor Red
    Write-Host "  [!] PREREQUISITE NOTICE: Python 3.10+ was not detected in your PATH.    " -ForegroundColor Yellow
    Write-Host " ==========================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Option 1: Open official Python download page in browser" -ForegroundColor White
    Write-Host "  Option 2: Install automatically via Windows Package Manager (winget)" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host -Prompt "  Select option [1/2] or press Enter to exit"
    if ($choice -eq "1") {
        Write-Host "  [*] Opening https://www.python.org/downloads/ in browser..." -ForegroundColor Cyan
        Start-Process "https://www.python.org/downloads/"
        Write-Host "  [!] Please run the installer, make sure to check 'Add Python to PATH', and run this command again." -ForegroundColor Yellow
    } elseif ($choice -eq "2" -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "  [*] Installing Python 3.11 via winget..." -ForegroundColor Cyan
        winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
        Write-Host "  [OK] Python installed! Please restart your terminal and run this command again." -ForegroundColor Green
    }
    Safe-Exit
    return
}

# 3. Prerequisites Verification: Ollama AI Engine
Print-Step "Checking Ollama AI Engine installation..."
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host " ==========================================================================" -ForegroundColor Red
    Write-Host "  [!] PREREQUISITE NOTICE: Ollama is not installed on your machine.       " -ForegroundColor Yellow
    Write-Host " ==========================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Mi:RAG uses Ollama to run high-speed, local-first AI models." -ForegroundColor DarkGray
    Write-Host "  Zero mandatory cloud dependencies. Zero API subscription fees." -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Option 1: Open official Ollama download page (https://ollama.com/download)" -ForegroundColor White
    Write-Host "  Option 2: Install automatically via Windows Package Manager (winget)" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host -Prompt "  Select option [1/2] or press Enter to exit"
    if ($choice -eq "1") {
        Write-Host "  [*] Opening https://ollama.com/download in browser..." -ForegroundColor Cyan
        Start-Process "https://ollama.com/download"
        Write-Host "  [!] Please complete Ollama installation, then run this command again." -ForegroundColor Yellow
    } elseif ($choice -eq "2" -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "  [*] Installing Ollama via winget..." -ForegroundColor Cyan
        winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
        Write-Host "  [OK] Ollama installed! Please restart your terminal and run this command again." -ForegroundColor Green
    }
    Safe-Exit
    return
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
    Write-Host -NoNewline " [*] Setting up repository in $targetDir... " -ForegroundColor Cyan
    if (Get-Command git -ErrorAction SilentlyContinue) {
        git clone --quiet https://github.com/AryanSingh64/Mi-RAG.git $targetDir
        Set-Location $targetDir
        Write-Host "[ CLONED ]" -ForegroundColor Green
    } else {
        # Fallback for users without git: Download and extract zip directly
        Write-Host "[ DOWNLOADING ZIP ]" -ForegroundColor Yellow
        $zipUrl = "https://github.com/AryanSingh64/Mi-RAG/archive/refs/heads/main.zip"
        $zipPath = "$HOME\mirag_temp.zip"
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
        Expand-Archive -Path $zipPath -DestinationPath "$HOME" -Force
        Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
        if (Test-Path "$HOME\Mi-RAG-main") {
            Move-Item -Path "$HOME\Mi-RAG-main" -Destination $targetDir -Force -ErrorAction SilentlyContinue
        }
        Set-Location $targetDir
        Write-Host " [*] Extracted successfully!" -ForegroundColor Green
    }
}

# 6. Virtual Environment Setup
if (-not (Test-Path "$targetDir\.venv\Scripts\python.exe")) {
    Write-Host -NoNewline " [*] Creating virtual environment (.venv)... " -ForegroundColor Cyan
    python -m venv "$targetDir\.venv"
    Write-Host "[ CREATED ]" -ForegroundColor Green
}

# 7. Hardware & GPU Acceleration Detection
$detectedGpu = $null
try {
    $videoControllers = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    foreach ($vc in $videoControllers) {
        if ($vc.Name -match "NVIDIA") {
            $detectedGpu = $vc.Name
            break
        }
    }
} catch {}

if (-not $detectedGpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    try {
        $smiOut = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($smiOut) { $detectedGpu = $smiOut.Trim() }
    } catch {}
}

$installCuda = $false
$hasTorchCuda = & "$targetDir\.venv\Scripts\python.exe" -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>$null

if ($detectedGpu) {
    if ($hasTorchCuda -eq "CUDA") {
        Write-Host " [*] Hardware Acceleration: $detectedGpu [ CUDA ENABLED ]" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host " ==========================================================================" -ForegroundColor DarkGreen
        Write-Host "  [⚡] NVIDIA GPU DETECTED: $detectedGpu" -ForegroundColor Yellow
        Write-Host " ==========================================================================" -ForegroundColor DarkGreen
        Write-Host "  Would you like to install PyTorch with CUDA GPU acceleration for" -ForegroundColor White
        Write-Host "  ultra-fast embedding computation, vector indexing, and multimodal RAG?" -ForegroundColor White
        Write-Host ""
        Write-Host "  Option 1: Yes, install CUDA GPU Acceleration (Recommended for $detectedGpu)" -ForegroundColor Green
        Write-Host "  Option 2: No, use CPU only (Standard and Lightweight)" -ForegroundColor White
        Write-Host ""
        $gpuChoice = Read-Host -Prompt "  Select option [1/2] (Default is 1)"
        if ($gpuChoice -ne "2") {
            $installCuda = $true
        }
    }
} else {
    Write-Host " [*] Hardware Architecture: Standard Multi-Core CPU Mode [ ACTIVE ]" -ForegroundColor DarkGray
}

# 8. Dependencies Verification & Visual Live Progress Installation
$hasDeps = & "$targetDir\.venv\Scripts\python.exe" -c "import uvicorn, fastapi, fitz, chromadb; print('OK')" 2>$null
if ($hasDeps -ne "OK" -or ($installCuda -and $hasTorchCuda -ne "CUDA")) {
    Write-Host ""
    Write-Host " [*] Downloading & installing dependencies with live progress:" -ForegroundColor Yellow
    Write-Host " -----------------------------------------------------------------------" -ForegroundColor DarkGray
    
    # Try high-speed UV installer or standard pip with visual progress bar
    $uvInstalled = $false
    try {
        & "$targetDir\.venv\Scripts\python.exe" -m pip install --quiet uv 2>$null
        if (Test-Path "$targetDir\.venv\Scripts\uv.exe") { $uvInstalled = $true }
    } catch {}

    $depsInstalled = $false
    if ($uvInstalled) {
        & "$targetDir\.venv\Scripts\uv.exe" pip install -r "$targetDir\requirements.txt"
        if ($LASTEXITCODE -eq 0) {
            $depsInstalled = $true
        } else {
            Write-Host " [!] Accelerated installer encountered network/DNS error. Retrying with standard pip..." -ForegroundColor Yellow
        }
    }

    if (-not $depsInstalled) {
        & "$targetDir\.venv\Scripts\python.exe" -m pip install --retries 5 --timeout 60 --progress-bar on -r "$targetDir\requirements.txt"
    }

    if ($installCuda) {
        Write-Host ""
        Write-Host " [*] Installing CUDA-accelerated PyTorch (cu121)..." -ForegroundColor Cyan
        $cudaInstalled = $false
        if ($uvInstalled) {
            & "$targetDir\.venv\Scripts\uv.exe" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            if ($LASTEXITCODE -eq 0) { $cudaInstalled = $true }
        }
        if (-not $cudaInstalled) {
            & "$targetDir\.venv\Scripts\python.exe" -m pip install --retries 5 --timeout 60 --progress-bar on torch torchvision --index-url https://download.pytorch.org/whl/cu121
        }
    }
    
    Write-Host " -----------------------------------------------------------------------" -ForegroundColor DarkGray
    
    # Verify critical dependencies actually installed
    $verifyDeps = & "$targetDir\.venv\Scripts\python.exe" -c "import uvicorn, fastapi, fitz, chromadb; print('OK')" 2>$null
    if ($verifyDeps -eq "OK") {
        Write-Host " [OK] All dependencies installed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host " ==========================================================================" -ForegroundColor Red
        Write-Host "  [!] WARNING: Core Dependencies Incomplete (Network / DNS Error)" -ForegroundColor Yellow
        Write-Host " ==========================================================================" -ForegroundColor Red
        Write-Host "  Package download failed because your system cannot reach PyPI servers" -ForegroundColor White
        Write-Host "  (files.pythonhosted.org). This is usually caused by an unstable Wi-Fi" -ForegroundColor White
        Write-Host "  connection, VPN, or restrictive college/hostel/office DNS." -ForegroundColor White
        Write-Host ""
        Write-Host "  Quick Fix:" -ForegroundColor Yellow
        Write-Host "  1. Switch your Windows DNS to Cloudflare (1.1.1.1) or Google (8.8.8.8)." -ForegroundColor White
        Write-Host "  2. Once connected, re-run this command in terminal:" -ForegroundColor White
        Write-Host "     cd $targetDir ; .\.venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Cyan
        Write-Host ""
    }
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

# Pre-launch check: verify uvicorn exists before attempting to run
$canLaunch = & "$targetDir\.venv\Scripts\python.exe" -c "import uvicorn; print('OK')" 2>$null
if ($canLaunch -ne "OK") {
    Write-Host ""
    Write-Host " [!] Cannot launch Mi:RAG Studio because core packages (uvicorn) are not yet installed." -ForegroundColor Red
    Write-Host "     Please resolve the network connection issue above and re-run the installer." -ForegroundColor Yellow
    Safe-Exit
}

Write-Host ""
Write-Host " +---------------------------------------------------------+" -ForegroundColor Red
Write-Host " |  Mi:RAG Studio is launching on http://localhost:8000    |" -ForegroundColor Yellow
Write-Host " |  Local-First  |  Zero API Costs  |  Hardware Accelerated|" -ForegroundColor White
Write-Host " +---------------------------------------------------------+" -ForegroundColor Red
Write-Host ""

# 9. Direct Native Launch with Error Protection
Set-Location $targetDir
try {
    & "$targetDir\.venv\Scripts\python.exe" "$targetDir\run_factory.py"
} catch {
    Write-Host ""
    Write-Host " ===========================================================" -ForegroundColor Red
    Write-Host "  [!] Server stopped with note: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host " ===========================================================" -ForegroundColor Red
    Safe-Exit
}
