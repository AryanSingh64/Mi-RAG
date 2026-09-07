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

# Helper to register global 'mirag' terminal command
function Register-MiragCommand($repoDir) {
    try {
        $binDir = "$HOME\.mirag\bin"
        if (-not (Test-Path $binDir)) {
            New-Item -ItemType Directory -Path $binDir -Force | Out-Null
        }

        # 1. mirag.cmd & mirag.bat for standard Windows Command Prompt
        $cmdScript = @"
@echo off
set "MIRAG_DIR=$repoDir"
if exist "%CD%\run_factory.py" (
    if exist "%CD%\.venv\Scripts\python.exe" (
        set "MIRAG_DIR=%CD%"
    )
)
if not exist "%MIRAG_DIR%\.venv\Scripts\python.exe" (
    echo [!] Mi:RAG virtual environment not found in %MIRAG_DIR%. Please re-run the installer.
    pause
    exit /b 1
)
"%MIRAG_DIR%\.venv\Scripts\python.exe" "%MIRAG_DIR%\run_factory.py" %*
"@

        # 2. mirag.ps1 for PowerShell
        $escapedRepoDir = $repoDir -replace "'", "''"
        $psScript = @"
`$repoDir = '$escapedRepoDir'
if (Test-Path ".\run_factory.py") {
    if (Test-Path ".\.venv\Scripts\python.exe") {
        `$testUvi = & ".\.venv\Scripts\python.exe" -c "import uvicorn; print('ok')" 2>`$null
        if (`$testUvi -and `$testUvi.Trim() -eq "ok") {
            `$repoDir = (Get-Location).Path
        }
    }
}
`$pythonExe = "`$repoDir\.venv\Scripts\python.exe"
if (-not (Test-Path `$pythonExe)) {
    if (Test-Path "$HOME\Mi-RAG\.venv\Scripts\python.exe") {
        `$repoDir = "$HOME\Mi-RAG"
        `$pythonExe = "`$repoDir\.venv\Scripts\python.exe"
    }
}
`$runScript = "`$repoDir\run_factory.py"
if (-not (Test-Path `$pythonExe)) {
    Write-Host "[!] Mi:RAG virtual environment not found in `$repoDir. Please re-run the installer." -ForegroundColor Red
    Read-Host -Prompt "Press [Enter] to exit"
    exit 1
}
& `$pythonExe `$runScript `$args
"@

        # 3. Unix script for Git Bash / MSYS2 / WSL
        $posixDir = $repoDir -replace '\\', '/'
        $shScript = @"
#!/usr/bin/env bash
REPO_DIR="$posixDir"
if [ -f "./run_factory.py" ] && [ -f "./.venv/Scripts/python.exe" ]; then
    REPO_DIR="\$(pwd)"
fi
if [ ! -f "`$REPO_DIR/.venv/Scripts/python.exe" ]; then
    echo "[!] Mi:RAG virtual environment not found in `$REPO_DIR. Please re-run the installer."
    exit 1
fi
exec "`$REPO_DIR/.venv/Scripts/python.exe" "`$REPO_DIR/run_factory.py" "`$@"
"@

        $targetDirs = @("$HOME\.mirag\bin", "$HOME\.local\bin")
        foreach ($bDir in $targetDirs) {
            if (-not (Test-Path $bDir)) {
                New-Item -ItemType Directory -Path $bDir -Force | Out-Null
            }
            Set-Content -Path "$bDir\mirag.cmd" -Value $cmdScript -Encoding ASCII -Force
            Set-Content -Path "$bDir\mirag.bat" -Value $cmdScript -Encoding ASCII -Force
            Set-Content -Path "$bDir\mirag.ps1" -Value $psScript -Encoding UTF8 -Force
            Set-Content -Path "$bDir\mirag" -Value $shScript -Encoding ASCII -Force
        }

        # Add $binDir to User Environment PATH if not already present
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $pathParts = if ($userPath) { $userPath -split ';' } else { @() }
        if ($pathParts -notcontains $binDir) {
            $newPath = if ($userPath) { "$binDir;$userPath" } else { $binDir }
            [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        }

        # Update current session PATH so 'mirag' works immediately
        if (($env:Path -split ';') -notcontains $binDir) {
            $env:Path = "$binDir;$env:Path"
        }

        Write-Host " [*] Global CLI Command: Registered 'mirag' in terminal PATH" -ForegroundColor Green
    } catch {
        # Non-critical failure: don't abort setup if shortcut registration encounters an error
    }
}

# Helper to validate a python executable path
function Test-PythonExecutable($exePath) {
    if (-not $exePath -or -not (Test-Path $exePath)) { return $null }
    try {
        # Test basic execution and get major.minor version
        $res = & $exePath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($res -and ($res.Trim() -match '^\d+\.\d+$')) {
            $verStr = $res.Trim()
            $ver = [version]$verStr
            if ($ver.Major -eq 3 -and $ver.Minor -ge 10) {
                $isStore = ($exePath -like "*WindowsApps*")
                return [PSCustomObject]@{
                    Path    = (Resolve-Path $exePath).Path
                    Version = $verStr
                    Major   = $ver.Major
                    Minor   = $ver.Minor
                    IsStore = $isStore
                }
            }
        }
    } catch {}
    return $null
}

# Discovery function that finds and ranks working Python interpreters
function Find-HealthyPython {
    $found = @()
    $testedPaths = @{}

    # 1. Search via Python Launcher py.exe
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($tag in @("3.11", "3.12", "3.10", "3.13", "3")) {
            try {
                $pyExe = py -$tag -c "import sys; print(sys.executable)" 2>$null
                if ($pyExe) {
                    $cleaned = $pyExe.Trim()
                    if (-not $testedPaths.ContainsKey($cleaned.ToLower())) {
                        $testedPaths[$cleaned.ToLower()] = $true
                        $candidate = Test-PythonExecutable $cleaned
                        if ($candidate) { $found += $candidate }
                    }
                }
            } catch {}
        }
    }

    # 2. Search common official Python installation directories
    $searchPatterns = @(
        "$env:LocalAppData\Programs\Python\Python*",
        "$env:ProgramFiles\Python*",
        "${env:ProgramFiles(x86)}\Python*",
        "C:\Python*"
    )
    foreach ($pat in $searchPatterns) {
        Get-Item $pat -ErrorAction SilentlyContinue | ForEach-Object {
            $exe = Join-Path $_.FullName "python.exe"
            if (Test-Path $exe) {
                $key = $exe.ToLower()
                if (-not $testedPaths.ContainsKey($key)) {
                    $testedPaths[$key] = $true
                    $candidate = Test-PythonExecutable $exe
                    if ($candidate) { $found += $candidate }
                }
            }
        }
    }

    # 3. Search PATH items
    Get-Command python -All -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Source) {
            $key = $_.Source.ToLower()
            if (-not $testedPaths.ContainsKey($key)) {
                $testedPaths[$key] = $true
                $candidate = Test-PythonExecutable $_.Source
                if ($candidate) { $found += $candidate }
            }
        }
    }

    if ($found.Count -eq 0) { return $null }

    # Ranking:
    # 1st priority: Non-store official interpreters (store AppX execution aliases are fragile)
    # 2nd priority: Python 3.11 or 3.12 (best wheel compatibility for torch cu121 & chromadb), then 3.10, then 3.13
    $ranked = $found | Sort-Object -Property @{
        Expression = { if ($_.IsStore) { 1 } else { 0 } }
    }, @{
        Expression = {
            if ($_.Minor -eq 11) { 0 }
            elseif ($_.Minor -eq 12) { 1 }
            elseif ($_.Minor -eq 10) { 2 }
            elseif ($_.Minor -eq 13) { 3 }
            else { 4 }
        }
    }

    return @($ranked)[0]
}

# 2. Prerequisites Verification: Python 3.10+
Print-Step "Checking Python installation..."
$activePython = Find-HealthyPython

if (-not $activePython) {
    Write-Host ""
    Write-Host " ==========================================================================" -ForegroundColor Red
    Write-Host "  [!] PREREQUISITE NOTICE: Working Python 3.10+ was not detected.         " -ForegroundColor Yellow
    Write-Host " ==========================================================================" -ForegroundColor Red
    Write-Host "  Mi:RAG requires a functional Python 3.10, 3.11, 3.12, or 3.13 install." -ForegroundColor DarkGray
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
} else {
    Write-Host " [*] Base Python: Python $($activePython.Version) ($($activePython.Path)) [ OK ]" -ForegroundColor DarkCyan
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
if (Test-Path ".\run_factory.py") {
    $targetDir = (Get-Location).Path
    Write-Host " [*] Running from local repository at $targetDir" -ForegroundColor Cyan
} elseif (Test-Path "$HOME\Mi-RAG\run_factory.py") {
    $targetDir = "$HOME\Mi-RAG"
    Write-Host -NoNewline " [*] Checking for updates in $targetDir... " -ForegroundColor Cyan
    Set-Location $targetDir
    try {
        $gitOutput = git pull --quiet 2>&1
        Write-Host "[ UP TO DATE ]" -ForegroundColor Green
    } catch {
        Write-Host "[ OFFLINE MODE ]" -ForegroundColor Yellow
    }
} else {
    $targetDir = "$HOME\Mi-RAG"
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

# 6. Virtual Environment Setup & Health Verification
$venvDir = "$targetDir\.venv"
$venvPython = "$venvDir\Scripts\python.exe"
$needsVenvRebuild = $false

if (Test-Path $venvPython) {
    # Verify existing virtual environment is actually functional
    $venvHealthTest = & $venvPython -c "import sys; print('HEALTHY')" 2>$null
    if ($venvHealthTest -eq "HEALTHY") {
        Print-Step "Virtual environment (.venv) verified..."
    } else {
        Write-Host ""
        Write-Host " ==========================================================================" -ForegroundColor Red
        Write-Host "  [!] NOTICE: Existing virtual environment (.venv) is corrupted or invalid." -ForegroundColor Yellow
        Write-Host " ==========================================================================" -ForegroundColor Red
        Write-Host "  The interpreter at $venvPython failed to execute." -ForegroundColor White
        Write-Host "  This usually occurs when base Python was moved, updated, or created via" -ForegroundColor White
        Write-Host "  restricted Microsoft Store execution aliases." -ForegroundColor White
        Write-Host ""
        Write-Host "  Mi:RAG can automatically repair this by rebuilding the virtual environment" -ForegroundColor White
        Write-Host "  using your verified base Python: $($activePython.Path) ($($activePython.Version))" -ForegroundColor Cyan
        Write-Host ""
        
        $repairPerm = Read-Host -Prompt "  Repair and recreate virtual environment now? [Y/n] (Default is Y)"
        if ($repairPerm -ne "n" -and $repairPerm -ne "N") {
            Write-Host "  [*] Removing broken virtual environment..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $venvDir -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 600
            $needsVenvRebuild = $true
        } else {
            Write-Host "  [!] Warning: Retaining unverified virtual environment as requested." -ForegroundColor Yellow
        }
    }
} else {
    $needsVenvRebuild = $true
}

if ($needsVenvRebuild) {
    Write-Host -NoNewline " [*] Creating virtual environment (.venv) with Python $($activePython.Version)... " -ForegroundColor Cyan
    & $activePython.Path -m venv $venvDir
    $newVenvHealth = & $venvPython -c "import sys; print('HEALTHY')" 2>$null
    if ($newVenvHealth -eq "HEALTHY") {
        Write-Host "[ CREATED & VERIFIED ]" -ForegroundColor Green
    } else {
        Write-Host "[ FAILED ]" -ForegroundColor Red
        Write-Host "  [!] Failed to initialize a working virtual environment." -ForegroundColor Red
        Write-Host "      Interpreter path: $($activePython.Path)" -ForegroundColor Yellow
        Safe-Exit
        return
    }
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

$gpuPrefFile = "$venvDir\.gpu_preference"
$installCuda = $false
$hasTorchCuda = & $venvPython -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>$null

# Dynamic PyTorch CUDA index: Python 3.13+ officially requires cu124/cu126; Python 3.10-3.12 uses cu121
$cudaTag = if ($activePython.Minor -ge 13) { "cu124" } else { "cu121" }
$cudaIndex = "https://download.pytorch.org/whl/$cudaTag"

if ($detectedGpu) {
    if ($hasTorchCuda -eq "CUDA") {
        Write-Host " [*] Hardware Acceleration: $detectedGpu [ CUDA ENABLED ]" -ForegroundColor Green
    } elseif (Test-Path $gpuPrefFile) {
        $savedPref = (Get-Content $gpuPrefFile -Raw).Trim()
        if ($savedPref -eq "cpu") {
            Write-Host " [*] Hardware Acceleration: CPU Mode (Persisted Preference)" -ForegroundColor DarkGray
        } elseif ($savedPref -eq "cuda") {
            Write-Host " [*] Hardware Acceleration: $detectedGpu [ CONFIG: CUDA ]" -ForegroundColor Cyan
            $installCuda = $true
        }
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
        if ($gpuChoice -eq "2") {
            Set-Content -Path $gpuPrefFile -Value "cpu" -Force
            Write-Host " [*] Configured for CPU-only mode." -ForegroundColor DarkGray
        } else {
            Set-Content -Path $gpuPrefFile -Value "cuda" -Force
            $installCuda = $true
        }
    }
} else {
    Write-Host " [*] Hardware Architecture: Standard Multi-Core CPU Mode [ ACTIVE ]" -ForegroundColor DarkGray
}

# 8. Dependencies Verification & Visual Live Progress Installation
$hasDeps = & $venvPython -c "import uvicorn, fastapi, fitz, chromadb; print('OK')" 2>$null
if ($hasDeps -ne "OK" -or ($installCuda -and $hasTorchCuda -ne "CUDA")) {
    Write-Host ""
    Write-Host " [*] Downloading & installing dependencies with live progress:" -ForegroundColor Yellow
    Write-Host " -----------------------------------------------------------------------" -ForegroundColor DarkGray
    
    # Try high-speed UV installer or fallback to standard pip
    $uvInstalled = $false
    try {
        & $venvPython -m pip install --quiet uv 2>$null
        if (Test-Path "$venvDir\Scripts\uv.exe") {
            # Test uv execution
            $uvTest = & "$venvDir\Scripts\uv.exe" --version 2>$null
            if ($uvTest) { $uvInstalled = $true }
        }
    } catch {}

    $depsInstalled = $false
    if ($uvInstalled) {
        & "$venvDir\Scripts\uv.exe" pip install -r "$targetDir\requirements.txt"
        if ($LASTEXITCODE -eq 0) {
            $depsInstalled = $true
        } else {
            Write-Host " [!] Accelerated installer encountered an issue. Falling back to standard pip..." -ForegroundColor Yellow
        }
    }

    if (-not $depsInstalled) {
        $pipOutput = & $venvPython -m pip install --retries 5 --timeout 60 --progress-bar on -r "$targetDir\requirements.txt" 2>&1
        $pipExit = $LASTEXITCODE
        if ($pipExit -eq 0) {
            $depsInstalled = $true
        } else {
            # Check the nature of the pip failure
            $errText = $pipOutput | Out-String
            if ($errText -match "Unable to create process") {
                Write-Host ""
                Write-Host " [!] Error: Interpreter failed to spawn child process." -ForegroundColor Red
                Write-Host "     This indicates a permissions or execution alias conflict with the base Python." -ForegroundColor Yellow
            } elseif ($errText -match "Could not find a version" -or $errText -match "No matching distribution") {
                Write-Host ""
                Write-Host " [!] Error: Package compatibility issue for Python $($activePython.Version)." -ForegroundColor Red
            } elseif ($errText -match "timed out" -or $errText -match "Could not fetch URL" -or $errText -match "Temporary failure in name resolution") {
                Write-Host ""
                Write-Host " [!] Error: Network connection timeout reaching PyPI." -ForegroundColor Yellow
            }
        }
    }

    if ($installCuda) {
        Write-Host ""
        Write-Host " [*] Installing CUDA-accelerated PyTorch ($cudaTag)..." -ForegroundColor Cyan
        $cudaInstalled = $false
        if ($uvInstalled) {
            & "$venvDir\Scripts\uv.exe" pip install --upgrade torch torchvision --index-url $cudaIndex
            if ($LASTEXITCODE -eq 0) { $cudaInstalled = $true }
        }
        if (-not $cudaInstalled) {
            & $venvPython -m pip install --retries 5 --timeout 60 --upgrade --progress-bar on torch torchvision --index-url $cudaIndex
        }
    }
    
    Write-Host " -----------------------------------------------------------------------" -ForegroundColor DarkGray
    
    # Verify critical dependencies actually installed
    $verifyDeps = & $venvPython -c "import uvicorn, fastapi, fitz, chromadb; print('OK')" 2>$null
    if ($verifyDeps -eq "OK") {
        Write-Host " [OK] All dependencies installed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host " ==========================================================================" -ForegroundColor Red
        Write-Host "  [!] WARNING: Core Dependencies Incomplete" -ForegroundColor Yellow
        Write-Host " ==========================================================================" -ForegroundColor Red
        Write-Host "  One or more core packages (uvicorn, fastapi, chromadb, pymupdf) failed to install." -ForegroundColor White
        Write-Host ""
        Write-Host "  Troubleshooting Options:" -ForegroundColor Yellow
        Write-Host "  1. Switch your Windows DNS to Cloudflare (1.1.1.1) or Google (8.8.8.8) if downloads timed out." -ForegroundColor White
        Write-Host "  2. Or manually run in your terminal:" -ForegroundColor White
        Write-Host "     cd $targetDir ; .\.venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Cyan
        Write-Host ""
    }
}

# 9. Automatic Port Freeing & Launch Banner
try {
    $connections = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        if ($conn.OwningProcess -and $conn.OwningProcess -ne $PID) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
} catch {}

# Pre-launch check: verify uvicorn exists before attempting to run
$canLaunch = & $venvPython -c "import uvicorn; print('OK')" 2>$null
if ($canLaunch -ne "OK") {
    Write-Host ""
    Write-Host " [!] Cannot launch Mi:RAG Studio because core packages (uvicorn) are not yet installed." -ForegroundColor Red
    Write-Host "     Please resolve the package installation issue above and re-run the installer." -ForegroundColor Yellow
    Safe-Exit
    return
}

# 9. Register Global CLI Shortcut ('mirag')
Register-MiragCommand $targetDir

Write-Host ""
Write-Host " +---------------------------------------------------------+" -ForegroundColor Red
Write-Host " |  Mi:RAG Studio is launching on http://localhost:8000    |" -ForegroundColor Yellow
Write-Host " |  Local-First  |  Zero API Costs  |  Hardware Accelerated|" -ForegroundColor White
Write-Host " |  CLI Command  : Type 'mirag' in any terminal to launch  |" -ForegroundColor Cyan
Write-Host " +---------------------------------------------------------+" -ForegroundColor Red
Write-Host ""

# 10. Direct Native Launch with Error Protection
Set-Location $targetDir
try {
    & $venvPython "$targetDir\run_factory.py"
} catch {
    Write-Host ""
    Write-Host " ===========================================================" -ForegroundColor Red
    Write-Host "  [!] Server stopped with note: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host " ===========================================================" -ForegroundColor Red
    Safe-Exit
}
