#!/usr/bin/env bash
# ==============================================================================
#  Mi:RAG — Minimal RAG Factory Automated Installer (Linux / macOS)
#  Repository: https://github.com/AryanSingh64/Mi-RAG
# ==============================================================================

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DARKGRAY='\033[1;30m'
NC='\033[0m' # No Color

clear 2>/dev/null || true

# 1. High-Impact Bold Red Pure ASCII Logo
echo ""
echo -e "${RED}  __  __ _       ____      _    ____  ${NC}"
echo -e "${RED} |  \/  (_)     |  _ \    / \  / ___| ${NC}"
echo -e "${RED} | |\/| | |  _  | |_) |  / _ \| |  _  ${NC}"
echo -e "${RED} | |  | | | (_) |  _ <  / ___ \ |_| | ${NC}"
echo -e "${RED} |_|  |_|_|     |_| \_\/_/   \_\____| ${NC}"
echo ""
echo -e "${DARKGRAY} ===========================================================${NC}"
echo -e "${YELLOW}   [ MINIMAL RAG ] - Autonomous Multimodal RAG Engine       ${NC}"
echo -e "${DARKGRAY} ===========================================================${NC}"
echo ""

print_step() {
    echo -ne " ${CYAN}[*] $1 ${NC}"
    sleep 0.08
    echo -e "${GREEN}[ OK ]${NC}"
}

safe_exit() {
    echo ""
    echo -e "${DARKGRAY}-----------------------------------------------------------${NC}"
    echo -e "${DARKGRAY}Setup paused. The terminal window will stay open.${NC}"
    read -p "Press [Enter] to close: " _dummy 2>/dev/null || true
    exit 1
}

# 2. Prerequisites Verification: Python 3.10+
print_step "Checking Python installation..."

find_healthy_python() {
    local candidates=("python3.11" "python3.12" "python3.10" "python3.13" "python3" "python")
    for cmd in "${candidates[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local p_path
            p_path=$(command -v "$cmd")
            # Verify the interpreter can run and check major.minor version
            local ver
            ver=$("$p_path" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
            if [ -n "$ver" ]; then
                local major minor
                major=$(echo "$ver" | cut -d. -f1)
                minor=$(echo "$ver" | cut -d. -f2)
                if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
                    echo "$p_path"
                    return 0
                fi
            fi
        fi
    done
    return 1
}

ACTIVE_PYTHON=$(find_healthy_python || true)

if [ -z "$ACTIVE_PYTHON" ]; then
    echo ""
    echo -e "${RED} ==========================================================================${NC}"
    echo -e "${YELLOW}  [!] PREREQUISITE NOTICE: Working Python 3.10+ was not detected.         ${NC}"
    echo -e "${RED} ==========================================================================${NC}"
    echo "  Mi:RAG requires Python 3.10, 3.11, 3.12, or 3.13."
    echo ""
    echo "  Option 1: Install Python via your system package manager"
    echo "  Option 2: Open official Python download page (https://www.python.org/downloads/)"
    echo ""
    read -p "  Select option [1/2] or press Enter to exit: " py_choice

    if [ "$py_choice" = "1" ]; then
        if command -v apt-get >/dev/null 2>&1; then
            echo -e "${CYAN}  [*] Installing Python via apt...${NC}"
            sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
        elif command -v dnf >/dev/null 2>&1; then
            echo -e "${CYAN}  [*] Installing Python via dnf...${NC}"
            sudo dnf install -y python3 python3-pip
        elif command -v pacman >/dev/null 2>&1; then
            echo -e "${CYAN}  [*] Installing Python via pacman...${NC}"
            sudo pacman -Sy --noconfirm python python-pip
        elif command -v brew >/dev/null 2>&1; then
            echo -e "${CYAN}  [*] Installing Python via brew...${NC}"
            brew install python@3.11
        fi
        echo -e "${GREEN}  [OK] Python installed! Please re-run the installer.${NC}"
    fi
    safe_exit
fi

PY_VER=$("$ACTIVE_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e " ${CYAN}[*] Base Python: Python $PY_VER ($ACTIVE_PYTHON) ${GREEN}[ OK ]${NC}"

# 3. Prerequisites Verification: Ollama AI Engine
print_step "Checking Ollama AI Engine installation..."
if ! command -v ollama >/dev/null 2>&1; then
    echo ""
    echo -e "${RED} ==========================================================================${NC}"
    echo -e "${YELLOW}  [!] PREREQUISITE NOTICE: Ollama is not installed on your machine.       ${NC}"
    echo -e "${RED} ==========================================================================${NC}"
    echo "  Mi:RAG uses Ollama to run high-speed, local-first AI models."
    echo "  Zero mandatory cloud dependencies. Zero API subscription fees."
    echo ""
    echo "  Option 1: Automatically install Ollama via official installer script"
    echo "  Option 2: Visit https://ollama.com/download"
    echo ""
    read -p "  Select option [1/2] or press Enter to exit: " ol_choice

    if [ "$ol_choice" = "1" ]; then
        echo -e "${CYAN}  [*] Running official Ollama installer...${NC}"
        curl -fsSL https://ollama.com/install.sh | sh
        echo -e "${GREEN}  [OK] Ollama installed successfully!${NC}"
    fi
    safe_exit
fi

# 4. Check / Auto-Start Ollama Service
echo -ne " ${CYAN}[*] Checking Ollama local service... ${NC}"
if curl -s -m 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}[ ACTIVE ]${NC}"
else
    echo -e "${YELLOW}[ STARTING SERVICE ]${NC}"
    ollama serve >/dev/null 2>&1 &
    sleep 3
    if curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo -e " ${GREEN}[*] Ollama service connected successfully!${NC}"
    else
        echo -e " ${YELLOW}[*] Ollama service launched in background.${NC}"
    fi
fi

# 5. Repository Discovery / Directory Setup
if [ -f "./run_factory.py" ]; then
    TARGET_DIR="$(pwd)"
    echo -e " ${CYAN}[*] Running from local repository at $TARGET_DIR${NC}"
elif [ -f "$HOME/Mi-RAG/run_factory.py" ]; then
    TARGET_DIR="$HOME/Mi-RAG"
    echo -ne " ${CYAN}[*] Checking for updates in $TARGET_DIR... ${NC}"
    cd "$TARGET_DIR"
    if git pull --quiet 2>/dev/null; then
        echo -e "${GREEN}[ UP TO DATE ]${NC}"
    else
        echo -e "${YELLOW}[ OFFLINE MODE ]${NC}"
    fi
else
    TARGET_DIR="$HOME/Mi-RAG"
    echo -ne " ${CYAN}[*] Setting up repository in $TARGET_DIR... ${NC}"
    if command -v git >/dev/null 2>&1; then
        git clone --quiet https://github.com/AryanSingh64/Mi-RAG.git "$TARGET_DIR"
        cd "$TARGET_DIR"
        echo -e "${GREEN}[ CLONED ]${NC}"
    else
        mkdir -p "$TARGET_DIR"
        curl -fsSL https://github.com/AryanSingh64/Mi-RAG/archive/refs/heads/main.tar.gz | tar -xz -C "$TARGET_DIR" --strip-components=1
        cd "$TARGET_DIR"
        echo -e "${GREEN}[ EXTRACTED ]${NC}"
    fi
fi

cd "$TARGET_DIR"

# 6. Virtual Environment Setup & Health Verification
VENV_DIR="$TARGET_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
NEEDS_VENV_REBUILD=false

if [ -f "$VENV_PYTHON" ]; then
    VENV_TEST=$("$VENV_PYTHON" -c "import sys; print('HEALTHY')" 2>/dev/null || true)
    if [ "$VENV_TEST" = "HEALTHY" ]; then
        print_step "Virtual environment (.venv) verified..."
    else
        echo ""
        echo -e "${RED} ==========================================================================${NC}"
        echo -e "${YELLOW}  [!] NOTICE: Existing virtual environment (.venv) is invalid or broken.  ${NC}"
        echo -e "${RED} ==========================================================================${NC}"
        echo "  The interpreter at $VENV_PYTHON failed to execute."
        echo "  This usually occurs when base Python was moved, updated, or reinstalled."
        echo ""
        read -p "  Repair and recreate virtual environment now? [Y/n] (Default is Y): " repair_choice
        if [ "$repair_choice" != "n" ] && [ "$repair_choice" != "N" ]; then
            echo -e "  ${YELLOW}[*] Removing broken virtual environment...${NC}"
            rm -rf "$VENV_DIR"
            NEEDS_VENV_REBUILD=true
        else
            echo -e "  ${YELLOW}[!] Warning: Retaining unverified virtual environment.${NC}"
        fi
    fi
else
    NEEDS_VENV_REBUILD=true
fi

if [ "$NEEDS_VENV_REBUILD" = true ]; then
    echo -ne " ${CYAN}[*] Creating virtual environment (.venv) with Python $PY_VER... ${NC}"
    "$ACTIVE_PYTHON" -m venv "$VENV_DIR"
    NEW_VENV_TEST=$("$VENV_PYTHON" -c "import sys; print('HEALTHY')" 2>/dev/null || true)
    if [ "$NEW_VENV_TEST" = "HEALTHY" ]; then
        echo -e "${GREEN}[ CREATED & VERIFIED ]${NC}"
    else
        echo -e "${RED}[ FAILED ]${NC}"
        echo -e "${RED}  [!] Failed to initialize a working virtual environment with $ACTIVE_PYTHON.${NC}"
        safe_exit
    fi
fi

# 7. Hardware & GPU Acceleration Detection
DETECTED_GPU=""
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
elif lspci 2>/dev/null | grep -i "nvidia" >/dev/null 2>&1; then
    DETECTED_GPU=$(lspci 2>/dev/null | grep -i "nvidia" | head -n 1 | cut -d: -f3 | sed 's/^[ \t]*//')
fi

GPU_PREF_FILE="$VENV_DIR/.gpu_preference"
INSTALL_CUDA=false
HAS_TORCH_CUDA=$("$VENV_PYTHON" -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null || true)

CUDA_TAG="cu121"
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [ "$PY_MINOR" -ge 13 ]; then
    CUDA_TAG="cu124"
fi
CUDA_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"

if [ -n "$DETECTED_GPU" ]; then
    if [ "$HAS_TORCH_CUDA" = "CUDA" ]; then
        echo -e " ${GREEN}[*] Hardware Acceleration: $DETECTED_GPU [ CUDA ENABLED ]${NC}"
    elif [ -f "$GPU_PREF_FILE" ]; then
        SAVED_PREF=$(cat "$GPU_PREF_FILE" 2>/dev/null | tr -d '[:space:]')
        if [ "$SAVED_PREF" = "cpu" ]; then
            echo -e " ${DARKGRAY}[*] Hardware Acceleration: CPU Mode (Persisted Preference)${NC}"
        elif [ "$SAVED_PREF" = "cuda" ]; then
            echo -e " ${CYAN}[*] Hardware Acceleration: $DETECTED_GPU [ CONFIG: CUDA ]${NC}"
            INSTALL_CUDA=true
        fi
    else
        echo ""
        echo -e "${GREEN} ==========================================================================${NC}"
        echo -e "${YELLOW}  [⚡] NVIDIA GPU DETECTED: $DETECTED_GPU${NC}"
        echo -e "${GREEN} ==========================================================================${NC}"
        echo "  Would you like to install PyTorch with CUDA GPU acceleration for"
        echo "  ultra-fast embedding computation, vector indexing, and multimodal RAG?"
        echo ""
        echo -e "${GREEN}  Option 1: Yes, install CUDA GPU Acceleration (Recommended for $DETECTED_GPU)${NC}"
        echo "  Option 2: No, use CPU only (Standard and Lightweight)"
        echo ""
        read -p "  Select option [1/2] (Default is 1): " gpu_choice
        if [ "$gpu_choice" = "2" ]; then
            echo "cpu" > "$GPU_PREF_FILE"
            echo -e " ${DARKGRAY}[*] Configured for CPU-only mode.${NC}"
        else
            echo "cuda" > "$GPU_PREF_FILE"
            INSTALL_CUDA=true
        fi
    fi
else
    echo -e " ${DARKGRAY}[*] Hardware Architecture: Standard Multi-Core CPU Mode [ ACTIVE ]${NC}"
fi

# 8. Dependencies Verification & Installation
HAS_DEPS=$("$VENV_PYTHON" -c "import uvicorn, fastapi, fitz, chromadb; print('OK')" 2>/dev/null || true)

if [ "$HAS_DEPS" != "OK" ] || { [ "$INSTALL_CUDA" = true ] && [ "$HAS_TORCH_CUDA" != "CUDA" ]; }; then
    echo ""
    echo -e " ${YELLOW}[*] Downloading & installing dependencies:${NC}"
    echo -e "${DARKGRAY} -----------------------------------------------------------------------${NC}"

    UV_INSTALLED=false
    if "$VENV_PYTHON" -m pip install --quiet uv 2>/dev/null && [ -f "$VENV_DIR/bin/uv" ]; then
        if "$VENV_DIR/bin/uv" --version >/dev/null 2>&1; then
            UV_INSTALLED=true
        fi
    fi

    DEPS_INSTALLED=false
    if [ "$UV_INSTALLED" = true ]; then
        if "$VENV_DIR/bin/uv" pip install -r "$TARGET_DIR/requirements.txt"; then
            DEPS_INSTALLED=true
        else
            echo -e "${YELLOW} [!] Accelerated installer encountered an issue. Falling back to standard pip...${NC}"
        fi
    fi

    if [ "$DEPS_INSTALLED" = false ]; then
        "$VENV_PYTHON" -m pip install --retries 5 --timeout 60 -r "$TARGET_DIR/requirements.txt"
    fi

    if [ "$INSTALL_CUDA" = true ]; then
        echo ""
        echo -e " ${CYAN}[*] Installing CUDA-accelerated PyTorch ($CUDA_TAG)...${NC}"
        CUDA_INSTALLED=false
        if [ "$UV_INSTALLED" = true ]; then
            if "$VENV_DIR/bin/uv" pip install --upgrade torch --index-url "$CUDA_INDEX"; then
                CUDA_INSTALLED=true
            fi
        fi
        if [ "$CUDA_INSTALLED" = false ]; then
            "$VENV_PYTHON" -m pip install --retries 5 --timeout 60 --upgrade torch --index-url "$CUDA_INDEX"
        fi
    fi

    echo -e "${DARKGRAY} -----------------------------------------------------------------------${NC}"

    VERIFY_DEPS=$("$VENV_PYTHON" -c "import uvicorn, fastapi, fitz, chromadb; print('OK')" 2>/dev/null || true)
    if [ "$VERIFY_DEPS" = "OK" ]; then
        echo -e " ${GREEN}[OK] All dependencies installed successfully!${NC}"
    else
        echo ""
        echo -e "${RED} ==========================================================================${NC}"
        echo -e "${YELLOW}  [!] WARNING: Core Dependencies Incomplete${NC}"
        echo -e "${RED} ==========================================================================${NC}"
        echo "  One or more packages failed to install. Try running:"
        echo -e "${CYAN}     cd $TARGET_DIR && ./.venv/bin/pip install -r requirements.txt${NC}"
        echo ""
    fi
fi

# 9. Automatic Port Freeing
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8000/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fi

# Pre-launch check
CAN_LAUNCH=$("$VENV_PYTHON" -c "import uvicorn; print('OK')" 2>/dev/null || true)
if [ "$CAN_LAUNCH" != "OK" ]; then
    echo ""
    echo -e "${RED} [!] Cannot launch Mi:RAG Studio because core packages (uvicorn) are not yet installed.${NC}"
    safe_exit
fi

echo ""
echo -e "${RED} +---------------------------------------------------------+${NC}"
echo -e "${YELLOW} |  Mi:RAG Studio is launching on http://localhost:8000    |${NC}"
echo -e " |  Local-First  |  Zero API Costs  |  Hardware Accelerated|"
echo -e "${RED} +---------------------------------------------------------+${NC}"
echo ""

# 11. Launch
cd "$TARGET_DIR"
exec "$VENV_PYTHON" "$TARGET_DIR/run_factory.py"
