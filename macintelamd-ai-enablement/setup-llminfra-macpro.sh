#!/usr/bin/env bash
#===============================================================================
# setup-llminfra-macpro.sh
#
# Automated LLM Inference Setup for Mac Pro 7,1 (2019) with AMD Radeon Pro
# Vega II / Vega II Duo GPUs via Vulkan (MoltenVK) backend.
#
# Why Vulkan? On Intel Mac Pro with AMD discrete GPUs:
#   - Metal backend: tensor API disabled for pre-Apple-Silicon GPUs → garbage output
#   - MLX: Apple Silicon only → not an option
#   - ROCm: Linux only (but Vega 20 = MI60 → works great if you dual-boot)
#   - Vulkan via MoltenVK: works on macOS, proper GPU compute, our winner
#
# Tested on:
#   - Mac Pro 7,1 (2019), 2.7 GHz 24-Core Intel Xeon W
#   - AMD Radeon Pro Vega II Duo 32 GB (x2 cards = 4 GPU dies)
#   - 256 GB DDR4 ECC RAM
#   - macOS Tahoe 26.x
#
# Usage:
#   chmod +x setup-llminfra-macpro.sh
#   ./setup-llminfra-macpro.sh [--model MODEL] [--quant QUANT] [--server]
#
# Examples:
#   ./setup-llminfra-macpro.sh                          # Qwen 3.5-35B-A3B interactive chat
#   ./setup-llminfra-macpro.sh --server                  # OpenAI-compatible API server
#   ./setup-llminfra-macpro.sh --model "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" --quant Q4_K_M
#
# Author: Fatih E. NAR 
# Project: https://github.com/open-experiments/Telco-AIX
#===============================================================================
set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration defaults
#-------------------------------------------------------------------------------
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp"
LLAMA_CPP_DIR="${HOME}/projects/llama.cpp"
DEFAULT_MODEL="bartowski/Qwen_Qwen3.5-35B-A3B-GGUF"
DEFAULT_QUANT="Q4_K_M"
SERVER_MODE=false
NUM_GPU_LAYERS=99
CONTEXT_SIZE=4096
SERVER_PORT=8080
REPEAT_PENALTY=1.1

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#-------------------------------------------------------------------------------
# Parse arguments
#-------------------------------------------------------------------------------
MODEL="${DEFAULT_MODEL}"
QUANT="${DEFAULT_QUANT}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)  MODEL="$2"; shift 2 ;;
        --quant)  QUANT="$2"; shift 2 ;;
        --server) SERVER_MODE=true; shift ;;
        --dir)    LLAMA_CPP_DIR="$2"; shift 2 ;;
        --port)   SERVER_PORT="$2"; shift 2 ;;
        --ctx)    CONTEXT_SIZE="$2"; shift 2 ;;
        --ngl)    NUM_GPU_LAYERS="$2"; shift 2 ;;
        --repeat-penalty) REPEAT_PENALTY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--model MODEL] [--quant QUANT] [--server] [--dir DIR] [--port PORT] [--ctx SIZE] [--ngl LAYERS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL   HuggingFace model repo (default: ${DEFAULT_MODEL})"
            echo "  --quant QUANT   Quantization level (default: ${DEFAULT_QUANT})"
            echo "  --server        Launch as OpenAI-compatible API server instead of CLI chat"
            echo "  --dir DIR       Install directory for llama.cpp (default: ${LLAMA_CPP_DIR})"
            echo "  --port PORT     Server port (default: ${SERVER_PORT})"
            echo "  --ctx SIZE      Context window size (default: ${CONTEXT_SIZE})"
            echo "  --ngl LAYERS    Number of layers to offload to GPU (default: ${NUM_GPU_LAYERS})"
            echo "  --repeat-penalty P  Repeat penalty to avoid loops (default: ${REPEAT_PENALTY})"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------
log_step()  { echo -e "\n${CYAN}▸ $1${NC}"; }
log_ok()    { echo -e "${GREEN}✔ $1${NC}"; }
log_warn()  { echo -e "${YELLOW}⚠ $1${NC}"; }
log_error() { echo -e "${RED}✖ $1${NC}"; }

check_cmd() {
    if command -v "$1" &>/dev/null; then
        log_ok "$1 found: $(command -v "$1")"
        return 0
    else
        return 1
    fi
}

#===============================================================================
# STEP 1: Validate environment
#===============================================================================
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Mac Pro 7,1 LLM Inference Setup (Vulkan/MoltenVK)           ║"
echo "║  GPU-Accelerated AI on AMD Radeon Pro Vega II                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_step "Step 1/5: Validating environment..."

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    log_error "This script is designed for macOS. Exiting."
    exit 1
fi
log_ok "macOS detected: $(sw_vers -productVersion)"

# Check architecture (Intel)
ARCH=$(uname -m)
if [[ "${ARCH}" != "x86_64" ]]; then
    log_warn "Detected ${ARCH} — this script targets Intel Mac Pro (x86_64)."
    log_warn "For Apple Silicon, use Metal or MLX instead of Vulkan."
fi
log_ok "Architecture: ${ARCH}"

# Check for AMD GPU
if system_profiler SPDisplaysDataType 2>/dev/null | grep -qi "Vega"; then
    GPU_INFO=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | sed 's/.*: //')
    log_ok "AMD GPU detected: ${GPU_INFO}"
else
    log_warn "No AMD Vega GPU detected. Vulkan backend may not work as expected."
fi

# Check CPU cores for parallel build
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
log_ok "CPU cores available for build: ${NCPU}"

#===============================================================================
# STEP 2: Install dependencies via Homebrew
#===============================================================================
log_step "Step 2/5: Installing dependencies..."

if ! check_cmd brew; then
    log_error "Homebrew not found. Install it from https://brew.sh"
    exit 1
fi

DEPS=(cmake vulkan-headers vulkan-loader molten-vk shaderc)
MISSING_DEPS=()

for dep in "${DEPS[@]}"; do
    if brew list "${dep}" &>/dev/null; then
        log_ok "${dep} already installed"
    else
        MISSING_DEPS+=("${dep}")
    fi
done

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Installing missing packages: ${MISSING_DEPS[*]}${NC}"
    brew install "${MISSING_DEPS[@]}"
    log_ok "All dependencies installed"
else
    log_ok "All dependencies already present"
fi

#===============================================================================
# STEP 3: Clone and build llama.cpp with Vulkan backend
#===============================================================================
log_step "Step 3/5: Building llama.cpp with Vulkan backend..."

if [[ -d "${LLAMA_CPP_DIR}" ]]; then
    log_warn "llama.cpp directory exists at ${LLAMA_CPP_DIR}"
    echo -e "${YELLOW}Pulling latest changes...${NC}"
    cd "${LLAMA_CPP_DIR}"
    git pull --ff-only 2>/dev/null || log_warn "Could not pull (may have local changes)"
else
    echo -e "${YELLOW}Cloning llama.cpp...${NC}"
    mkdir -p "$(dirname "${LLAMA_CPP_DIR}")"
    git clone "${LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
    cd "${LLAMA_CPP_DIR}"
fi

# Clean previous build if exists
if [[ -d "build" ]]; then
    log_warn "Removing previous build directory..."
    rm -rf build
fi

# Configure with Vulkan ON, Metal OFF (critical for AMD GPUs on Intel Mac)
echo -e "${YELLOW}Configuring cmake (Vulkan=ON, Metal=OFF)...${NC}"
cmake -B build \
    -DGGML_VULKAN=ON \
    -DGGML_METAL=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Build using all available cores
echo -e "${YELLOW}Building with ${NCPU} cores...${NC}"
cmake --build build --config Release -j "${NCPU}"

# Verify build artifacts
if [[ -f "build/bin/llama-cli" ]] && [[ -f "build/bin/llama-server" ]]; then
    log_ok "Build successful!"
    log_ok "  llama-cli:    ${LLAMA_CPP_DIR}/build/bin/llama-cli"
    log_ok "  llama-server: ${LLAMA_CPP_DIR}/build/bin/llama-server"
else
    log_error "Build failed — expected binaries not found."
    exit 1
fi

# Verify Vulkan library was built
if [[ -f "build/bin/libggml-vulkan.dylib" ]]; then
    log_ok "Vulkan backend library: build/bin/libggml-vulkan.dylib"
else
    log_error "Vulkan backend library not found. Check cmake output above."
    exit 1
fi

#===============================================================================
# STEP 4: Download model
#===============================================================================
log_step "Step 4/5: Preparing model..."

HF_MODEL_TAG="${MODEL}:${QUANT}"
echo -e "${YELLOW}Model: ${HF_MODEL_TAG}${NC}"
echo -e "${YELLOW}The model will be downloaded on first run if not already cached.${NC}"

#===============================================================================
# STEP 5: Launch inference
#===============================================================================
log_step "Step 5/5: Launching inference..."

if [[ "${SERVER_MODE}" == true ]]; then
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Starting OpenAI-Compatible API Server                       ║"
    echo "║  Endpoint: http://localhost:${SERVER_PORT}/v1                ║"
    echo "║                                                              ║"
    echo "║  Test with:                                                  ║"
    echo "║  curl http://localhost:${SERVER_PORT}/v1/chat/completions \\ ║"
    echo "║    -H 'Content-Type: application/json' \\                    ║"
    echo "║    -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",║"
    echo "║         \"content\":\"Hello!\"}]}'                           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    "${LLAMA_CPP_DIR}/build/bin/llama-server" \
        -hf "${HF_MODEL_TAG}" \
        -ngl "${NUM_GPU_LAYERS}" \
        -c "${CONTEXT_SIZE}" \
        --repeat-penalty "${REPEAT_PENALTY}" \
        --no-mmproj \
        --port "${SERVER_PORT}" \
        --host 0.0.0.0
else
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Starting Interactive Chat (Vulkan GPU-Accelerated)          ║"
    echo "║  Type your message at the > prompt                           ║"
    echo "║  Type /exit or Ctrl+C to quit                                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    "${LLAMA_CPP_DIR}/build/bin/llama-cli" \
        -hf "${HF_MODEL_TAG}" \
        -ngl "${NUM_GPU_LAYERS}" \
        -c "${CONTEXT_SIZE}" \
        --repeat-penalty "${REPEAT_PENALTY}"
fi
