#!/usr/bin/env bash
#===============================================================================
# run-model.sh
#
# Quick model launcher for Mac Pro 7,1 (Intel + AMD Vega II) LLM inference.
# Loads GGUF models from HuggingFace via llama.cpp Vulkan backend.
#
# Prerequisites: Run setup-llminfra-macpro.sh first to build llama.cpp.
#
# Usage:
#   ./run-model.sh                     # Interactive model picker
#   ./run-model.sh --list              # List curated models with VRAM estimates
#   ./run-model.sh --pick 1            # Load model #1 from catalog
#   ./run-model.sh --hf "user/repo" --quant Q4_K_M   # Custom HF model
#   ./run-model.sh --server --pick 3   # Launch model #3 as API server
#
# Author: Fatih E. NAR (@fenar) & Contributors
# Project: https://github.com/open-experiments/Telco-AIX
#===============================================================================
set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/projects/llama.cpp}"
SERVER_MODE=false
SERVER_PORT=8080
NUM_GPU_LAYERS=99
CONTEXT_SIZE=4096
REPEAT_PENALTY=1.1
PICK=""
CUSTOM_HF=""
CUSTOM_QUANT="Q4_K_M"
LIST_ONLY=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

#-------------------------------------------------------------------------------
# Model Catalog
#
# Format: "HF_REPO|QUANT|DISPLAY_NAME|TYPE|ACTIVE_PARAMS|VRAM_EST|NOTES"
# VRAM_EST is approximate for the given quantization.
# Models are ordered by VRAM usage (smallest first).
#
# To add new models: append to this array following the same format.
# When Qwen 3.6 drops, just add the entry here.
#-------------------------------------------------------------------------------
MODELS=(
    #--- Qwen 3.5 Family ---
    "bartowski/Qwen_Qwen3.5-4B-GGUF|Q4_K_M|Qwen 3.5 4B|Dense|4B|~3 GB|Fast, lightweight tasks"
    "bartowski/Qwen_Qwen3.5-9B-GGUF|Q4_K_M|Qwen 3.5 9B|Dense|9B|~6 GB|Good general purpose"
    "bartowski/Qwen_Qwen3.5-27B-GGUF|Q4_K_M|Qwen 3.5 27B|Dense|27B|~16 GB|Strong reasoning"
    "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF|Q4_K_M|Qwen 3.5 35B-A3B (MoE)|MoE|~3B active|~20 GB|★ Recommended — proven 37.6 tok/s"
    "bartowski/Qwen_Qwen3.5-9B-GGUF|Q8_0|Qwen 3.5 9B (Q8)|Dense|9B|~10 GB|Higher quality, slightly slower"
    "bartowski/Qwen_Qwen3.5-27B-GGUF|Q8_0|Qwen 3.5 27B (Q8)|Dense|27B|~29 GB|High quality reasoning"
    "bartowski/Qwen_Qwen3.5-122B-A10B-GGUF|Q4_K_M|Qwen 3.5 122B-A10B (MoE)|MoE|~10B active|~62 GB|Fits in VRAM — large MoE with headroom"

    #--- Kimi Family ---
    "bartowski/moonshotai_Kimi-K2-Instruct-0905-GGUF|IQ2_XXS|Kimi K2 Instruct (IQ2)|MoE|~32B active|~120 GB|1T MoE — needs RAM spillover"
    "bartowski/moonshotai_Kimi-K2-Thinking-GGUF|IQ2_XXS|Kimi K2 Thinking (IQ2)|MoE|~32B active|~120 GB|1T MoE w/ reasoning — needs RAM spillover"

    #--- Llama Family ---
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Q4_K_M|Llama 3.1 8B|Dense|8B|~5 GB|Fast but may have quality issues on Vulkan"
    "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF|Q4_K_M|Llama 3.1 70B|Dense|70B|~40 GB|Large dense — partial GPU offload"

    #--- DeepSeek Family ---
    "bartowski/DeepSeek-R1-0528-Qwen3-8B-GGUF|Q4_K_M|DeepSeek R1 Qwen3 8B|Dense|8B|~5 GB|Reasoning-tuned on Qwen3 base"

    #--- Mistral / Mixtral ---
    "bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF|Q4_K_M|Mistral Small 3.2 24B|Dense|24B|~14 GB|Solid multilingual"
)

#-------------------------------------------------------------------------------
# Parse arguments
#-------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --list|-l)      LIST_ONLY=true; shift ;;
        --pick|-p)      PICK="$2"; shift 2 ;;
        --hf)           CUSTOM_HF="$2"; shift 2 ;;
        --quant|-q)     CUSTOM_QUANT="$2"; shift 2 ;;
        --server|-s)    SERVER_MODE=true; shift ;;
        --port)         SERVER_PORT="$2"; shift 2 ;;
        --ctx)          CONTEXT_SIZE="$2"; shift 2 ;;
        --ngl)          NUM_GPU_LAYERS="$2"; shift 2 ;;
        --repeat-penalty) REPEAT_PENALTY="$2"; shift 2 ;;
        --llama-dir)    LLAMA_CPP_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Model Selection:"
            echo "  --list, -l              List all curated models with VRAM estimates"
            echo "  --pick N, -p N          Load model #N from the catalog"
            echo "  --hf REPO               Load custom HuggingFace model (e.g., 'user/repo-GGUF')"
            echo "  --quant Q, -q Q         Quantization for custom model (default: Q4_K_M)"
            echo ""
            echo "Runtime Options:"
            echo "  --server, -s            Launch as OpenAI-compatible API server"
            echo "  --port PORT             Server port (default: 8080)"
            echo "  --ctx SIZE              Context window size (default: 4096)"
            echo "  --ngl LAYERS            GPU layers to offload (default: 99)"
            echo "  --repeat-penalty P      Repeat penalty (default: 1.1)"
            echo "  --llama-dir DIR         Path to llama.cpp (default: ~/projects/llama.cpp)"
            echo ""
            echo "Examples:"
            echo "  $0                              # Interactive picker"
            echo "  $0 --list                       # Show model catalog"
            echo "  $0 --pick 4                     # Load Qwen 3.5 35B-A3B (recommended)"
            echo "  $0 --pick 4 --server            # Same, as API server"
            echo "  $0 --hf 'unsloth/Qwen3.5-9B-GGUF' --quant Q5_K_M"
            echo "  $0 --hf 'bartowski/moonshotai_Kimi-K2.5-GGUF' --quant IQ2_XXS"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

#-------------------------------------------------------------------------------
# Validate llama.cpp installation & Vulkan backend
#-------------------------------------------------------------------------------
CLI_BIN="${LLAMA_CPP_DIR}/build/bin/llama-cli"
SRV_BIN="${LLAMA_CPP_DIR}/build/bin/llama-server"
VULKAN_LIB="${LLAMA_CPP_DIR}/build/bin/libggml-vulkan.dylib"
METAL_LIB="${LLAMA_CPP_DIR}/build/bin/libggml-metal.dylib"

echo -e "\n${CYAN}▸ Pre-flight checks...${NC}"

# Check binaries
if [[ ! -f "${CLI_BIN}" ]] || [[ ! -f "${SRV_BIN}" ]]; then
    echo -e "${RED}✖ llama.cpp binaries not found at ${LLAMA_CPP_DIR}/build/bin/${NC}"
    echo -e "${YELLOW}  Run setup-llminfra-macpro.sh first to build llama.cpp with Vulkan.${NC}"
    exit 1
fi
echo -e "${GREEN}✔ llama.cpp binaries found${NC}"

# CRITICAL: Verify Vulkan backend was compiled
if [[ ! -f "${VULKAN_LIB}" ]]; then
    echo -e "${RED}✖ Vulkan backend NOT found: ${VULKAN_LIB}${NC}"
    echo -e "${RED}  llama.cpp was built WITHOUT Vulkan — inference will fall back to CPU!${NC}"
    echo -e "${YELLOW}  Rebuild with: cmake -B build -DGGML_VULKAN=ON -DGGML_METAL=OFF${NC}"
    exit 1
fi
echo -e "${GREEN}✔ Vulkan backend: $(basename "${VULKAN_LIB}")${NC}"

# CRITICAL: Warn if Metal backend is also present (it takes priority and breaks AMD GPUs)
if [[ -f "${METAL_LIB}" ]]; then
    echo -e "${RED}✖ Metal backend ALSO found: ${METAL_LIB}${NC}"
    echo -e "${RED}  Metal takes priority over Vulkan and produces garbage on AMD GPUs!${NC}"
    echo -e "${YELLOW}  Rebuild with: cmake -B build -DGGML_VULKAN=ON -DGGML_METAL=OFF${NC}"
    exit 1
fi
echo -e "${GREEN}✔ Metal backend absent (correct for AMD GPU)${NC}"

# Check MoltenVK / Vulkan loader availability
if command -v vulkaninfo &>/dev/null; then
    VULKAN_DEVS=$(vulkaninfo --summary 2>/dev/null | grep -c "GPU" || true)
    if [[ "${VULKAN_DEVS}" -gt 0 ]]; then
        echo -e "${GREEN}✔ Vulkan devices detected: ${VULKAN_DEVS} GPU(s)${NC}"
        vulkaninfo --summary 2>/dev/null | grep "deviceName" | while read -r line; do
            echo -e "  ${DIM}${line}${NC}"
        done
    else
        echo -e "${YELLOW}⚠ vulkaninfo found no GPUs — MoltenVK may not be configured${NC}"
    fi
elif [[ -d "/opt/homebrew/share/vulkan" ]] || [[ -d "/usr/local/share/vulkan" ]]; then
    echo -e "${GREEN}✔ Vulkan/MoltenVK libraries present via Homebrew${NC}"
else
    echo -e "${YELLOW}⚠ Cannot verify Vulkan devices (vulkaninfo not in PATH)${NC}"
    echo -e "${DIM}  Install with: brew install vulkan-tools (optional, for diagnostics)${NC}"
fi

# Display VRAM summary
echo -e "${GREEN}✔ Target: 128 GB HBM2 across 4 Vega II dies (32 GB/die)${NC}"
echo ""

#-------------------------------------------------------------------------------
# Display model catalog
#-------------------------------------------------------------------------------
print_catalog() {
    echo -e "\n${CYAN}${BOLD}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║  Model Catalog — Mac Vulkan Inference                              ║${NC}"
    echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════════════╝${NC}\n"

    printf "  ${BOLD}%-4s %-38s %-6s %-12s %-10s${NC}\n" "#" "Model" "Type" "Active" "VRAM Est."
    printf "  %-4s %-38s %-6s %-12s %-10s\n" "----" "--------------------------------------" "------" "------------" "----------"

    local i=1
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r _repo _quant name type active vram notes <<< "${entry}"

        # Color code by VRAM fit
        local color="${GREEN}"
        local vram_num
        vram_num=$(echo "${vram}" | grep -oE '[0-9]+')
        if [[ ${vram_num} -gt 128 ]]; then
            color="${RED}"
        elif [[ ${vram_num} -gt 100 ]]; then
            color="${YELLOW}"
        fi

        local star=""
        if echo "${notes}" | grep -q "★"; then
            star=" ★"
        fi

        printf "  ${color}%-4s %-38s %-6s %-12s %-10s${NC}${DIM} %s${NC}\n" \
            "[${i}]" "${name}" "${type}" "${active}" "${vram}" "${notes}"

        i=$((i + 1))
    done

    echo ""
    echo -e "  ${GREEN}■${NC} Fits in VRAM    ${YELLOW}■${NC} Tight fit    ${RED}■${NC} Needs RAM spillover"
    echo -e "  ${DIM}VRAM pool: 128 GB HBM2 across 4 Vega II dies (32 GB/die) + 256 GB system RAM for spillover${NC}"
    echo ""
}

#-------------------------------------------------------------------------------
# List mode
#-------------------------------------------------------------------------------
if [[ "${LIST_ONLY}" == true ]]; then
    print_catalog
    exit 0
fi

#-------------------------------------------------------------------------------
# Model selection
#-------------------------------------------------------------------------------
HF_MODEL_TAG=""

if [[ -n "${CUSTOM_HF}" ]]; then
    # Custom HuggingFace model
    HF_MODEL_TAG="${CUSTOM_HF}:${CUSTOM_QUANT}"
    echo -e "${CYAN}▸ Custom model: ${HF_MODEL_TAG}${NC}"

elif [[ -n "${PICK}" ]]; then
    # Pick from catalog
    if [[ "${PICK}" -lt 1 ]] || [[ "${PICK}" -gt ${#MODELS[@]} ]]; then
        echo -e "${RED}✖ Invalid pick: ${PICK}. Valid range: 1–${#MODELS[@]}${NC}"
        exit 1
    fi
    local_entry="${MODELS[$((PICK - 1))]}"
    IFS='|' read -r repo quant name type active vram notes <<< "${local_entry}"
    HF_MODEL_TAG="${repo}:${quant}"
    echo -e "${CYAN}▸ Selected: ${name} (${type}, ${active} active, ${vram})${NC}"
    echo -e "${DIM}  ${notes}${NC}"

else
    # Interactive picker
    print_catalog
    echo -e "${BOLD}Select a model number (or 'c' for custom HF model):${NC}"
    read -rp "> " choice

    if [[ "${choice}" == "c" ]] || [[ "${choice}" == "C" ]]; then
        read -rp "HuggingFace repo (e.g., user/model-GGUF): " custom_repo
        read -rp "Quantization [Q4_K_M]: " custom_q
        custom_q="${custom_q:-Q4_K_M}"
        HF_MODEL_TAG="${custom_repo}:${custom_q}"
        echo -e "${CYAN}▸ Custom model: ${HF_MODEL_TAG}${NC}"
    elif [[ "${choice}" -ge 1 ]] && [[ "${choice}" -le ${#MODELS[@]} ]] 2>/dev/null; then
        local_entry="${MODELS[$((choice - 1))]}"
        IFS='|' read -r repo quant name type active vram notes <<< "${local_entry}"
        HF_MODEL_TAG="${repo}:${quant}"
        echo -e "${CYAN}▸ Selected: ${name} (${type}, ${active} active, ${vram})${NC}"
        echo -e "${DIM}  ${notes}${NC}"
    else
        echo -e "${RED}✖ Invalid selection.${NC}"
        exit 1
    fi

    # Ask launch mode if not set via flag
    if [[ "${SERVER_MODE}" == false ]]; then
        echo ""
        echo -e "${BOLD}Launch mode:${NC}"
        echo -e "  ${GREEN}[1]${NC} Interactive Chat (CLI)"
        echo -e "  ${GREEN}[2]${NC} OpenAI-Compatible API Server (http://localhost:${SERVER_PORT}/v1)"
        read -rp "> " mode_choice
        if [[ "${mode_choice}" == "2" ]]; then
            SERVER_MODE=true
        fi
    fi
fi

#-------------------------------------------------------------------------------
# Launch inference
#-------------------------------------------------------------------------------
echo ""

if [[ "${SERVER_MODE}" == true ]]; then
    echo -e "${CYAN}${BOLD}╔═════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║  Launching OpenAI-Compatible API Server                     ║${NC}"
    echo -e "${CYAN}${BOLD}║  Endpoint: http://localhost:${SERVER_PORT}/v1               ║${NC}"
    echo -e "${CYAN}${BOLD}╚═════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${DIM}  Test: curl http://localhost:${SERVER_PORT}/v1/chat/completions \\${NC}"
    echo -e "${DIM}    -H 'Content-Type: application/json' \\${NC}"
    echo -e "${DIM}    -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'${NC}"
    echo ""

    exec "${SRV_BIN}" \
        -hf "${HF_MODEL_TAG}" \
        -ngl "${NUM_GPU_LAYERS}" \
        -c "${CONTEXT_SIZE}" \
        --repeat-penalty "${REPEAT_PENALTY}" \
        --no-mmproj \
        --reasoning-format deepseek \
        --port "${SERVER_PORT}" \
        --host 0.0.0.0
else
    echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║  Launching Interactive Chat (Vulkan GPU-Accelerated)       ║${NC}"
    echo -e "${CYAN}${BOLD}║  Type your message at the > prompt                         ║${NC}"
    echo -e "${CYAN}${BOLD}║  Type /exit or Ctrl+C to quit                              ║${NC}"
    echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    exec "${CLI_BIN}" \
        -hf "${HF_MODEL_TAG}" \
        -ngl "${NUM_GPU_LAYERS}" \
        -c "${CONTEXT_SIZE}" \
        --repeat-penalty "${REPEAT_PENALTY}" \
        --no-mmproj
fi
