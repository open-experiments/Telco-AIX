# Benchmark Results — Mac Pro 7,1 LLM Inference via Vulkan

Benchmark results from GPU-accelerated local inference on Mac Pro 7,1 (2019) with AMD Radeon Pro Vega II Duo GPUs using llama.cpp Vulkan backend.

## Test Environment

| Component | Spec |
|---|---|
| Machine | Mac Pro 7,1 (2019) |
| CPU | 2.7 GHz 24-Core Intel Xeon W |
| GPU | 2x AMD Radeon Pro Vega II Duo (4 GPU dies, 64 GB HBM2 total) |
| RAM | 256 GB 2400 MHz DDR4 ECC |
| OS | macOS Tahoe 26.3.1 |
| Backend | llama.cpp (Vulkan/MoltenVK, Metal OFF) |
| Server | llama-server with OpenAI-compatible API (`/v1/chat/completions`) |
| Flags | `-ngl 99 -c 4096 --repeat-penalty 1.1 --no-mmproj` |

## Models Tested

| Model | Type | Active Params | Quant | VRAM Used | All Layers on GPU? |
|---|---|---|---|---|---|
| **Qwen 3.5-35B-A3B** | MoE (256 experts, 8 active) | ~3B per token | Q4_K_M | ~20 GB | Yes — across all 4 Vulkan dies |
| Meta Llama 3.1 8B Instruct | Dense | 8B | Q4_K_M | ~4.9 GB | Yes |

## Performance Summary

| Model | Prompt Eval (tok/s) | Generation (tok/s) | Avg Latency/Token | Notes |
|---|---|---|---|---|
| **Qwen 3.5-35B-A3B** (Q4_K_M) | 19.6–58.5 | **37.9–38.6** | 25.9 ms | Consistent across long generations |
| Llama 3.1 8B (Q4_K_M) | ~60 | ~50 | ~20 ms | Faster but quality issues (see below) |

Prompt eval speed varies based on cache state — 58.5 tok/s with warm cache, 19.6 tok/s on cold prompt.

## GPU Utilization

All 4 Vega II dies show balanced utilization during inference:

```
AMD Radeon Pro Vega II Duo (Slot 3, GPU 2)  —  llama-server  39.5%
AMD Radeon Pro Vega II Duo (Slot 1, GPU 2)  —  llama-server  39.5%
AMD Radeon Pro Vega II Duo (Slot 1, GPU 1)  —  llama-server  39.5%
AMD Radeon Pro Vega II Duo (Slot 3, GPU 1)  —  llama-server  39.5%
```

VRAM is distributed across all 4 Vulkan devices with minimal host memory spillover (305 MB on Host vs ~32 GB across GPUs for Qwen 3.5-35B-A3B).

---

## Test Cases

All tests run against **Qwen 3.5-35B-A3B** (Q4_K_M) via `curl` to the OpenAI-compatible API at `localhost:8080`.

### Test 1: Multi-Step Mathematical Reasoning

**Prompt:** *A telecom operator has 3 cell towers forming a triangle with sides 8km, 11km, and 15km. Place a new tower at the centroid. Calculate exact centroid coordinates if Tower A is at origin (0,0) and Tower B is at (15,0). Compute coverage overlap area assuming each tower has a 6km radius.*

**Parameters:** Default temperature

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"A telecom operator has 3 cell towers forming a triangle with sides 8km, 11km, and 15km. They want to place a new tower at the centroid to maximize overlapping coverage. Calculate the exact centroid coordinates if Tower A is at origin (0,0), Tower B is at (15,0), and Tower C position must be derived from the side lengths. Then compute the coverage overlap area assuming each tower has a 6km radius."}]}'
```

**Performance:**

| Metric | Value |
|---|---|
| Tokens generated | 3,989 |
| Generation speed | 38.62 tok/s |
| Total time | 103.4 seconds |
| Reasoning mode | Activated (natural end) |

**Result Analysis:**

The model engaged full reasoning mode (`budget=2147483647`) and produced an extensive chain-of-thought. Key accuracy observations:

- **Correct:** Identified both valid placements for Tower C using cosine rule (`C_x = (15² + 8² - 11²) / (2×15) = 5.6`, `C_y = √(64 - 31.36) ≈ 5.71`)
- **Correct:** Computed centroid as `((0 + 15 + 5.6)/3, (0 + 0 + 5.71)/3) ≈ (6.87, 1.90)`
- **Correct:** Identified that Towers A and B (15km apart) have **zero overlap** since 15km > 2×6km radius, while AC (8km) and BC (11km) pairs do overlap
- **Weakness:** The reasoning chain was excessively verbose (~4K tokens), revisiting the same calculations multiple times with self-correction loops. The math was sound but delivery was inefficient.

**Verdict:** ✅ Mathematically correct. Would benefit from `temperature: 0.3` to reduce reasoning wandering on math tasks.

---

### Test 2: Telco Domain Expertise — The NOC Paradox

**Prompt:** *"I have a paradox: My network monitoring shows 99.999% uptime but my customers report 15% call drop rate. My NOC says everything is green. My revenue assurance team says we are losing $2M/month from silent failures. Explain at least 5 specific technical scenarios where all of these statements can simultaneously be true, and for each one, tell me exactly which monitoring probe or KPI I am missing."*

**Parameters:** `temperature: 0.6`

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"I have a paradox: My network monitoring shows 99.999% uptime but my customers report 15% call drop rate. My NOC says everything is green. My revenue assurance team says we are losing $2M/month from silent failures. Explain at least 5 specific technical scenarios where all of these statements can simultaneously be true, and for each one, tell me exactly which monitoring probe or KPI I am missing."}],"temperature":0.6}'
```

**Performance:**

| Metric | Value |
|---|---|
| Tokens generated | 3,165 |
| Generation speed | 37.86 tok/s |
| Total time | 83.7 seconds |
| Reasoning mode | Activated → natural end |

**Result Analysis:**

The model immediately framed the core insight: **"Infrastructure Green" vs "Service Red"** — monitoring measures the health of L1-L3 "pipes" and "signaling logic" but misses L4-L7 "payload delivery" and "customer experience." This is textbook telco operations reasoning.

Five scenarios identified, each with specific missing KPIs:

| # | Scenario | Root Cause | Missing KPI |
|---|---|---|---|
| 1 | **Signaling Success / Media Plane Failure** | SS7/SIP/Diameter signaling green but RTP media path blocked (firewall/NAT). Call connects, audio never flows, drops after 10-20s timeout. | RTP Packet Loss per Call, MOS Score |
| 2 | **Interconnect Peering / Silent Discard** | Your network healthy but peering partners silently drop packets. NOC only monitors your side. | Interconnect Packet Loss Rate, Inter-Operator Settlement Reconciliation |
| 3 | **Handover Context Transfer Failures** | eNodeB/gNodeB up with no alarms, but Cell A → Cell B handover fails due to signaling race condition or MME/AMF database latency. | Handover Success Rate (HOSR), RRC Failure Rate per Sector |
| 4 | **QoS Policing Misconfiguration (QCI/5QI Mismatch)** | "Committed" utilization looks normal, but QoS markings misconfigured for VoLTE. Voice packets deprioritized under load. | QoS Class-specific Drop Rate, Jitter & Latency per QoS Flow |
| 5 | **Billing/CRM Correlation Gap** | Billing triggers on "Call Connected" (signaling), not "Media Delivered." Customers billed for silent/dropped calls → refunds → $2M/month loss. | CDR Accuracy, Call Termination Reason Code Distribution |
| 6 | **Soft Errors / BER (bonus)** | Physical layer bit error rate causing packet loss without triggering link-down alarms. | Per-link BER monitoring, 2G/3G/4G/5G signaling protocol-level KPIs |

**Verdict:** ✅ Excellent domain expertise. Every scenario is technically valid, operationally realistic, and directly answers "which probe/KPI am I missing." The model correctly identified the L1-L3 vs L4-L7+ monitoring gap — the kind of reasoning a senior NOC architect would produce.

---

## Model Comparison Notes

### Qwen 3.5-35B-A3B (MoE) — Recommended

- **Quality:** Strong reasoning, solid domain knowledge, handles multi-step problems well
- **Speed:** 37.9–38.6 tok/s — consistent across short and long generations
- **VRAM:** ~20 GB Q4_K_M — fits comfortably across 4 Vega II dies with room to spare
- **Thinking mode:** Uses `<think>` reasoning chains that activate automatically; ends naturally on both math and domain tasks
- **Tradeoff:** MoE architecture means only ~3B params active per token, but the 256-expert pool delivers surprisingly deep knowledge

### Llama 3.1 8B Instruct — Faster But Problematic

- **Quality:** Produced off-topic responses and persistent repetition loops on this hardware
- **Speed:** ~50 tok/s — faster due to smaller model, but quality didn't justify the speed advantage
- **Issues:** Required explicit `--repeat-penalty 1.1` to avoid infinite loops. Chat template auto-detection had inconsistent behavior. Likely a Vulkan backend floating-point precision interaction with the model's sampling.
- **Verdict:** Not recommended as the primary model on this setup. Use Qwen 3.5-35B-A3B instead.

---

## Reproducing These Benchmarks

```bash
# 1. Build llama.cpp with Vulkan (see setup-llminfra-macpro.sh)
./setup-llminfra-macpro.sh --server

# 2. Run test cases against the server
# Test 1: Math reasoning
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"A telecom operator has 3 cell towers forming a triangle with sides 8km, 11km, and 15km. They want to place a new tower at the centroid to maximize overlapping coverage. Calculate the exact centroid coordinates if Tower A is at origin (0,0), Tower B is at (15,0), and Tower C position must be derived from the side lengths. Then compute the coverage overlap area assuming each tower has a 6km radius."}]}'

# Test 2: Telco domain paradox
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"I have a paradox: My network monitoring shows 99.999% uptime but my customers report 15% call drop rate. My NOC says everything is green. My revenue assurance team says we are losing $2M/month from silent failures. Explain at least 5 specific technical scenarios where all of these statements can simultaneously be true, and for each one, tell me exactly which monitoring probe or KPI I am missing."}],"temperature":0.6}'
```

---

## Key Takeaways

1. **Qwen 3.5-35B-A3B is the sweet spot** for this hardware — MoE architecture delivers strong quality at ~38 tok/s while fitting entirely in GPU VRAM
2. **Vulkan/MoltenVK is production-viable** on Intel Mac + AMD GPU for local LLM serving at interactive speeds
3. **`--repeat-penalty 1.1` is essential** — Vulkan backend precision on Vega II can cause sampling loops without it
4. **All 4 GPU dies are utilized** with balanced load distribution — no wasted silicon
5. **64 GB HBM2 headroom** means larger models (70B+ at lower quant) are feasible with partial GPU/RAM split

*Benchmarked: April 2026 | Hardware: Mac Pro 7,1 | Model: Qwen 3.5-35B-A3B Q4_K_M | Backend: llama.cpp Vulkan*
