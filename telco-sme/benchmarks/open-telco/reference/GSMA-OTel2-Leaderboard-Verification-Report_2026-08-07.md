# GSMA Open Telco Leaderboard — Verification Report

**Project:** GSMA-OTEL2-Test @ venice.narlabs.io (RHOAI 3.5.0-ea.2, OpenShift / Kubernetes v1.35)
**Date:** 2026-08-07
**Objective:** Independently verify the [GSMA Open Telco Leaderboard](https://huggingface.co/spaces/GSMA/open-telco-leaderboard) rank-1 claim for **OTel-2.0-LLM-31B-IT** by running the official [gsma-labs/evals](https://github.com/gsma-labs/evals) suite (Inspect AI) against the publicly downloadable checkpoint, with its **base model (Gemma-4-31B-IT)** evaluated under identical conditions as a control.

| | OTel-2.0-LLM-31B-IT | Base Gemma-4-31B-IT |
|---|---|---|
| Source | OTel-2.0-LLM-31B-IT (public Hugging Face checkpoint) | `google/gemma-4-31B-it` (OTel's branch point) |
| Revision | **pinned `e120ca76`** (2026-08-07 snapshot) | **pinned `842da379`** |
| Runtime | vLLM v0.26.0 (upstream `vllm-openai`), TP=1, BF16, max-model-len 65,536 | same (multimodal profiling disabled) |
| GPU | 1× RTX PRO 6000 Blackwell 96GB | 1× RTX PRO 6000 Blackwell 96GB |

## Methodology

- Official `gsma-labs/evals` tasks, **GSMA/ot-lite** sample sets (leaderboard default): TeleQnA 1,000 · TeleTables 100 · TeleMath 100 · TeleLogs 100 · 3GPP-TSG 100 · ORANBench 150 · srsRANBench 150 — 1,600 samples per model, temperature 0.0, 1 epoch, inspect-ai 0.3.252.
- Both models ran uncapped with **zero truncated generations** (avg output 9–942 tokens/task), so all scores are clean measurements.

## Results

Accuracy (± stderr). The **claim column is the leaderboard's own entry for OTel-2.0-LLM-31B-IT** — the exact model name we deployed — currently **rank #1 with a claimed 0.9027 average** (leaderboard data refreshed 2026-08-07). The base Gemma-4-31B-IT column isolates what the OTel post-training changed.

| Benchmark | Leaderboard claim: OTel-2.0-LLM-31B-IT (#1) | **OTel-2.0-31B-IT (measured)** | **Base Gemma-4-31B-IT (measured)** |
|---|---|---|---|
| TeleQnA | 0.917 ± 0.003 | 0.795 ± 0.013 | **0.805** ± 0.013 |
| TeleTables | 0.798 ± 0.018 | 0.340 ± 0.048 | **0.350** ± 0.048 |
| ORANBench | 0.936 ± 0.006 | 0.787 ± 0.034 | **0.827** ± 0.031 |
| srsRANBench | 0.915 ± 0.008 | **0.853** ± 0.029 | 0.820 ± 0.032 |
| TeleMath | 0.898 ± 0.014 | 0.580 ± 0.050 | **0.740** ± 0.044 |
| TeleLogs | 0.982 ± 0.005 | 0.420 ± 0.050 | **0.530** ± 0.050 |
| 3GPP-TSG | 0.873 ± 0.008 | **0.600** ± 0.049 | 0.470 ± 0.050 |
| **Average** | **0.903** | 0.625 | **0.649** |

Gap of measured OTel vs its claim: **−0.278 average** (worst: TeleLogs −0.562, TeleTables −0.458; every benchmark far outside combined error bars). Gap of measured OTel vs its own base model: **−0.024** (gains only on 3GPP-TSG +0.130 and srsRAN +0.033; regressions on TeleMath −0.160 and TeleLogs −0.110).

## Findings

**1. The leaderboard's rank-1 claim for OTel-2.0-LLM-31B-IT is not reproduced by the public checkpoint.** The leaderboard claims 0.9027 average for this exact model name; the publicly downloadable checkpoint (revision `e120ca76`, the only one available on 2026-08-07) measures **0.625** — a −0.278 gap, with per-benchmark shortfalls up to −0.56 (TeleLogs 0.42 measured vs 0.982 claimed) and −0.46 (TeleTables). As measured, this checkpoint would land around **rank #26–27 of ~85** (near o4-mini/gemini-2.0-flash), not #1. The most plausible reconciliations: the leaderboard evaluated a **different (newer/internal) checkpoint** than the week-0 public snapshot — consistent with the model card's "checkpoint update expected within hours / weekly weight updates" notice and the broken config.json we had to patch — and/or full datasets and a different serving/decoding configuration. Either way, **the public artifact does not support the published #1 score today**, and the claim is not reproducible without a pinned revision attached to the leaderboard entry.

**2. The tested checkpoint shows signs of being an unfinished push.** Its `config.json` lacks the `architectures` field (we had to patch in `Gemma4ForCausalLM` for vLLM to load it as a generative model), and the model card itself warns the checkpoint "is expected to be updated within the next few hours" with weekly weight updates thereafter. Any future verification should re-pin: our numbers are for revision `e120ca76` only.

**3. The base-model control shows this OTel checkpoint's post-training has not (yet) paid off on these benchmarks.** Base Gemma-4-31B-IT averages **0.649 vs OTel's 0.625** under identical serving and eval conditions. The finetune's only significant gain is 3GPP-TSG (+0.13 — standards-document work, plausibly the domain corpus showing through). Against that, OTel *regresses* on TeleMath (−0.16) and TeleLogs (−0.11) — a classic post-training trade: domain instruction data displacing general reasoning ability. TeleQnA/TeleTables/ORANBench/srsRAN deltas are within ~1–2 stderr (noise). The claim implies the finetune adds ~+0.25 over its base; we measure −0.02.

**4. Weakest OTel areas:** TeleTables (0.34) and TeleLogs (0.42) — table interpretation and log diagnostics — exactly where the leaderboard entry claims near-perfect scores (0.798 / 0.982). This is the single biggest divergence and a constructive question to raise with the leaderboard maintainers.

## Gap-Attribution Experiments

Two follow-up experiments tested the leading benign explanations for the −0.278 claim gap:

| Experiment | Hypothesis tested | Result | Verdict |
|---|---|---|---|
| TeleTables on **full** dataset (`GSMA/ot-full`, 496/500 scored) | Lite subset is hardness-curated → full runs score much higher | **0.373** vs 0.340 lite (claim: 0.798) | Explains ~+0.03 of the −0.46 gap — **not the cause** |
| TeleMath with **thinking enabled** (`enable_thinking=true` via chat template; 97/100 scored) | Claimed scores came from a thinking-mode run | **0.619** vs 0.580 without (claim: 0.898) | Explains ~+0.04 of the −0.32 gap — **not the cause** |

Notes on the thinking experiment: the checkpoint's chat template *defaults thinking off* and pre-closes the thought channel (`<|channel>thought<channel|>`), forcing direct answers. With `enable_thinking=true` the rendered prompt verifiably changes (`<|think|>` mode marker injected into the system turn — confirmed via the `/tokenize` endpoint). Yet across all 99 scored samples the model produced **zero thought-channel content** — no `reasoning_content`, no channel markers in raw text — and its answers were actually *shorter* on average than the non-thinking run (622 vs 742 output tokens). The +0.04 accuracy delta is re-run noise, not a thinking effect. In other words, **the OTel post-training appears to have rendered the base model's thinking capability inoperative**: the template permits a reasoning phase, but the finetuned weights never open the thought channel, answering directly regardless of the mode flag. This forecloses the "the leaderboard run simply used reasoning mode" defense — even if it did, the public checkpoint cannot reproduce that behavior. Base Gemma-4 still outscores OTel on TeleMath (0.740, no thinking) against OTel's thinking-enabled 0.619.

**Conclusion:** dataset tier and thinking mode together account for well under a tenth of the claim gap — and the thinking pathway is demonstrably non-functional in the public weights. By elimination, the dominant explanation is that **the leaderboard's score was produced by a different checkpoint than the public `e120ca76` snapshot** (and/or a materially different eval methodology) — reinforcing the report's central recommendation: leaderboard entries need pinned model revisions to be verifiable.

## Caveats

- Lite sample sets (leaderboard default); stderr on 100-sample tasks is ±0.03–0.05, so small deltas there are not significant.
- Single epoch, temperature 0 — matches the repo's reference `run_evals.py` configuration.
- The leaderboard's own runs may use different sample counts (stderr analysis of published scores suggests full-dataset runs for some entries).

## Deployment & Ops Notes (what was built/changed)

- **New project `gsma-otel2-test` (GSMA-OTEL2-Test)**: 150Gi PVC with pinned weights, custom ServingRuntime `custom-vllm-otel2` (vLLM v0.26.0, gemma4 parsers), InferenceService `otel2-llm-31b-it` → https://otel2-llm-31b-it-gsma-otel2-test.apps.venice.narlabs.io (57.8GiB weights on GPU, 24.75GiB KV cache, loads in ~38s from PVC).
- **Base model**: PVC `gemma4-base-model` (google/gemma-4-31B-it @ `842da379`), ServingRuntime `custom-vllm-gemma4-base`, InferenceService `gemma4-31b-it-base` → https://gemma4-31b-it-base-gsma-otel2-test.apps.venice.narlabs.io.
- **Telco-SME portal**: https://telco-sme-portal-gsma-otel2-test.apps.venice.narlabs.io (admin/minad), wired to the OTel endpoint; pinned `gradio<6` (upstream repo should backport this).
- **Config patch on PVC**: added `"architectures": ["Gemma4ForCausalLM"]` to the OTel checkpoint's config.json.
- Eval logs (Inspect `.eval` format, per-sample transcripts): `gsma-evals/logs/leaderboard-run/` (OTel) and `gsma-evals/logs/gemma4-run/` (base) in the session workspace.

## Reproduction

```bash
git clone https://github.com/gsma-labs/evals && cd evals
pip install inspect-ai openai datasets && pip install -e .
export OTEL_BASE_URL="https://otel2-llm-31b-it-gsma-otel2-test.apps.venice.narlabs.io/v1" OTEL_API_KEY=none
python3 -m inspect_ai eval-set src/evals/*/[a-z]*.py \
  --model openai-api/otel/otel2-llm-31b-it \
  --temperature 0 --max-connections 12 --log-dir logs/run
```
