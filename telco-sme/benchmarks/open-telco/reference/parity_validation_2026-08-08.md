# Parity Validation — otel_eval.py vs gsma-labs/evals (Inspect AI)

**Date:** 2026-08-08
**Model under test:** OTel-2.0-LLM-31B-IT, revision `e120ca76`, served by vLLM
v0.26.0 (TP=1, BF16) on venice.narlabs.io (1× RTX PRO 6000 Blackwell 96GB).
**Config (both harnesses):** temperature 0.0, GSMA ot-lite datasets, 1 epoch.
Inspect AI 0.3.252 ran uncapped; otel_eval.py ran with its default 8,192-token
cap (no sample reached it) and streaming enabled. Zero request errors.

| Benchmark | n | otel_eval.py (this repo) | inspect-ai (gsma-labs/evals) | Δ |
|---|---|---|---|---|
| TeleQnA | 1000 | 0.791 | 0.795 | −0.004 |
| TeleTables | 100 | 0.350 | 0.340 | +0.010 |
| ORANBench | 150 | 0.793 | 0.787 | +0.007 |
| srsRANBench | 150 | 0.853 | 0.853 | 0.000 |
| TeleMath | 100 | 0.570 | 0.580 | −0.010 |
| TeleLogs | 100 | 0.430 | 0.420 | +0.010 |
| 3GPP-TSG | 100 | 0.610 | 0.600 | +0.010 |

All deltas are ≤1 percentage point (≤1 sample on the 100-sample tasks;
4 samples of 1,000 on TeleQnA) — well inside run-to-run variation for a
nondeterministic serving stack even at temperature 0 (batching order changes
floating-point reduction order). The two harnesses are measurement-equivalent.

Notes:
- Streaming (SSE) is the default transport in otel_eval.py because
  non-streaming long generations get silently killed by common middleboxes
  (kube-rbac-proxy ~30s, haproxy route timeouts, corporate egress proxies) —
  discovered empirically during the leaderboard verification runs.
- The default 8,192-token generation cap prevents pathological runaway
  chain-of-thought loops at temperature 0 (observed on hard TeleTables
  samples); no legitimate answer in this validation came near the cap.
