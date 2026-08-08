# Telco-SME Benchmarks & Evals

All benchmark and evaluation assets of the Telco-SME experiment live here,
organized by suite. Every suite is self-contained — datasets and scoring
included — so results remain reproducible with nothing but this repository.

| Suite | What it is | How to run |
|---|---|---|
| [`open-telco/`](open-telco/) | **Self-contained Open-Telco eval framework** — the 8 GSMA telecom benchmarks (TeleQnA, TeleTables, TeleMath, TeleLogs, 3GPP-TSG, ORANBench, srsRANBench, 6G-Bench) with lite + full datasets embedded (gzipped JSONL) and a single-file runner. Parity-validated against the official Inspect AI harness (≤1pp on all 7 leaderboard tasks). Includes the 2026-08 leaderboard verification report and claim snapshots. | `cd open-telco && python3 otel_eval.py --endpoint https://<route>/v1 --model <name>` — or use the portal's **🏆 Benchmark** tab |
| [`vendor-genai-tests/`](vendor-genai-tests/) | Original Telco-AIX vendor GenAI test sets — Ericsson, Nokia, Mavenir question sets with graded result reports, plus the Telco5G GenAI benchmark PDFs and per-question Qwen3 answer reports. | Manual prompt-based testing; see `benchmark_detailed_description.md` |
| [`telcos-last-exam/`](telcos-last-exam/) | "Telco's Last Exam" — a hardest-questions telecom exam with reference answers and per-model answer sheets (frontier + open models). | Manual; compare against `answers.md` |
| [`model-reports/`](model-reports/) | Per-model benchmark answer sets and performance reports gathered on this lab (Qwen3-32B, Qwen3-30B-A3B-MoE, Seed-36B perf suite). | See each model's folder |
| [`embeddings/`](embeddings/) | Embeddings model benchmark notes. | See `benchmark.md` |

## Quick start (Open-Telco suite)

```bash
cd open-telco
pip install requests   # only dependency
python3 otel_eval.py --endpoint https://<your-model-route>/v1 --model <served-name>
# lab clusters with self-signed certs: add --insecure (or --ca-bundle <pem>)
```

Or interactively: open the SME web portal → **🏆 Benchmark** tab → pick
benchmarks → Run. Live per-task progress and accuracies stream into the UI.

## Reporting discipline

When publishing results from any suite, always record: model **revision hash**,
serving stack + version (e.g. vLLM v0.26.0), quantization/precision,
temperature, dataset tier and sample counts, and the run date. The
`open-telco/reference/` folder shows why: a leaderboard score without a pinned
revision cannot be verified (see the verification report there).
