# Telco-AIX Self-Contained Telco LLM Eval Framework

A zero-external-dependency benchmark harness for telecom LLMs. Everything
needed to evaluate a model — datasets, prompts, scorers, runner — lives in
this folder. If every upstream source (the GSMA leaderboard, the
`gsma-labs/evals` repo, the Hugging Face datasets) disappeared tomorrow,
these benchmarks still run, byte-for-byte identical.

Born out of the **GSMA Open Telco Leaderboard verification exercise**
(2026-08-07, see `reference/GSMA-OTel2-Leaderboard-Verification-Report_2026-08-07.md`),
where we measured the public OTel-2.0-LLM-31B-IT checkpoint at 0.625
average against its claimed rank-1 score of 0.903 — and learned the hard way
why pinned, self-contained, independently runnable evals matter.

## What's included

```
evals/
├── otel_eval.py          # single-file runner (Python 3.9+, needs only `requests`)
├── datasets/
│   ├── lite/             # GSMA ot-lite snapshot (leaderboard default) — 8 tasks, gzipped JSONL
│   ├── full/             # GSMA ot-full snapshot — 8 tasks incl. TeleQnA 10k
│   └── PROVENANCE.md     # exact snapshot sources, dates, schemas
├── reference/
│   ├── leaderboard_scores_2026-08-07.csv           # leaderboard claim snapshot (CSV era)
│   ├── leaderboard_scores_parquet_2026-08-07.csv   # newer snapshot incl. OTel-2.0 rank-1 entry
│   ├── venice_measured_results_2026-08-07.json     # our independently measured numbers
│   └── GSMA-OTel2-Leaderboard-Verification-Report_2026-08-07.md
└── results/              # runner output lands here (per-run folders)
```

Benchmarks: **TeleQnA · TeleTables · TeleMath · TeleLogs · 3GPP-TSG ·
ORANBench · srsRANBench** (the 7 leaderboard tasks, run by default) plus
**6G-Bench** (opt-in via `--tasks`).

The historical Telco-AIX vendor benchmark sets (Ericsson / Nokia / Mavenir
GenAI tests and their results) are preserved untouched in `../benchmarks/`.

## Quick start

```bash
pip install requests   # the only dependency

python3 otel_eval.py \
  --endpoint https://<your-model-route>/v1 \
  --model <served-model-name> \
  --max-connections 12
```

Works against any OpenAI-compatible server (vLLM, RHOAI/KServe, TGI, llama.cpp
server, OpenAI/compatible SaaS). For endpoints with self-signed certs (typical
lab OpenShift routes) add `--insecure`, or better, pass the router CA:

```bash
H=<your-model-route-host>
echo | openssl s_client -connect $H:443 -servername $H -showcerts 2>/dev/null \
  | awk '/BEGIN CERT/,/END CERT/' > lab-ca.pem
python3 otel_eval.py --endpoint https://$H/v1 --model <name> --ca-bundle lab-ca.pem
```

Useful flags:

- `--tier full` — full datasets (TeleQnA 10,000 etc.) instead of lite
- `--tasks teleqna,telemath` — subset; add `sixg_bench` for 6G-Bench
- `--limit 50` — quick smoke run
- `--api-key KEY` — sent as `Authorization: Bearer` (vLLM `--api-key`, rbac proxies)
- `--max-tokens 8192` — cap generations (recommended for long-reasoning models)
- `--extra-body '{"chat_template_kwargs":{"enable_thinking":true}}'` — provider extras
- `--temperature`, `--timeout`, `--output-dir`

Output per run: `results/<timestamp>_<model>_<tier>/` containing
`summary.json`, `SUMMARY.md` (paste-ready table), and one JSONL transcript
per task (per-sample prompt outcome, parsed answer, target, latency, tokens).

## Fidelity to the GSMA harness

Prompts and scoring are 1:1 ports of `gsma-labs/evals` (Inspect AI) as of
2026-08-07: identical multiple-choice template (`ANSWER: $LETTER`), identical
TeleMath system prompt and `\boxed{}` numeric scoring (1% tolerance),
TeleLogs "soft" first-integer scoring, and the 3GPP-TSG working-group regex
scorer. Datasets are byte-faithful snapshots (see `datasets/PROVENANCE.md`).

Parity was validated by re-running the same model (OTel-2.0-LLM-31B-IT,
revision `e120ca76`, vLLM v0.26.0, temperature 0) with both harnesses —
results agree within sampling stderr on every task (see
`reference/parity_validation_2026-08-08.md`).

## Reference numbers (measured on venice.narlabs.io, 2026-08-07)

RTX PRO 6000 Blackwell 96GB, vLLM v0.26.0, temperature 0, lite tier:

| Benchmark | OTel-2.0-31B-IT (`e120ca76`) | Gemma-4-31B-IT base (`842da379`) | Leaderboard claim (OTel-2.0) |
|---|---|---|---|
| TeleQnA | 0.795 | 0.805 | 0.917 |
| TeleTables | 0.340 | 0.350 | 0.798 |
| ORANBench | 0.787 | 0.827 | 0.936 |
| srsRANBench | 0.853 | 0.820 | 0.915 |
| TeleMath | 0.580 | 0.740 | 0.898 |
| TeleLogs | 0.420 | 0.530 | 0.982 |
| 3GPP-TSG | 0.600 | 0.470 | 0.873 |
| **Average** | **0.625** | **0.649** | **0.903** |

Full analysis in the verification report under `reference/`.

## Design notes

- **Everything embedded**: gzipped JSONL keeps the datasets ~4.5MB in-repo.
- **stdlib + requests only**: no inspect-ai, no HF hub, no pandas/pyarrow at
  runtime; nothing to bit-rot.
- **Reproducibility discipline**: always record the model *revision hash*,
  serving stack + version, temperature, tier, and date alongside results —
  the exact metadata whose absence made the leaderboard claim unverifiable.
