# Dataset Provenance

All benchmark datasets in this directory are **verbatim snapshots** taken on
**2026-08-07/08** from the GSMA Open Telco evaluation datasets on Hugging Face,
converted from their original formats (JSON / Parquet) to gzipped JSONL with
no content modification.

| Tier | Source (Hugging Face) | Snapshot date | Records |
|---|---|---|---|
| `lite/` | `GSMA/ot-lite` (JSON files) | 2026-08-07 | teleqna 1,000 · teletables 100 · telemath 100 · telelogs 100 · three_gpp 100 · oranbench 150 · srsranbench 150 · sixg_bench 150 |
| `full/` | `GSMA/ot-full` (Parquet files) | 2026-08-08 | teleqna 10,000 · teletables 500 · telemath 500 · telelogs 864 · three_gpp 2,000 · oranbench 1,500 · srsranbench 1,502 · sixg_bench 3,722 |

The lite tier is the default sample set used by the
[GSMA Open Telco Leaderboard](https://huggingface.co/spaces/GSMA/open-telco-leaderboard)
harness ([gsma-labs/evals](https://github.com/gsma-labs/evals)).

Upstream benchmark sources credited by GSMA: netop/TeleQnA, TeleTables,
TeleMath, TeleLogs (Huawei GTS et al.), prnshv/ORANBench, prnshv/srsRANBench,
3GPP-TSG, 6G-Bench. Licenses follow the respective upstream dataset licenses;
these copies exist to guarantee long-term reproducibility of Telco-AIX
benchmark results independent of upstream availability.

Record schemas (per line of JSONL):
- MCQ tasks (`teleqna`, `teletables`, `oranbench`, `srsranbench`, `sixg_bench`):
  `question` (str), `choices` (list[str]), `answer` (int index), plus
  task-specific metadata fields.
- `telemath`, `telelogs`, `three_gpp`: `question` (str), `answer` (str), plus
  metadata.

A snapshot of the leaderboard's published scores as of 2026-08-07 is kept in
`../reference/leaderboard_scores_2026-08-07.csv` (and the newer parquet-backed
snapshot including the OTel-2.0-LLM-31B-IT rank-1 entry in
`../reference/leaderboard_scores_parquet_2026-08-07.csv`).
