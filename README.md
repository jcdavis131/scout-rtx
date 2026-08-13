# Scout RTX Offload

> Solo personal project, no connection to employer, built with public/free-tier only. Fork of [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx).

Autonomous experiment runner for a local Windows RTX 4080/4090 box. A cloud session queues tasks; the GPU box runs hill-climb training experiments and publishes results (`results.tsv` / `results.jsonl`) as GitHub releases, which the [scout-cli](https://github.com/jcdavis131/scout-cli) dashboard reads back. Integrated as the `scout rtx` plugin.

Status: experimental, tuned to one specific machine.

## Differences from upstream

- GPU presets for RTX 4080 16GB and RTX 4090 24GB instead of generic 8/10GB.
- Four experiment programs (`programs/`): base bpb hill-climb, Ava model experiments, an ONNX turnover detector, and a writing-detector weight search.
- Cloud-to-local task offload via `bb-offload/queue.json`.
- Auto-publish of results as GitHub releases, plus a dashboard that polls them.

## Quickstart (Windows PowerShell, on the GPU box)

```powershell
git clone https://github.com/jcdavis131/scout-rtx.git
cd scout-rtx
.\scripts\setup-win.ps1 -Program programs\program-ava.md -Tag scout-ava
.\scripts\run-autonomous.ps1 -Program programs\program-ava.md -Tag scout-ava -MaxExperiments 20
# publishes a results release every 5 experiments and at the end
```

## Run the trainer once (smoke test)

The quickstart above drives a 20-experiment autonomous loop that publishes GitHub releases. To just prove the trainer works on your box — no token, no releases, a few minutes:

```powershell
$env:AUTORESEARCH_OUT_DIR = "$env:TEMP\scout-out"   # keep the checkpoint out of the checkout
uv sync                        # deps from pyproject.toml; no lockfile is committed
uv run prepare.py              # ~1 GB TinyStories parquet + BPE tokenizer
uv run train.py --smoke-test   # 3 optimizer steps, then a short bpb eval
```

`setup-win.ps1` runs the same sync/prepare/smoke sequence during setup (it skips `prepare.py` under `-SkipSetup`, and it sets no out-dir, so its smoke test leaves `checkpoint_pre_eval.pt` in the checkout); the block above is the standalone version. `prepare.py` is idempotent — re-runs return immediately once the parquet and tokenizer exist — and `train.py` does **not** call it for you.

**A CUDA device is required.** `detect_runtime` raises `CUDA is required. No CUDA device detected.` before any work. There is no CPU path.

**Set `AUTORESEARCH_OUT_DIR`.** `train.py` saves `checkpoint_pre_eval.pt` on every run, smoke test included, and defaults to the working directory — so without it a measuring run drops a checkpoint into your checkout. `AUTORESEARCH_CACHE_DIR` likewise relocates the dataset and tokenizer (default: `%LOCALAPPDATA%\autoresearch` on Windows unless a legacy `~/.cache/autoresearch` already exists, which wins; `~/.cache/autoresearch` elsewhere).

### Reading the output

Four things look broken and are not:

- `Warning: laptop GPUs are outside the supported desktop matrix` — expected on an RTX laptop part; it selects the `compatibility` profile, which costs only autotune.
- `training_seconds: 0.0` and `mfu_percent: n/a` — `--smoke-test` stops at 3 steps and timing only accumulates past step 10, so there is no steady state to report.
- `val_bpb` is a real number from an untrained model. It is noise, not a result.
- The four `dataloader_percent` / `gpu_wait_percent` / `other_percent` / `loop_bound` lines print **twice with identical values** — once under `[loop diagnostic]` the moment training returns, once in the final summary. The early copy exists so an eval failure or a timeout cannot discard the measurement. A log with the `[loop diagnostic]` block but no final summary is a partial success.

`loop_bound` is the one number worth reading: it answers whether the loop is starved by CPU-side batch building, and no GPU-side tuning (batch size, checkpointing, MFU) is worth proposing until it says otherwise. `docs/HARDWARE_PROFILE.md` explains how the split is measured and how far it can be trusted.

The containerized GPU lane runs the same two entrypoints in the same order, launched with `pip` and plain `python` instead of `uv`; `herdmux.train.json` is that command, its mounts and its expected output.

## Cloud-side commands

```bash
scout rtx status                      # queue depth, best val_bpb so far
scout rtx queue add --task "..." --program programs/program-ava.md
scout rtx releases list               # published result releases
scout rtx releases sync --tag <tag>
scout rtx results --best
scout rtx dashboard
```

## Result provenance

The end-to-end pipeline was verified 2026-07-15 with a demo-seeded release (`v0.6.0-demo-0715`). The 0.9935 val_bpb in that release is a synthetic demo value, not a real training result. Publish scripts refuse to fabricate rows: they exit if `results.tsv` is missing unless an explicit `-Demo`/`--demo` flag is passed, which tags the row `status=demo`.

Training stops on a wall-clock budget rather than a step count, so how much data a run saw depends on how fast the box happened to be — a lower `val_bpb` can mean a better model or just a faster machine-hour. Each row in `bb-offload/results/results.jsonl` therefore carries `num_steps` and `training_seconds` alongside the score; two scores are comparable when those match. (`results.tsv` keeps its five columns, since scout-cli parses them positionally.)

A crashed experiment is recorded with `val_bpb 0` and `status=crash`. Lower bpb is better, so 0 would otherwise be the best value the column can hold — every reader of "best val_bpb" (both publish scripts, `scout rtx status`, `scout rtx results --best`, `scout rtx sync`) therefore skips non-positive values. A run of nothing but crashes reports `unknown`, not a perfect score.

The offload queue (`bb-offload/queue.json`) is hand-copied to the GPU box, so it can arrive truncated. `scout rtx queue add` and `queue list` refuse to read a corrupt queue as an empty one — `add` exits non-zero without touching the file rather than overwriting the pending tasks it could not parse. `queue clear` still overwrites, since discarding is its job, but reports that it did.

Dashboard server code (actions, schema, cron sync) lives in [scout-cli](https://github.com/jcdavis131/scout-cli), not in this repo. The upstream project's original README is preserved as `README.upstream.md`.

Solo personal project, no connection to employer, built with public/free-tier only.
