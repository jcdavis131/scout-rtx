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

Dashboard server code (actions, schema, cron sync) lives in [scout-cli](https://github.com/jcdavis131/scout-cli), not in this repo. The upstream project's original README is preserved as `README.upstream.md`.

Solo personal project, no connection to employer, built with public/free-tier only.
