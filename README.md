# Scout RTX Offload — Alienware RTX 4080/4090 + Scout CLI 🐾

> Solo personal project, no connection to employer, built with public/free-tier only. Fork of [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx).

**Integrated with [jcdavis131/scout-cli](https://github.com/jcdavis131/scout-cli) — primary command `scout` (aliases `bb`, `bigbang`, `kitty`, `dv`)**

## Why this custom?
| Feature | Upstream win-rtx | This custom |
|---------|------------------|-------------|
| GPU | Generic 8/10GB | Alienware RTX 4080 16GB ada-16gb batch32 / RTX 4090 24GB ada-24gb-plus batch64 |
| Programs | Single | 4 tracks: base, ava, turnover, write |
| Orchestration | Manual | Scout plugin `scout rtx` + Hatch sync + PowerShell auto-loop |
| Offload | Local only | Cloud → local via `bb-offload/queue.json` |
| Dashboard | None | `scout rtx dashboard` + web artifact |

## Quickstart — Alienware (Windows PowerShell)
```powershell
git clone https://github.com/jcdavis131/scout-rtx.git
cd scout-rtx
.\scripts\setup-win.ps1 -Program programs\program-ava.md -Tag scout-ava
.\scripts\run-autonomous.ps1 -Program programs\program-ava.md -Tag scout-ava -Iterations 20
```

## Cloud → Local offload
```bash
# In Hatch / anywhere
scout rtx status
scout rtx queue add --task "Optimize Ava router entropy 0.7" --program programs/program-ava.md
scout rtx queue list
# On Alienware, run-autonomous picks it up, writes to bb-offload/results/results.jsonl
scout rtx results --best
scout rtx dashboard
```

## Programs
- `program-base.md` — RTX baseline 5-min bpb hill-climb
- `program-ava.md` — Ava v6.4 Jacobian + Router entropy + WSD + Critic hl=30 + Frontier 11 cats
- `program-turnover.md` — Turnover Shield <2MB ONNX 120d+4heads
- `program-write.md` — evolve write detector weights participial 0.5 char 0.8 phrase 3.0

## Scout CLI Integration
Uses scout-cli v0.6.0 `scout rtx` plugin (manifest in bigbang-bridge/). Ava routes `offload to rtx` → 0.95 confidence. Agent hints wired.

Solo personal project, no connection to employer, built with public/free-tier only.
