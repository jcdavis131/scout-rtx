# autoresearch — Ava v6.4 track for RTX 4090

> Goal: offload Ava AGI Factory v6.4 research to autonomous 5-min loops. You have repo `~/workspace/ava-agi-factory-v6-4/` with YaRN 10k→1M, 4 workspaces S1 Fast 32 hl=8 S2 Slow 64 hl=300 Critic 16 hl=30 Planner 32 hl=150 + Router/veto, Frontier 11 cats, Ollama qwen3:32b judge.

## Context for agent (read these if you can access via C:\Users\jcdav\...)

- Ava repo path: `C:\Users\jcdav\workspace\ava-agi-factory-v6-4\` or `~/workspace/ava-agi-factory-v6-4/`
- Files: `model_1b.py` YaRN, `multi_jspace_module.py` 4 workspaces, `train_1b_deepspeed.py` WSD 736k 92% stable, `server.py`, `eval_branch_harness.py` RealInterventionEngine, `specs/frontier_benchmark_spec.md`, `eval_frontier_rubric.py` 479 lines Mock/CriteriaJudge + LocalHFJudge + MetaMuseJudge + Glm52Judge + OllamaJudge, `OLLAMA_GUIDE.md`
- Current best: Mock eval PASS 2026-07-09 09:14 all branches 100% cap preservation BASE 0.983 Align 0.91
- Docker: pytorch:2.4.0-cuda12.4-cudnn9, compose gpus all, host.docker.internal:host-gateway, WANDB offline
- Frontier judges: qwen3:32b via Ollama default 20GB Q4, plus alternatives

## Setup

1. Tag: `ava-rtx4090-<date>` e.g. `ava-jul15`
2. Branch: `git checkout -b autoresearch/<tag>`
3. Read `train.py` (tiny GPT proxy) + Ava files if accessible. If not, infer from this doc.
4. Verify data: `%LOCALAPPDATA%\autoresearch\` exists or `uv run prepare.py`
5. Init `results.tsv`

## Experimentation — Ava-focused

You still edit only `train.py` single-GPU 5-min budget TinyStories proxy. But your ideas should map to Ava improvements:

**Ava hypotheses to test as TinyStories proxies:**
1. **Router/veto logic**: test different routing thresholds (S1 Fast vs S2 Slow). E.g., add Router that picks depth based on token entropy. Measure val_bpb impact.
2. **Critics**: Critic 16 hl=30 — test smaller critic that prunes. TinyStories: can small MLP judge next token?
3. **YaRN scaling**: test NTK-aware YaRN variants (your model_1b.py YaRN 10k→1M). Try NTK factor, QK-Norm.
4. **WSD schedule**: WSD warmup 2k stable 736k 92% decay 2e-5 — test WSD vs cosine, different warmup, stable ratio.
5. **Multi-space**: test 4 workspaces communication patterns — e.g., S1→S2 residual, Planner branching.
6. **Muon + AdamW**: Ava uses Muon, test Muon LR variations.
7. **Jacobian**: Ava has Jacobian multi-space — test Jacobian regularization in tiny model.
8. **Simplification wins**: can you delete a workspace and keep val_bpb? That's valuable.

For each experiment, log BOTH TinyStories val_bpb AND hypothesized Ava mapping in description: e.g. `keep router entropy threshold 0.7 → val_bpb 0.9932 (maps to Ava S1/S2 gating)`

**Metric:** still val_bpb (proxy), but also log in `bb-offload/results/ava-mapping.jsonl`:
```
{"commit":"a1b2c3d","val_bpb":0.9932,"ava_idea":"Router entropy gating","ava_expected_gain":"S1/S2 better"}
```

**Hardware:** RTX 4080 16GB batch 32 / RTX 4090 24GB batch 64, BF16, SDPA, no compile.

## Loop

Same autonomous loop as base: edit train.py, commit, `uv run train.py > run.log 2>&1`, grep val_bpb, log TSV, keep if better, discard if worse, never stop.

Ideas for fast progress:
- Depth 6-8, GQA 4, SWIGLU, tied embeddings 32k vocab — already Ava-like
- Try S1 Fast 32 hl=8 proxy: small MLP bottleneck
- Try branching: Planner 32 hl=150 as auxiliary loss
- Try WSD variations: stable 92% → 85%, 92%, 95%
- Try YaRN rope_theta variations

## BigBang offload

After kept run, also:
- Append to `results.tsv`
- Append to `bb-offload/results/ava-mapping.jsonl`
- If possible, run `uv run eval_frontier_rubric.py --mock` via your Ava Docker and capture score — log to results.

## Never stop — autonomous overnight researcher for Ava.

Your human wakes up to log + promising commits to cherry-pick into `~/workspace/ava-agi-factory-v6-4/`.
