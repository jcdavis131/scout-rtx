# Hardware Profile — Davis Alienware RTX 4080/4090

## Detected from memory

- Machine: Alienware Windows box, Chrome Windows device_id 3de351a2-90b6-47e6-8c6f-755be480367c online at 2026-07-15T00:54:38Z, plus Android
- Path: `C:\Users\jcdav\workspace\vector-hoops` etc, training log `pipeline/cache/train_full.log`
- GPU: RTX 4080/4090 — user says RTX 4080/4090, CUDA local training at C:\Users\jcdav\...
- Docker: pytorch:2.4.0-cuda12.4-cudnn9, compose gpus all, extra_hosts host.docker.internal:host-gateway, WANDB offline
- Ollama: Ollama+Docker on personal machine, qwen3:32b ~20GB Q4, deepseek-r1:32b, llama3.3:70b ~40GB optional, glm4:9b-chat
- Ollama install Windows PowerShell: winget, ollama serve, ollama pull qwen3:32b deepseek-r1:32b glm4:9b-chat

## Upstream GPU profile logic (from train.py)

In `train.py` _resolve_gpu_profile:

- Architecture detection via torch.cuda.get_device_capability()
- Turing (7,5) >=8GB VRAM
- Ampere (8,6) >=10GB
- Ada (8,9) >=10GB
- Blackwell (12,0) >=10GB

Profiles:
- Turing 8-11GB: batch (8,4,2,1), checkpoint True, eval cap 4
- Mid-tier 10-15GB: batch (16,8,4), checkpoint True, eval cap 16 (profile default)
- 16GB: batch (32,16,8,4), checkpoint modes (False,True), default False, eval cap 16
- 24GB+: batch (64,32,16,8,4), checkpoint False, eval cap 16

Tier boundaries apply a ~0.5 GB tolerance (`VRAM_TIER_TOLERANCE_GB`) because real cards
under-report total VRAM (a 16 GB card shows ~15.99 GB); so >=15.5 GB lands in the 16GB
tier and >=23.5 GB in the 24GB+ tier.

Your RTX 4080 16GB → `ada-16gb` profile: batch candidates 32,16,8,4, checkpoint modes (False,True), default False, eval cap 16. Autotune will pick 32 usually, maybe 16 if using checkpoint.
Your RTX 4090 24GB → `ada-24gb-plus`: batch 64,32,16,8,4, checkpoint False, eval cap 16, autotune picks 64.

## Custom tuning for Davis

We keep upstream profile logic but pre-document optimal candidates for your box:

### RTX 4080 16GB (Ada, 9728 cores, 16GB GDDR6X, 320W)

- Peak FLOPS used for MFU: `_get_gpu_peak_flops` in train.py returns 242.5e12 (242.5 TFLOPS)
  for "4080" — the dense BF16 tensor-core figure the fork's MFU math is calibrated against
  (the "4080 super" entry is 260e12, matched first by substring order).
- Recommended batch: 32 without checkpoint for MFU ~40%
- If OOM near 16GB, fallback to 16 + checkpoint True
- BF16 amp_dtype (torch.cuda.is_bf16_supported includes emulation false → true on Ada ≥8.0)
- TF32 enabled: torch.backends.cuda.matmul.allow_tf32 = True
- SDPA backend: PyTorch SDPA run in eager mode — torch.compile is disabled in this fork's
  runtime path, so there is no compiled/FA3 fast path; the SDPA kernel dispatch (flash/mem-efficient/math)
  is left to PyTorch at runtime.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` mitigates fragmentation on Windows.

### RTX 4090 24GB (Ada, 16384 cores, 24GB GDDR6X, 450W, peak BF16 ~330 TFLOPS)

- Recommended batch: 64 without checkpoint, eval batch cap 16
- Can handle 64+32+16+8+4 candidates
- Same BF16, TF32, SDPA
- Can get ~500M tokens / 5min vs ~300M on 4080

### Optimizations for this fork

- No torch.compile (disabled in this fork runtime path) to keep stability on Windows consumer GPUs. Original upstream had FA3/fast path on H100 but removed for Windows.
- Autotune: short eager-mode pass with 2 warmup + 3 measure steps, 90% memory fraction, caches per GPU fingerprint to `%LOCALAPPDATA%\autoresearch\gpu-profile-v2.json`. Use `AUTORESEARCH_DISABLE_AUTOTUNE=1` to skip, `AUTORESEARCH_AUTOTUNE_REFRESH=1` to refresh.
- Windows-specific: LOCALAPPDATA cache, not .cache.
- Run artifacts: `checkpoint_pre_eval.pt` is written to `AUTORESEARCH_OUT_DIR` when set, otherwise
  to the working directory (unchanged default). Containerized runs set it to a mounted output
  directory so a measuring run never leaves a checkpoint inside the checked-out repo — see
  `herdmux.train.json`.

## Is the loop input-bound? (read this before tuning anything GPU-side)

Batch size, checkpointing and MFU are all GPU-side knobs. They are worth nothing if the
GPU is idle waiting for the CPU to hand it a batch. Settle that first.

### Why the dataloader is the suspect

`make_dataloader` in `prepare.py` is a single-process Python generator with no worker
processes and no prefetch, and `train.py` calls `next(train_loader)` *inline* inside the
gradient-accumulation loop — so every microsecond it spends sits on the critical path.
Per optimizer step at `device_batch_size=16` that is `TOTAL_BATCH_SIZE // (16 * 2048) = 16`
batches, each packing `B` rows of `MAX_SEQ_LEN + 1 = 2049` tokens. Packing is best-fit:
for **each document placed into a row** it linearly scans `doc_buffer`, which `refill_buffer`
keeps topped up to `buffer_size=1000`. So the Python-level work per row is roughly
`(2049 / mean_doc_tokens) * 1000` iterations, and TinyStories documents are short relative
to a 2049-token row. Refills also run the BPE encoder over 128 documents at a time.

That is a strong prior, not a measurement — hence the instrument below.

### The instrument

`train.py` prints three lines in its final summary:

```
dataloader_percent: <share of step wall-clock building batches in next(train_loader)>
gpu_wait_percent:   <share spent blocked on the GPU>
loop_bound:         input-bound | compute-bound | mixed
```

`gpu_wait_percent` covers the three points where the CPU actually blocks on the device: the
`train_loss.item()` read, the trailing `torch.cuda.synchronize()`, and the dataloader's wait
for its previous host-to-device copy to land. The first two already existed in the loop.
The third is inside `make_dataloader` and is there for correctness, not measurement — see
*The pinned-buffer wait* below — but it is timed and charged to `gpu_wait_percent` rather
than to `dataloader_percent`, because a device wait counted as batch-building time is
exactly how a compute-bound loop would misreport as input-bound.

Timing only the trailing synchronize would have been wrong: `.item()` drains most of the
queue first, so on a compute-bound run the GPU wait would have vanished into unattributed
step time.

### The pinned-buffer wait

`make_dataloader` stages each batch in one pinned CPU buffer and enqueues the copy to the
GPU with `non_blocking=True`. That copy reads the buffer when the stream reaches it, not
when `copy_` returns. `train.py` queues `grad_accum_steps` microbatches between
synchronization points, so the host can run far enough ahead to overwrite the buffer while
the previous copy is still in flight — handing the GPU *this* batch's tokens for the
*previous* batch, with correct shapes and a loss that still falls. The loader now waits on
a CUDA event before reusing the buffer.

The hazard is latent in the input-bound regime (the GPU keeps up, so there is nothing in
flight to clobber) and fires when compute-bound. It never affected `val_bpb`: `evaluate_bpb`
calls `.item()` every iteration, which blocks past each copy before the next one is staged.
Autotune (`_benchmark_train_candidate`) has the same run-ahead as training, but it reports
`tok_per_sec` and peak memory — batch *content* does not change what that costs, so its
numbers were valid either way.

Reading it (`_classify_step_bound` in `train.py`):

- **input-bound** — `dataloader_percent >= 50` and `gpu_wait_percent < 20`. The GPU empties
  the queue faster than the CPU refills it. Fix the input pipeline; ignore GPU knobs.
- **compute-bound** — `gpu_wait_percent >= 50`. GPU-side tuning is now worth doing.
- **mixed** — neither dominates; the thresholds deliberately refuse to guess.

Caveats, so nobody over-reads it:

- **Step 0 is excluded**, so a `--smoke-test` run samples only 2 steps. It pays one-time
  CUDA context init, allocator warmup and first-kernel dispatch — cost that inflates its
  wall time without inflating its dataloader time, which would bias `data_fraction` *down*
  and hand back a false "not input-bound". Note the dataloader's fill-from-empty is *not*
  what step 0 pays: `make_dataloader` is a generator, so its first buffer fill (to
  `buffer_size=1000`, BPE encoding included) runs on the pre-loop `next(train_loader)`,
  before any timing starts. Step 0's `next()` calls pay only steady-state refill.
- **Time inside `next()` overlaps GPU kernels queued earlier**, so the dataloader share
  alone cannot prove starvation — which is exactly why the verdict needs *both* numbers.

## How autoresearch finds best model for your platform

Because time budget fixed 5-min, batch size directly trades tokens vs steps. Larger batch → more tokens per step but fewer steps. Autotune probes candidates and picks max tokens without OOM.

For your 4080/4090, you will see after smoke test:

```
val_bpb: 0.99...
peak_vram_mb: ~12000 for 4080 / ~18000 for 4090 depending batch
mfu_percent: 30-45%
total_tokens_M: 300-500M
num_steps: 500-1000
```

Lower val_bpb is better, vocab-independent.

## Recommendations for offloading

- **Turnover Shield research**: depth 4-6, width small, batch 32, no checkpoint, target params 0.2-0.5M. Fits your 4080 easily, can run 100 exps overnight.
- **Ava research**: depth 6-8, GQA 4, YaRN, WSD, batch 32/64, params 50M. Your 4090 can handle.
- **Write research**: depth 4, small, batch 16, params 0.1M, detector logic inside train.py, fast.

## Comparison to your Ava Docker stack

Your Ava Docker pytorch:2.4.0-cuda12.4-cudnn9 is slightly older than this fork's torch 2.9.1 cu128. For consistency, you can either:

- Use uv native (this fork's recommended) for fast 5-min loops
- Or port wins into Ava Docker for longer runs: copy train.py idea into Ava model_1b.py

Both share same CUDA driver, so VRAM usage comparable.

## Verifying your setup

In PowerShell:

```powershell
nvidia-smi
# should show RTX 4080 or 4090, driver >= 560, CUDA 12.8 etc
uv run python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported())"
```

Expected: `NVIDIA GeForce RTX 4090 (8, 9) True` or similar.

If BF16 false, fallback FP16 still works but slower.

## Notes for future Blackwell

If you upgrade to RTX 5090 32GB Blackwell (12,0): Blackwell has capability (12,0) with the same >=10GB floor, so `_resolve_gpu_profile` yields the `blackwell-24gb-plus` profile — same batch candidates (64,32,16,8,4) and no default checkpointing as ada-24gb-plus. Peak FLOPS 360e12 for "5090" per the lookup table.

## Solo disclaimer

Solo personal project, no connection to employer, built with public/free-tier only. No work data.
