# Multi-GPU DDP Migration Plan for Projection-Head Training

This document describes a staged migration from single-process training to multi-GPU DistributedDataParallel (DDP) for the projection-head training loop.

Scope:
- Target training path: `projection_head_imagenet.py` -> `imagenet_projection/train_loop.py`
- Preserve current behavior as much as possible.
- Use gradient accumulation as the first optimization lever.
- Do not include functional changes to model architecture.

## Current Constraints (Observed)

- Training currently runs in a single process and single `device` selection path.
- One sampled episode is processed per loop iteration.
- Gradient accumulation is already implemented in `train_loop.py`.
- Sampler RNGs are local and currently not rank-aware.
- Logging/checkpoint/validation are all process-local and ungated.

## Core Design Decisions

1. DDP backend: NCCL on CUDA nodes.
2. Process model: one process per GPU via `torchrun`.
3. Parallelized module: projection `head` only (trainable parameters).
4. Frozen TabICL backbone: keep replicated per rank (no DDP wrap required).
5. Gradient synchronization: suppress on non-step micro-batches with `no_sync`.
6. Side effects (W&B, checkpoint writes, console-heavy logging): rank 0 only.
7. Validation strategy (phase 1): rank 0 only.

## Effective Batch Semantics

With DDP and accumulation, optimizer updates correspond to:

`effective_batch ~ world_size * gradient_accumulation_steps * (episode micro-batch)`

Because episode/query sizes vary per step, exact sample-equivalence is approximate unless losses are weighted by global query counts.

## Commit-Sized Execution Plan

## Commit 1: Distributed Launch Plumbing (No Training Semantics Change)

Files:
- `projection_head_imagenet.py`

Tasks:
- Add CLI/config options for distributed execution (or derive from env):
  - `--distributed`
  - `--dist-backend` (default `nccl`)
  - `--dist-url` (default `env://`)
  - `--rank`, `--world-size`, `--local-rank` (env-backed)
- Initialize process group when distributed is enabled.
- Set per-process CUDA device from local rank.
- Expose `is_distributed`, `rank`, `world_size`, `local_rank` to train loop call.
- Add clean shutdown via `dist.destroy_process_group()`.

Verification:
- `torchrun --nproc_per_node=2 ... --mode train-loop` starts 2 workers without hangs.
- Each rank reports a distinct local GPU index.

## Commit 2: DDP Wrap + Accumulation-Aware Gradient Sync

Files:
- `imagenet_projection/train_loop.py`

Tasks:
- Extend function args with distributed context (`is_distributed`, `rank`, `world_size`, `local_rank`).
- Build projection head as before, then wrap in DDP only when distributed.
- Use a helper `head_for_state = head.module if wrapped else head` for save/load/state dict.
- In accumulation loop:
  - For micro-batches where optimizer step is not due, run backward under `head.no_sync()`.
  - On final micro-batch of accumulation window, run normal backward to trigger all-reduce.
- Keep optimizer/scheduler stepping cadence identical to current logic.

Verification:
- Single GPU run produces near-identical metrics to pre-DDP.
- 2-GPU run performs one all-reduce per optimizer step, not per micro-batch.

## Commit 3: Rank-Safe Logging and Checkpointing

Files:
- `imagenet_projection/train_loop.py`

Tasks:
- Gate `wandb.init` and `wandb.log` to rank 0 only.
- Gate periodic and final `torch.save` to rank 0 only.
- Ensure rank 0 writes latest/best checkpoints and prints authoritative summaries.
- Add barriers around critical save points if needed to avoid race/shutdown hazards.

Verification:
- Only one W&B run is created.
- Checkpoint files are written once and are not corrupted.

## Commit 4: Sampler Rank Awareness (Avoid Duplicate Episode Work)

Files:
- `imagenet_projection/sampling.py`
- `imagenet_projection/train_loop.py`

Tasks:
- Introduce deterministic per-rank RNG offseting (minimum viable):
  - sampler seed = base_seed + rank * offset
  - split RNG / heldout RNG similarly rank-offset
- Optionally add a stronger strategy for globally partitioned episode streams if strict non-overlap is required.
- Document reproducibility expectations for distributed vs single-process runs.

Verification:
- Across ranks, sampled metadata (`class_ids`, `dataset_size`) diverges as expected.
- Throughput scales better than naive duplicated-sample behavior.

## Commit 5: Metric Aggregation Policy

Files:
- `imagenet_projection/train_loop.py`

Tasks:
- Decide and implement global metric reductions:
  - minimum: reduce train loss/accuracy-like metrics for rank-0 reporting.
  - optional: weighted reductions using query counts.
- Keep local diagnostics (timings, hard sampling usage) either local or reduced intentionally.

Verification:
- Rank-0 logs reflect global, not rank-local, estimates for selected metrics.

## Commit 6: Validation Path Hardening

Files:
- `imagenet_projection/train_loop.py`

Tasks:
- Phase 1: execute validation and held-out validation only on rank 0.
- Ensure other ranks skip validation sections and synchronize where required.
- Keep baseline PCA/RP eval logic untouched, just rank-gated.

Verification:
- No deadlocks at validation boundaries.
- Validation cadence and checkpoint decisions remain stable.

## Commit 7: Resume Semantics in Distributed Mode

Files:
- `imagenet_projection/train_loop.py`

Tasks:
- Rank 0 loads checkpoint metadata/state and broadcasts required state to other ranks, or all ranks load safely from same path.
- Ensure model/optimizer/scheduler states are identical after resume.
- Revisit RNG restoration policy:
  - per-rank RNG should remain rank-distinct post-resume.
- Keep W&B resume behavior rank-0 only.

Verification:
- Resume from mid-run checkpoint continues without divergence or duplicated side effects.

## Commit 8: Launch Scripts + Operator UX

Files:
- `bash_scripts/` (new or updated launch scripts)
- Optional docs (`DVM_EXPERIMENTS.md` or dedicated README)

Tasks:
- Add canonical `torchrun` launch examples.
- Document key knobs:
  - `gradient_accumulation_steps`
  - world size
  - LR policy under scaling
  - validation-on-rank-0 behavior
- Provide troubleshooting notes for NCCL init, rank mismatch, and hangs.

Verification:
- One copy/paste launch command for 1-GPU and N-GPU runs.

## Optional Enhancements After Baseline DDP

- Mixed precision (`torch.cuda.amp`) with scaler.
- Activation checkpointing for memory pressure.
- More robust distributed validation reductions.
- Async checkpointing or lower checkpoint frequency under high rank count.

## Risk Register

1. Duplicate episode sampling across ranks:
- Impact: poor scaling.
- Mitigation: rank-aware seeds / partitioned sampler.

2. Over-synchronization with accumulation:
- Impact: communication overhead.
- Mitigation: `no_sync` for non-step micro-batches.

3. Side-effect races (W&B/checkpoints):
- Impact: corrupted outputs, duplicate runs.
- Mitigation: rank-0 ownership.

4. Metric drift due to variable query-size episodes:
- Impact: misleading comparisons vs single-GPU.
- Mitigation: weighted global reductions.

## Minimal Acceptance Criteria

- 2-GPU run completes end-to-end with no deadlocks.
- Checkpoint resume works in distributed mode.
- Exactly one W&B run and one set of checkpoints are produced.
- Throughput improves over 1-GPU baseline under comparable settings.
- Training quality remains within expected variance.

## Suggested Rollout Sequence

1. Commit 1 + 2 on a short smoke run.
2. Add commit 3 (safe side-effects).
3. Add commit 4 (sampling quality/scaling fix).
4. Add commit 5/6/7 to harden correctness.
5. Add commit 8 for team usability.
