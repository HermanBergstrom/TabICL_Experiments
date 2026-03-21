# imagenet_projection Handoff README

This package contains the modularized ImageNet projection workflow used by
`projection_head_imagenet.py`.

The goal of this package is to keep extraction, sampling, episode construction,
training, and checkpoint I/O separated so future edits can be localized.

## What Owns What

- `config.py`
  - Dataclasses used across the pipeline:
    - `ExtractConfig`
    - `SamplingConfig`

- `extraction.py`
  - ImageNet embedding extraction and shard persistence.
  - Owns:
    - image dataset wrapper + collate
    - DINOv3 loading
    - model output feature normalization
    - manifest/shard writing
    - split extraction loop
  - Main function for extraction mode:
    - `extract_split(...)`

- `sampling.py`
  - Sampling from saved shard embeddings.
  - Owns:
    - `ShardedEmbeddingSampler`
    - one-shot sampled dataset creation (`sample_one_dataset`)
    - repeated sampling loop (`sample_datasets_loop`)

- `episodes.py`
  - Support/query split logic and label remapping for TabICL episodes.
  - Owns:
    - `sample_support_query_indices`
    - `class_safe_support_query_indices`
    - `remap_labels_from_support`

- `projection_methods.py`
  - Projection module construction.
  - Current supported methods:
    - `projection_head`
    - `zca_projection_head` (episode-level ZCA whitening before adapter)
    - `spectral_hypernetwork` (support-conditioned generated projection)
    - `vanilla_hypernetwork` (support mean/std-conditioned generated projection)
  - Main function:
    - `build_projection_module(...)`

- `train_loop.py`
  - Training orchestration for projection learning on sampled episodes.
  - Includes validation and held-out validation episodes.
  - Main function:
    - `train_projection_head_with_sampler(...)`

- `checkpoints.py`
  - Unified save/load utilities for projection checkpoints.
  - Supports backward compatibility between:
    - `state_dict`
    - `head_state_dict`
    - raw state dict format

- `__init__.py`
  - Re-exports commonly used symbols for package-level imports.

## Entry Script Relationship

- Top-level CLI entrypoint remains:
  - `projection_head_imagenet.py`
- That file should stay thin and delegate implementation to this package.
- If a large feature is added to the pipeline, prefer adding a new module here
  instead of expanding the top-level script.

## Runtime Modes

The top-level script supports three modes:

1. `extract`
   - Uses `extraction.py` to build shard files and manifests.
2. `sample-loop`
   - Uses `sampling.py` to emit sampled datasets from persisted shards.
3. `train-loop`
   - Uses `train_loop.py` for projection training on sampled episodes.

## Data Flow

1. `extract` mode creates:
   - `<output_dir>/<split>_manifest.json`
   - `<output_dir>/<split>_shard_*.pt`
2. `train-loop` and `sample-loop` read those manifests/shards.
3. `train-loop` builds episodes:
   - sampled dataset -> support/query split -> label remap -> TabICL forward
4. Optional checkpoint save writes projection state and metadata.

## Checkpoint Contract

Recommended checkpoint keys:

- `projection_method`
- `state_dict`
- `head_state_dict` (kept for compatibility)
- `history`
- optional metadata:
  - `head_type`, `input_dim`, `output_dim`, `head_hidden_dim`, `head_dropout`
  - `sampling_config`, `train_args`

Loading behavior is centralized in `checkpoints.py`.

## Common Change Scenarios

1. Add a new projection method
   - Update `projection_methods.py`.
   - Add method choice to top-level CLI (`--projection-method`) if needed.
   - Keep train loop unchanged when possible.

## ZCA Baseline Notes

- `zca_projection_head` keeps the learned adapter architecture unchanged.
- Whitening is fit from the support subset of each sampled episode and then
  applied to both support and query before the learned projection head.
- `--zca-epsilon` controls the eigenvalue floor used by whitening for
  numerical stability.

## Spectral Hypernetwork Baseline Notes

- `spectral_hypernetwork` builds an episode-specific projection matrix from
  support-set spectral statistics.
- The context encoder uses top-k singular values and right-singular vectors of
  support features.
- `--hyper-top-k` controls how many spectral components are used.

## Vanilla Hypernetwork Baseline Notes

- `vanilla_hypernetwork` avoids SVD and conditions on support summary stats.
- The context vector is formed from per-feature support mean and standard
  deviation, then mapped to a generated projection matrix and bias.

2. Change extraction details
   - Update only `extraction.py`.
   - Avoid touching training or sampling modules unless output format changes.

3. Change sampling policy
   - Update only `sampling.py` and/or `episodes.py`.

4. Change validation protocol
   - Update `train_loop.py`.

5. Change checkpoint schema
   - Update `checkpoints.py` first.
   - Preserve compatibility paths unless explicitly deprecating old artifacts.

## Practical Commands

From repo root:

- Extract embeddings:
  - `python projection_head_imagenet.py --mode extract --split both`

- Sample datasets for inspection:
  - `python projection_head_imagenet.py --mode sample-loop --split train --n-datasets 5`

- Train projection head:
  - `python projection_head_imagenet.py --mode train-loop --split train --train-output <path>`

## Notes For Future Agents

- Keep `projection_head_imagenet.py` orchestration-only.
- Prefer editing one module per concern.
- Run a syntax check after edits:
  - `python -m py_compile projection_head_imagenet.py imagenet_projection/*.py`
- If changing checkpoint fields, verify both training save path and eval load path
  in `projection_head_eval.py` still work.
- You can use 'source /project/aip-rahulgk/hermanb/environments/aditya_tabicl/bin/activate' to activate the appropriate working environment.
