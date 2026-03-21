# ImageNet Projection Experiment Structure

This document proposes a modular structure for reusing the ImageNet train loop
with multiple projection methods.

## Goals

- Keep sampling and TabICL episode logic reusable.
- Make projection method swappable without touching the loop.
- Keep checkpoint formats forward-compatible and backward-compatible.

## Current Modules Added

- `imagenet_projection/projection_methods.py`
  - `build_projection_module(...)`: method registry entry point.
  - Currently supports `projection_head` and exposes metadata for saving/eval.
- `imagenet_projection/checkpoints.py`
  - `save_projection_checkpoint(...)`: canonical save path.
  - `load_projection_head_checkpoint(...)`: unified load for `state_dict`,
    `head_state_dict`, and raw state dict formats.

## Recommended Next Split

1. `imagenet_projection/extraction.py`
   - Move extraction-only concerns from `projection_head_imagenet.py`:
     - `IndexedImageFolder`, `_collate_indexed`
     - `_make_transform`, `_load_dinov3`, `_extract_model_features`
     - `_init_manifest`, `_flush_shard`, `_extract_split`

2. `imagenet_projection/sampling.py`
   - Move shard sampler and static dataset sampling logic:
     - `ShardedEmbeddingSampler`
     - `sample_datasets_loop`, `_sample_one_dataset`

3. `imagenet_projection/episodes.py`
   - Move support/query and label-remapping helpers:
     - `_sample_support_query_indices`
     - `_class_safe_support_query_indices`
     - `_remap_labels_from_support`

4. `imagenet_projection/train_loop.py`
   - Keep generic loop that depends on:
     - a projection module (from `projection_methods.py`)
     - episode utilities (from `episodes.py`)
     - sampler (from `sampling.py`)

5. `imagenet_projection/cli.py`
   - Keep argument parsing and command dispatch.
   - Leave `projection_head_imagenet.py` as a thin entrypoint.

## Hypernetwork Path

To add hypernetwork-generated linear projections later:

- Add a new method case in `build_projection_module(...)`, for example
  `method="hyper_linear"`.
- Return `(module, metadata)` with stable keys:
  - `projection_method`
  - `input_dim`
  - `output_dim`
- Keep training loop unchanged if module still maps `X -> Z`.
- Save with `save_projection_checkpoint(...)` so eval can inspect metadata.

## Checkpoint Compatibility Contract

For all future methods, include at minimum:

- `projection_method`
- `state_dict`
- `history`

For projection-head compatibility with existing scripts, also include:

- `head_state_dict`
- `head_type`, `input_dim`, `output_dim`, `head_hidden_dim`, `head_dropout`
