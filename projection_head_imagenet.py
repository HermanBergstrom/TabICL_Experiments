#!/usr/bin/env python
"""CLI entrypoint for ImageNet extraction/sampling/projection training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from finetune_projection_head import FrozenTabICLConfig
from imagenet_projection.save_checkpoints import save_projection_checkpoint
from imagenet_projection.config import ExtractConfig, SamplingConfig
from imagenet_projection.extraction import extract_split, load_dinov3, resolve_device
from imagenet_projection.sampling import sample_datasets_loop
from imagenet_projection.train_loop import train_projection_head_with_sampler


def _enable_live_output() -> None:
    """Best-effort line-buffering so logs appear promptly in batch jobs."""
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass


def _sanitize_run_name(raw_name: str | None) -> str:
    """Normalize run name into a filesystem-safe slug."""
    if raw_name is None:
        return "run"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._-")
    return cleaned or "run"


def parse_args() -> tuple[argparse.Namespace, ExtractConfig]:
    parser = argparse.ArgumentParser(
        description="Extract full-ImageNet DINOv3 embeddings and train projection methods.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["extract", "sample-loop", "train-loop"],
        default="extract",
        help="Run feature extraction, class-first dataset sampling loop, or training loop.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/datasets/imagenet"),
        help="ImageNet root directory containing train/ and val/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_features/imagenet_dinov3"),
        help="Directory where shard files and manifests are written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="Which ImageNet split to process.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=50_000, help="Number of samples per .pt shard.")
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("../dinov3"),
        help="Path to local DINOv3 repository for torch.hub.load(..., source='local').",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        help="Path to DINOv3 checkpoint weights.",
    )
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--amp-dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--resume", action="store_true", help="Resume from existing manifests/shards in output_dir.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory for faster host->GPU transfer.")
    parser.add_argument("--n-datasets", type=int, default=10)
    parser.add_argument("--min-dataset-size", type=int, default=512)
    parser.add_argument("--max-dataset-size", type=int, default=4096)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--max-classes", type=int, default=200)
    parser.add_argument("--min-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-replacement", action="store_true", help="Allow sampling rows with replacement within class pools.")
    parser.add_argument("--no-remap-labels", action="store_true", help="Keep original ImageNet class ids in sampled y instead of remapping to [0, K).")
    parser.add_argument("--max-cached-shards", type=int, default=8)
    parser.add_argument(
        "--sampler-backend",
        type=str,
        choices=["shard", "preload"],
        default="preload",
        help=(
            "Sampling backend. 'preload' loads all split features once into RAM for "
            "maximum throughput; 'shard' keeps lower memory usage but higher per-step I/O."
        ),
    )
    parser.add_argument(
        "--sampling-distribution",
        type=str,
        choices=["uniform", "log-uniform"],
        default="log-uniform",
        help=(
            "Dataset size sampling strategy. 'log-uniform' samples in log-space to favor "
            "smaller datasets (aligns with scaling laws); 'uniform' samples linearly."
        ),
    )
    parser.add_argument(
        "--enable-hard-sampling",
        action="store_true",
        help=(
            "Enable hard negative class sampling using cosine similarity of class centroids. "
            "This selects more challenging classification tasks by favoring similar classes."
        ),
    )
    parser.add_argument(
        "--hard-sampling-temperature",
        type=float,
        default=0.1,
        help="Temperature for hard sampling softmax. Lower = stricter nearest neighbor selection.",
    )
    parser.add_argument(
        "--hard-sampling-prob",
        type=float,
        default=0.5,
        help="Probability of using hard sampling (vs. uniform). Default 0.5 = 50/50 split.",
    )
    parser.add_argument(
        "--hard-sampling-device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for hard sampling similarity matrix computation.",
    )
    parser.add_argument("--sample-output-dir", type=Path, default=None, help="Optional directory to persist sampled datasets as .npz files.")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--ortho-lambda",
        type=float,
        default=0.01,
        help="Scale for orthogonal regularization on hypernetwork-generated projection matrix W.",
    )
    parser.add_argument("--head-type", type=str, choices=["linear", "mlp"], default="linear")
    parser.add_argument(
        "--projection-method",
        type=str,
        choices=["projection_head", "zca_projection_head", "spectral_hypernetwork", "vanilla_hypernetwork"],
        default="projection_head",
    )
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument(
        "--zca-epsilon",
        type=float,
        default=1e-5,
        help="Numerical floor used by ZCA whitening when --projection-method zca_projection_head.",
    )
    parser.add_argument(
        "--hyper-top-k",
        type=int,
        default=16,
        help="Top-k singular components used by spectral_hypernetwork context encoder.",
    )
    parser.add_argument(
        "--hyper-encoder-type",
        type=str,
        choices=["mlp", "attention"],
        default="mlp",
        help="Context encoder used by spectral_hypernetwork.",
    )
    parser.add_argument(
        "--hyper-attn-heads",
        type=int,
        default=4,
        help="Number of attention heads when --hyper-encoder-type attention.",
    )
    parser.add_argument(
        "--hyper-attn-layers",
        type=int,
        default=2,
        help="Number of transformer encoder layers when --hyper-encoder-type attention.",
    )
    parser.add_argument(
        "--disable-random-projection-init",
        action="store_true",
        help=(
            "Use standard PyTorch initialization for hypernetwork decoders instead of "
            "random-projection initialization."
        ),
    )
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--query-fraction-min", type=float, default=0.1)
    parser.add_argument("--query-fraction-max", type=float, default=0.4)
    parser.add_argument("--min-query-size", type=int, default=8)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Accumulate gradients over this many sampled episodes before each optimizer step. "
            "Useful for larger effective batch sizes with limited GPU memory."
        ),
    )
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--val-split", type=str, choices=["none", "train", "val"], default="val")
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument(
        "--val-heldout-classes",
        type=int,
        default=None,
        help=(
            "Number of ImageNet classes to hold out for generalization validation. "
            "If set, these classes are excluded from training and used only for held-out validation."
        ),
    )
    parser.add_argument("--tabicl-train-shuffles", type=int, default=4)
    parser.add_argument("--tabicl-shuffle", type=str, choices=["none", "random", "shift"], default="random")
    parser.add_argument("--tabicl-softmax-temperature", type=float, default=0.9)
    parser.add_argument("--tabicl-checkpoint-version", type=str, default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--tabicl-model-path", type=Path, default=None)
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="imagenet-projection")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("model_weights/hypernetwork_checkpoints"),
        help=(
            "Base directory for training checkpoints. "
            "A per-run subdirectory <wandb-run-name>_<timestamp> is created automatically. "
            "When --resume-training is used, this should point to a specific existing run directory."
        ),
    )
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume training loop state from checkpoint-dir/latest.pt if present.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=500,
        help="Save latest training checkpoint every N optimization steps.",
    )
    parser.add_argument("--train-output", type=Path, default=None, help="Optional path to save trained head checkpoint (.pt).")

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be > 0")
    if args.img_size <= 0 or args.resize_size <= 0:
        raise ValueError("--img-size and --resize-size must be > 0")
    if args.n_datasets <= 0:
        raise ValueError("--n-datasets must be > 0")
    if args.min_dataset_size <= 0 or args.max_dataset_size <= 0:
        raise ValueError("--min-dataset-size and --max-dataset-size must be > 0")
    if args.min_dataset_size > args.max_dataset_size:
        raise ValueError("--min-dataset-size cannot exceed --max-dataset-size")
    if args.min_classes <= 1:
        raise ValueError("--min-classes must be > 1")
    if args.min_classes > args.max_classes:
        raise ValueError("--min-classes cannot exceed --max-classes")
    if args.min_per_class <= 0:
        raise ValueError("--min-per-class must be > 0")
    if args.max_cached_shards <= 0:
        raise ValueError("--max-cached-shards must be > 0")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    if args.ortho_lambda < 0:
        raise ValueError("--ortho-lambda must be >= 0")
    if args.projection_dim <= 0:
        raise ValueError("--projection-dim must be > 0")
    if args.zca_epsilon <= 0:
        raise ValueError("--zca-epsilon must be > 0")
    if args.hyper_top_k <= 0:
        raise ValueError("--hyper-top-k must be > 0")
    if args.hyper_attn_heads <= 0:
        raise ValueError("--hyper-attn-heads must be > 0")
    if args.hyper_attn_layers <= 0:
        raise ValueError("--hyper-attn-layers must be > 0")
    if args.head_hidden_dim <= 0:
        raise ValueError("--head-hidden-dim must be > 0")
    if not (0.0 < args.query_fraction_min < 1.0):
        raise ValueError("--query-fraction-min must be in (0, 1)")
    if not (0.0 < args.query_fraction_max < 1.0):
        raise ValueError("--query-fraction-max must be in (0, 1)")
    if args.query_fraction_min > args.query_fraction_max:
        raise ValueError("--query-fraction-min cannot exceed --query-fraction-max")
    if args.min_query_size <= 0:
        raise ValueError("--min-query-size must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if args.val_every < 0:
        raise ValueError("--val-every must be >= 0")
    if args.val_batches <= 0:
        raise ValueError("--val-batches must be > 0")
    if args.tabicl_train_shuffles <= 0:
        raise ValueError("--tabicl-train-shuffles must be > 0")
    if args.checkpoint_interval_steps <= 0:
        raise ValueError("--checkpoint-interval-steps must be > 0")
    if args.enable_hard_sampling:
        if not (0.0 < args.hard_sampling_temperature <= 1.0):
            raise ValueError("--hard-sampling-temperature must be in (0, 1]")
        if not (0.0 < args.hard_sampling_prob <= 1.0):
            raise ValueError("--hard-sampling-prob must be in (0, 1]")

    splits = ["train", "val"] if args.split == "both" else [args.split]
    return args, ExtractConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        splits=splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        resize_size=args.resize_size,
        shard_size=args.shard_size,
        repo_dir=args.repo_dir,
        weights_path=args.weights,
        device=args.device,
        dtype=args.dtype,
        amp_dtype=args.amp_dtype,
        resume=bool(args.resume),
        pin_memory=bool(args.pin_memory),
    )


def main() -> None:
    _enable_live_output()
    args, config = parse_args()
    execution_started_at = datetime.now()
    compact_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    print(f"[args] {json.dumps(compact_args, sort_keys=True, separators=(',', ':'))}")

    if args.mode == "extract":
        config.output_dir.mkdir(parents=True, exist_ok=True)

        device = resolve_device(config.device)
        print(f"[info] Device: {device}")
        if device.type != "cuda" and config.amp_dtype != "float16":
            print("[info] Running on CPU; AMP settings are ignored.")

        model = load_dinov3(config.repo_dir, config.weights_path, device)
        for split in config.splits:
            extract_split(config=config, split=split, model=model, device=device)
        print("[done] ImageNet embedding extraction complete.")
        return

    if args.split == "both":
        raise ValueError("--split must be train or val when --mode sample-loop or --mode train-loop")

    sampling_cfg = SamplingConfig(
        output_dir=config.output_dir,
        split=args.split,
        n_datasets=args.n_datasets,
        min_dataset_size=args.min_dataset_size,
        max_dataset_size=args.max_dataset_size,
        min_classes=args.min_classes,
        max_classes=args.max_classes,
        min_per_class=args.min_per_class,
        seed=args.seed,
        remap_labels=not bool(args.no_remap_labels),
        allow_replacement=bool(args.allow_replacement),
        max_cached_shards=args.max_cached_shards,
        sampler_backend=args.sampler_backend,
        sampling_distribution=args.sampling_distribution,
        enable_hard_sampling=bool(args.enable_hard_sampling),
        hard_sampling_temperature=args.hard_sampling_temperature,
        hard_sampling_prob=args.hard_sampling_prob,
        hard_sampling_device=args.hard_sampling_device,
    )

    if args.mode == "sample-loop":
        sample_output_dir = args.sample_output_dir
        if sample_output_dir is not None:
            sample_output_dir.mkdir(parents=True, exist_ok=True)

        for i, X, y, meta in sample_datasets_loop(sampling_cfg):
            print(
                f"[sample {i:04d}] size={meta['dataset_size']} classes={meta['n_classes']} "
                f"X={tuple(X.shape)} y={tuple(y.shape)}"
            )
            if sample_output_dir is not None:
                np.savez_compressed(
                    sample_output_dir / f"sampled_dataset_{i:04d}.npz",
                    X=X,
                    y=y,
                    meta=json.dumps(meta),
                )

        print("[done] Sample loop complete.")
        return

    checkpoint_dir = args.checkpoint_dir
    if args.mode == "train-loop":
        if args.resume_training:
            # Resume must target an explicit run directory to avoid ambiguity.
            latest_checkpoint = checkpoint_dir / "latest.pt"
            if not latest_checkpoint.exists():
                raise ValueError(
                    "--resume-training requires --checkpoint-dir to point to an existing "
                    "run directory containing latest.pt"
                )
        else:
            run_name_slug = _sanitize_run_name(args.wandb_run_name)
            start_stamp = execution_started_at.strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = checkpoint_dir / f"{run_name_slug}_{start_stamp}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] Checkpoint directory: {checkpoint_dir}")

    tabicl_cfg = FrozenTabICLConfig(
        n_models=1,
        n_feature_shuffles=args.tabicl_train_shuffles,
        feature_shuffle_method=args.tabicl_shuffle,
        shuffle_seed=args.seed,
        checkpoint_version=args.tabicl_checkpoint_version,
        model_path=args.tabicl_model_path,
        softmax_temperature=args.tabicl_softmax_temperature,
    )

    device = resolve_device(args.device)
    print(f"[info] Training device: {device}")
    if args.val_split == "none" or args.val_every == 0:
        val_sampling_cfg = None
    else:
        val_sampling_cfg = SamplingConfig(
            output_dir=config.output_dir,
            split=args.val_split,
            n_datasets=max(1, args.val_batches),
            min_dataset_size=args.min_dataset_size,
            max_dataset_size=args.max_dataset_size,
            min_classes=args.min_classes,
            max_classes=args.max_classes,
            min_per_class=args.min_per_class,
            seed=args.seed + 101,
            remap_labels=not bool(args.no_remap_labels),
            allow_replacement=bool(args.allow_replacement),
            max_cached_shards=args.max_cached_shards,
            sampler_backend=args.sampler_backend,
            sampling_distribution=args.sampling_distribution,
            enable_hard_sampling=bool(args.enable_hard_sampling),
            hard_sampling_temperature=args.hard_sampling_temperature,
            hard_sampling_prob=args.hard_sampling_prob,
            hard_sampling_device=args.hard_sampling_device,
        )
        print(
            f"[info] Validation enabled: split={args.val_split} "
            f"every={args.val_every} steps, batches={args.val_batches}"
        )

    head, history, projection_meta = train_projection_head_with_sampler(
        sampling_config=sampling_cfg,
        val_sampling_config=val_sampling_cfg,
        val_heldout_classes_count=args.val_heldout_classes,
        device=device,
        num_steps=args.train_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        query_fraction_min=args.query_fraction_min,
        query_fraction_max=args.query_fraction_max,
        min_query_size=args.min_query_size,
        head_type=args.head_type,
        projection_dim=args.projection_dim,
        hidden_dim=args.head_hidden_dim,
        dropout=args.head_dropout,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        val_every=args.val_every,
        val_batches=args.val_batches,
        tabicl_config=tabicl_cfg,
        projection_method=args.projection_method,
        zca_epsilon=args.zca_epsilon,
        hyper_top_k=args.hyper_top_k,
        hyper_encoder_type=args.hyper_encoder_type,
        hyper_attn_heads=args.hyper_attn_heads,
        hyper_attn_layers=args.hyper_attn_layers,
        use_random_projection_init=not bool(args.disable_random_projection_init),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ortho_lambda=args.ortho_lambda,
        enable_wandb=not bool(args.disable_wandb),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_dir=checkpoint_dir,
        resume_training=bool(args.resume_training),
        checkpoint_interval_steps=args.checkpoint_interval_steps,
    )

    if args.train_output is not None:
        args.train_output.parent.mkdir(parents=True, exist_ok=True)
        save_projection_checkpoint(
            path=args.train_output,
            projection_module=head,
            history=history,
            extra={
                "sampling_config": sampling_cfg.__dict__,
                "train_args": vars(args),
                "projection_method": args.projection_method,
                "head_type": projection_meta.get("head_type"),
                "input_dim": projection_meta.get("input_dim"),
                "output_dim": projection_meta.get("output_dim"),
                "head_hidden_dim": projection_meta.get("head_hidden_dim"),
                "head_dropout": projection_meta.get("head_dropout"),
                "zca_epsilon": projection_meta.get("zca_epsilon"),
                "hyper_top_k": projection_meta.get("hyper_top_k"),
                "hyper_encoder_type": projection_meta.get("hyper_encoder_type"),
                "hyper_attn_heads": projection_meta.get("hyper_attn_heads"),
                "hyper_attn_layers": projection_meta.get("hyper_attn_layers"),
                "hyper_use_random_projection_init": projection_meta.get("hyper_use_random_projection_init"),
            },
        )
        print(f"[done] Saved trained head checkpoint to: {args.train_output}")

    print("[done] Train loop complete.")


if __name__ == "__main__":
    main()
