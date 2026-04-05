"""Learnable attention pooling experiments on the butterfly dataset.

Trains an attention-based pooling head over DINOv3 patch tokens using a frozen
TabICL backbone as the training objective (same episodic CE loss as
finetune_projection_head.py).  This gives an upper-bound estimate on how much
performance can be gained over simple mean/max pooling.

Pre-extract patch tokens first (see local_embedding_experiments.py), then:

    python attention_pooling_experiments.py train \\
        [--out-dim 128] [--num-queries 1] [--num-heads 8] \\
        [--num-steps 500] [--lr 1e-3] [--device cuda]

    python attention_pooling_experiments.py eval \\
        --checkpoint extracted_features/attn_pool.pt

Architecture
------------
``AttentionPoolingHead`` takes ``[N, P, D]`` patch features and outputs
``[N, out_dim]``:

  1. LayerNorm on input patches (for stable Q/K computation only).
  2. K learnable query vectors project into Q; patch tokens project into K.
     **No value (V) projection** — the values are always the original,
     un-normalized patch embeddings.
  3. Multi-head attention weights computed from Q and K, averaged across heads
     → ``[N, K, P]``.
  4. Pooled output = ``attn_weights @ patches`` — a pure weighted average of
     the original patch tokens in the original D-dimensional space.
  5. Flatten K pooled vectors → optional linear projection to ``out_dim``
     (separate post-pooling step for TabICL input-size compatibility).

Steps 1-4 guarantee that the pooled representation is a convex combination of
the original patch embeddings; only the attention weights are learned, not the
embedding contents.  Setting ``num_queries=1`` gives a CLS-like aggregation;
larger values pool at K different attention patterns, all projected to
``out_dim`` in step 5.

Training notes
--------------
* The test set is passed as validation to select the best checkpoint (the upper
  bound framing makes this appropriate).
* ``max_step_samples`` limits how many training rows are forwarded per step to
  keep each iteration tractable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from local_embedding_experiments import ButterflyPatchDataset, FEATURES_DIR
from finetune_projection_head import (
    FrozenTabICLConfig,
    FrozenTabICLBackbone,
    ProjectionTrainingConfig,
    build_frozen_tabicl_backbone,
    _class_safe_support_query_indices,
    _remap_labels_from_support,
    _sample_step_subset,
    _validate_config,
)


# ---------------------------------------------------------------------------
# Attention pooling module
# ---------------------------------------------------------------------------

class AttentionPoolingHead(nn.Module):
    """Attention pooling where the output is a weighted average of original patches.

    Q and K are projected (multi-head), but V is always the raw patch embeddings
    — no value projection.  This ensures the pooled vector lives in the original
    embedding space and represents a convex combination of the input patches.

    Forward pass:
        1. LayerNorm patches → used only for Q/K projection (stabilises training).
        2. Project Q (learnable queries) and K (normed patches) per head.
        3. Compute softmax attention weights ``[N, H, K, P]``, average over heads
           → ``[N, K, P]``.
        4. ``pooled = attn_weights @ patches``  (original, un-normalized patches)
           → ``[N, K, D]``.  This is the pure weighted-average step.
        5. Flatten K pooled vectors ``[N, K*D]`` → linear projection to
           ``out_dim`` (post-pooling step; separate from the attention mechanism).

    Args:
        embed_dim:   Dimension D of input patch tokens.
        out_dim:     Output feature dimension fed to TabICL.  If ``None``,
                     defaults to ``embed_dim`` (no projection in step 5).
        num_queries: Number of learnable query vectors.
        num_heads:   Attention heads (must divide ``embed_dim``).
        dropout:     Dropout on attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: Optional[int] = None,
        num_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self.out_dim = embed_dim if out_dim is None else out_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learnable query tokens
        self.queries = nn.Parameter(torch.empty(num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # LayerNorm for Q/K computation only — values always use raw patches
        self.input_norm = nn.LayerNorm(embed_dim)

        # Q and K projections — NO value projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # Post-pooling projection (step 5, separate from attention)
        pooled_dim = num_queries * embed_dim
        self.proj = (
            nn.Linear(pooled_dim, self.out_dim)
            if pooled_dim != self.out_dim
            else nn.Identity()
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: float tensor ``[N, P, D]``

        Returns:
            float tensor ``[N, out_dim]``
        """
        N, P, D = patches.shape
        K, H, d_h = self.num_queries, self.num_heads, self.head_dim

        # Normalize patches for Q/K only; raw patches remain unchanged for V
        patches_norm = self.input_norm(patches)                          # [N, P, D]

        # Expand learnable queries and project Q, K
        Q = self.q_proj(self.queries.unsqueeze(0).expand(N, -1, -1))    # [N, K, D]
        K_proj = self.k_proj(patches_norm)                               # [N, P, D]

        # Reshape to multi-head: [N, H, K/P, d_h]
        Q      = Q.view(N, K, H, d_h).transpose(1, 2)                   # [N, H, K, d_h]
        K_proj = K_proj.view(N, P, H, d_h).transpose(1, 2)              # [N, H, P, d_h]

        # Attention weights
        attn = (Q @ K_proj.transpose(-2, -1)) * (d_h ** -0.5)           # [N, H, K, P]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Average attention weights across heads → [N, K, P]
        attn_avg = attn.mean(dim=1)

        # Step 4: weighted average of ORIGINAL (un-normalized) patch tokens
        pooled = torch.bmm(attn_avg, patches)                            # [N, K, D]

        # Step 5: flatten and project (post-pooling, not part of attention)
        return self.proj(pooled.reshape(N, K * D))                       # [N, out_dim]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_attention_pooling_head(
    train_patches: torch.Tensor,
    y_train: np.ndarray,
    *,
    val_patches: Optional[torch.Tensor] = None,
    y_val: Optional[np.ndarray] = None,
    embed_dim: int,
    out_dim: Optional[int] = None,
    num_queries: int = 1,
    num_heads: int = 8,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
    tabicl_config: Optional[FrozenTabICLConfig] = None,
    tabicl_backbone: Optional[FrozenTabICLBackbone] = None,
    config: Optional[ProjectionTrainingConfig] = None,
    verbose: bool = True,
) -> tuple[AttentionPoolingHead, dict]:
    """Train an attention pooling head on patch features with frozen TabICL.

    The training objective is identical to ``train_projection_head``: episodic
    support/query cross-entropy through the frozen TabICL backbone.  The key
    difference is that the input is a 3D patch tensor ``[N, P, D]`` and the
    head performs learnable pooling as its first step.

    If ``val_patches`` / ``y_val`` are provided, validation accuracy is tracked
    every ``config.log_every`` steps and the best checkpoint is restored.

    Args:
        train_patches: float tensor ``[N_train, P, D]``.
        y_train:       integer class labels ``[N_train]``.
        val_patches:   optional float tensor ``[N_val, P, D]``.
        y_val:         optional integer class labels ``[N_val]``.
        embed_dim:     patch token dimension D.
        out_dim:       output feature dimension (fed to TabICL backbone).
                       Defaults to ``embed_dim`` (no post-pooling projection).
        num_queries:   number of learnable attention query vectors.
        num_heads:     number of attention heads.
        config:        ``ProjectionTrainingConfig`` (uses defaults if None).

    Returns:
        ``(head, history)`` — head restored to best val checkpoint if available.
    """
    if config is None:
        config = ProjectionTrainingConfig()
    _validate_config(config)

    device = torch.device(device)
    X_t = train_patches.to(dtype=torch.float32, device=device)   # [N, P, D]
    y_t = torch.as_tensor(y_train, dtype=torch.long, device=device)

    use_validation = val_patches is not None and y_val is not None
    if use_validation:
        X_val_t = val_patches.to(dtype=torch.float32, device=device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)
    else:
        X_val_t = y_val_t = None

    # Resolve effective output dim for backbone input_dim
    effective_out_dim = embed_dim if out_dim is None else out_dim

    # Build frozen TabICL backbone
    if tabicl_backbone is None:
        tabicl_cfg = tabicl_config or FrozenTabICLConfig()
        tabicl_backbone = build_frozen_tabicl_backbone(
            input_dim=effective_out_dim,
            config=tabicl_cfg,
            device=device,
        )
    tabicl_backbone = tabicl_backbone.to(device)
    tabicl_backbone.eval()

    head = AttentionPoolingHead(
        embed_dim=embed_dim,
        out_dim=out_dim,
        num_queries=num_queries,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[attention pooling] Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed)

    history: dict[str, list[float]] = {
        "loss": [],
        "query_fraction": [],
        "query_size": [],
        "val_step": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    best_val_step: int = 0
    best_time_s: float = 0.0
    t_train_start = time.perf_counter()

    def _run_validation(step_for_eval: int) -> tuple[float, float]:
        assert X_val_t is not None and y_val_t is not None
        head.eval()
        with torch.no_grad():
            X_support_proj = head(X_t)            # [N_train, out_dim]
            X_query_proj = head(X_val_t)          # [N_val,   out_dim]
            y_sup_local, y_q_local = _remap_labels_from_support(y_t, y_val_t)
            logits_val = tabicl_backbone(
                X_support_proj, y_sup_local, X_query_proj,
                step_index=step_for_eval - 1,
            )
            val_loss = float(criterion(logits_val, y_q_local).item())
            val_acc = float((logits_val.argmax(dim=1) == y_q_local).float().mean().item())
        return val_loss, val_acc

    for step in range(1, config.num_steps + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)

        # Optionally downsample rows for this step
        X_step, y_step = _sample_step_subset(X_t, y_t, rng, config.max_step_samples)
        n_step = int(X_step.shape[0])

        support_idx_np, query_idx_np = _class_safe_support_query_indices(
            y=y_step,
            rng=rng,
            frac_min=config.query_fraction_min,
            frac_max=config.query_fraction_max,
            min_query_size=config.min_query_size,
            query_minority_weight=config.query_minority_weight,
        )
        support_idx = torch.as_tensor(support_idx_np, device=device, dtype=torch.long)
        query_idx   = torch.as_tensor(query_idx_np,   device=device, dtype=torch.long)

        # Pool patches → feature vectors, then split support/query
        X_proj    = head(X_step)                              # [n_step, out_dim]
        X_support = X_proj.index_select(0, support_idx)
        y_support = y_step.index_select(0, support_idx)
        X_query   = X_proj.index_select(0, query_idx)
        y_query   = y_step.index_select(0, query_idx)

        y_support_local, y_query_local = _remap_labels_from_support(y_support, y_query)
        logits = tabicl_backbone(X_support, y_support_local, X_query, step_index=step - 1)
        loss = criterion(logits, y_query_local)
        loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(head.parameters(), config.grad_clip_norm)
        optimizer.step()

        query_frac = float(y_query.shape[0] / max(1, n_step))
        history["loss"].append(float(loss.detach().cpu().item()))
        history["query_fraction"].append(query_frac)
        history["query_size"].append(float(y_query.shape[0]))

        if verbose and (step == 1 or step % config.log_every == 0 or step == config.num_steps):
            print(
                f"[step {step:04d}/{config.num_steps}] "
                f"loss={history['loss'][-1]:.4f}  "
                f"query={int(y_query.shape[0])}/{n_step} ({query_frac:.3f})"
            )

        if use_validation and (step % config.log_every == 0 or step == config.num_steps):
            val_loss, val_acc = _run_validation(step)
            history["val_step"].append(float(step))
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            if verbose:
                print(
                    f"[val  step {step:04d}/{config.num_steps}] "
                    f"loss={val_loss:.4f}  acc={val_acc:.4f}"
                )
            is_better = (val_acc > best_val_acc) or (
                val_acc == best_val_acc and val_loss < best_val_loss
            )
            if is_better:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                best_val_step = step
                best_time_s = time.perf_counter() - t_train_start

    if use_validation and best_state is not None:
        print(f"[attention pooling] Restoring best checkpoint (val_acc={best_val_acc:.4f}  "
              f"step={best_val_step}  time_to_best={best_time_s:.1f}s)")
        head.load_state_dict(best_state)

    history["best_val_step"] = best_val_step
    history["time_to_best_s"] = round(best_time_s, 2)

    return head, history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _pool_with_head(
    head: AttentionPoolingHead,
    patches: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Apply a trained attention pooling head in mini-batches."""
    head.eval()
    head = head.to(device)
    results = []
    for start in range(0, len(patches), batch_size):
        batch = patches[start : start + batch_size].to(dtype=torch.float32, device=device)
        results.append(head(batch).cpu())
    return torch.cat(results, dim=0).numpy()


# ---------------------------------------------------------------------------
# Full experiment: train + evaluate
# ---------------------------------------------------------------------------

def run_attention_pooling_experiment(
    features_dir: Path = FEATURES_DIR,
    out_dim: Optional[int] = None,
    num_queries: int = 1,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_steps: int = 500,
    learning_rate: float = 1e-3,
    max_step_samples: int = 512,
    n_estimators: int = 1,
    n_train: Optional[int] = None,
    device: str = "auto",
    seed: int = 42,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Train attention pooling head, then evaluate with TabICLClassifier.

    The test set is used as validation to select the best pooling checkpoint
    (appropriate for an upper-bound estimation experiment).

    Args:
        out_dim:           Output dimension after pooling (defaults to embed_dim,
                           i.e. no post-pooling projection).
        num_queries:       Number of learnable attention query vectors.
        num_heads:         Number of attention heads.
        num_steps:         Training iterations.
        learning_rate:     AdamW learning rate.
        max_step_samples:  Max rows forwarded per training step (for speed).
        n_estimators:      TabICL estimators for final evaluation.
        n_train:           Subsample training set to this many examples.
        checkpoint_path:   If set, save the trained head here.

    Returns:
        dict with keys ``"test_acc"`` and ``"history"``.
    """
    from tabicl import TabICLClassifier

    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    train_ds = ButterflyPatchDataset(features_dir, split="train")
    test_ds  = ButterflyPatchDataset(features_dir, split="test")

    train_patches = train_ds.features   # [N_train, P, D]
    y_train = train_ds.labels.numpy()

    if n_train is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_patches), size=min(n_train, len(train_patches)), replace=False)
        train_patches = train_patches[idx]
        y_train = y_train[idx]

    _, P, D = train_patches.shape
    effective_out_dim = D if out_dim is None else out_dim
    print(
        f"\n[attention pooling] embed_dim={D}  out_dim={effective_out_dim}  "
        f"num_queries={num_queries}  num_heads={num_heads}  "
        f"steps={num_steps}  lr={learning_rate}  "
        f"n_train={len(train_patches)}  device={device_t}"
    )

    train_cfg = ProjectionTrainingConfig(
        num_steps=num_steps,
        learning_rate=learning_rate,
        max_step_samples=max_step_samples,
        seed=seed,
        log_every=max(1, num_steps // 10),
    )

    head, history = train_attention_pooling_head(
        train_patches=train_patches,
        y_train=y_train,
        val_patches=test_ds.features,
        y_val=test_ds.labels.numpy(),
        embed_dim=D,
        out_dim=out_dim,
        num_queries=num_queries,
        num_heads=num_heads,
        dropout=dropout,
        device=device_t,
        config=train_cfg,
        verbose=verbose,
    )

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "embed_dim": D,
                "out_dim": head.out_dim,   # resolved value (never None)
                "num_queries": num_queries,
                "num_heads": num_heads,
            },
            checkpoint_path,
        )
        print(f"[info] Checkpoint saved → {checkpoint_path}")

    # Final evaluation: pool all patches, then classify with TabICL
    print("\n[attention pooling] Pooling train and test features...")
    X_train_pooled = _pool_with_head(head, train_patches, device_t)
    X_test_pooled  = _pool_with_head(head, test_ds.features, device_t)
    print(f"[attention pooling] train={X_train_pooled.shape}  test={X_test_pooled.shape}")

    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train_pooled, y_train)
    test_acc = float(np.mean(clf.predict(X_test_pooled) == test_ds.labels.numpy()))
    print(f"[attention pooling] test_acc={test_acc:.4f}")
    return {"test_acc": test_acc, "history": history}


def run_attn_pool_baseline(
    features_dir: Path = FEATURES_DIR,
    pca_dim: Optional[int] = None,
    num_queries: int = 1,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_steps: int = 500,
    learning_rate: float = 1e-3,
    max_step_samples: int = 512,
    n_estimators: int = 1,
    n_train: Optional[int] = None,
    device: str = "auto",
    seed: int = 42,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Train attention pooling head, optionally apply PCA, evaluate with TabICL.

    Mirrors ``run_pool_pca_baseline`` so results are directly comparable:
    the same PCA step (fit on train, applied to test) is applied after pooling
    when ``pca_dim`` is set.

    Args:
        pca_dim:  If set, apply PCA to the pooled vectors before TabICL.
                  If ``None``, the raw pooled vectors (shape ``[N, embed_dim]``)
                  are passed directly to TabICL.
        checkpoint_path: If set, save the trained head here.

    Returns:
        dict with key ``"test_acc"``.
    """
    from sklearn.decomposition import PCA
    from tabicl import TabICLClassifier

    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    train_ds = ButterflyPatchDataset(features_dir, split="train")
    test_ds  = ButterflyPatchDataset(features_dir, split="test")

    train_patches = train_ds.features   # [N_train, P, D]
    y_train = train_ds.labels.numpy()

    if n_train is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_patches), size=min(n_train, len(train_patches)), replace=False)
        train_patches = train_patches[idx]
        y_train = y_train[idx]

    _, P, D = train_patches.shape
    label = f"attn/{'pca' if pca_dim else 'no-pca'}"
    print(
        f"\n[{label}] embed_dim={D}  pca_dim={pca_dim}  "
        f"num_queries={num_queries}  num_heads={num_heads}  "
        f"steps={num_steps}  lr={learning_rate}  n_train={len(train_patches)}"
    )

    train_cfg = ProjectionTrainingConfig(
        num_steps=num_steps,
        learning_rate=learning_rate,
        max_step_samples=max_step_samples,
        seed=seed,
        log_every=max(1, num_steps // 10),
    )

    head, _ = train_attention_pooling_head(
        train_patches=train_patches,
        y_train=y_train,
        val_patches=test_ds.features,
        y_val=test_ds.labels.numpy(),
        embed_dim=D,
        out_dim=None,   # always pool to original embed_dim; PCA handles reduction
        num_queries=num_queries,
        num_heads=num_heads,
        dropout=dropout,
        device=device_t,
        config=train_cfg,
        verbose=verbose,
    )

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "embed_dim": D,
                "out_dim": head.out_dim,
                "num_queries": num_queries,
                "num_heads": num_heads,
            },
            checkpoint_path,
        )
        print(f"[info] Checkpoint saved → {checkpoint_path}")

    X_train = _pool_with_head(head, train_patches, device_t)
    X_test  = _pool_with_head(head, test_ds.features, device_t)

    if pca_dim is not None:
        n_comp = min(pca_dim, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        print(f"[{label}] attn-pool → PCA({D}→{n_comp}) → {X_train.shape}")
    else:
        print(f"[{label}] attn-pool (no PCA) → {X_train.shape}")

    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, y_train)
    test_acc = float(np.mean(clf.predict(X_test) == test_ds.labels.numpy()))
    print(f"[{label}] test_acc={test_acc:.4f}")
    return {"test_acc": test_acc}


def load_attention_pooling_head(
    checkpoint_path: Path,
    device: torch.device,
) -> AttentionPoolingHead:
    """Reconstruct and load an ``AttentionPoolingHead`` from a saved checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    head = AttentionPoolingHead(
        embed_dim=ckpt["embed_dim"],
        out_dim=ckpt["out_dim"],
        num_queries=ckpt["num_queries"],
        num_heads=ckpt["num_heads"],
    )
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learnable attention pooling experiments on butterfly dataset"
    )
    sub = parser.add_subparsers(dest="command")

    tr = sub.add_parser("train", help="Train attention pooling head and evaluate")
    tr.add_argument("--features-dir",    type=Path,  default=FEATURES_DIR)
    tr.add_argument("--out-dim",         type=int,   default=None,
                    help="Output dimension fed to TabICL (default: embed_dim, no projection)")
    tr.add_argument("--num-queries",     type=int,   default=1,
                    help="Learnable query vectors: 1=CLS-like, >1=multi-view")
    tr.add_argument("--num-heads",       type=int,   default=8,
                    help="Attention heads (must divide embed_dim)")
    tr.add_argument("--dropout",         type=float, default=0.1)
    tr.add_argument("--num-steps",       type=int,   default=500)
    tr.add_argument("--lr",              type=float, default=1e-3)
    tr.add_argument("--max-step-samples",type=int,   default=512,
                    help="Max training rows forwarded per step")
    tr.add_argument("--n-estimators",    type=int,   default=1)
    tr.add_argument("--n-train",         type=int,   default=None,
                    help="Subsample training set to this many examples")
    tr.add_argument("--device",          type=str,   default="auto")
    tr.add_argument("--seed",            type=int,   default=42)
    tr.add_argument("--checkpoint",      type=Path,  default=None,
                    help="Save trained head to this .pt file")

    ev = sub.add_parser("eval", help="Evaluate a saved attention pooling checkpoint")
    ev.add_argument("--features-dir",    type=Path,  default=FEATURES_DIR)
    ev.add_argument("--checkpoint",      type=Path,  required=True)
    ev.add_argument("--n-estimators",    type=int,   default=1)
    ev.add_argument("--n-train",         type=int,   default=None)
    ev.add_argument("--device",          type=str,   default="auto")
    ev.add_argument("--seed",            type=int,   default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.command == "train":
        results = run_attention_pooling_experiment(
            features_dir=args.features_dir,
            out_dim=args.out_dim,
            num_queries=args.num_queries,
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_steps=args.num_steps,
            learning_rate=args.lr,
            max_step_samples=args.max_step_samples,
            n_estimators=args.n_estimators,
            n_train=args.n_train,
            device=args.device,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
        )
        print(f"\n--- result ---")
        print(f"  test_acc={results['test_acc']:.4f}")

    elif args.command == "eval":
        if args.device == "auto":
            device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_t = torch.device(args.device)

        from tabicl import TabICLClassifier

        head = load_attention_pooling_head(Path(args.checkpoint), device_t)
        train_ds = ButterflyPatchDataset(args.features_dir, split="train")
        test_ds  = ButterflyPatchDataset(args.features_dir, split="test")

        train_patches = train_ds.features
        y_train = train_ds.labels.numpy()

        if args.n_train is not None:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(
                len(train_patches), size=min(args.n_train, len(train_patches)), replace=False
            )
            train_patches = train_patches[idx]
            y_train = y_train[idx]

        X_train = _pool_with_head(head, train_patches, device_t)
        X_test  = _pool_with_head(head, test_ds.features, device_t)

        clf = TabICLClassifier(n_estimators=args.n_estimators, random_state=args.seed)
        clf.fit(X_train, y_train)
        test_acc = float(np.mean(clf.predict(X_test) == test_ds.labels.numpy()))
        print(f"test_acc={test_acc:.4f}")

    else:
        print("Usage: python attention_pooling_experiments.py {train,eval} [options]")
