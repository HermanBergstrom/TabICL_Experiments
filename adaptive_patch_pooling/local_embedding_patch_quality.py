"""Refactored patch-quality evaluation using IterativePALPooler.

This copy keeps the same CLI and result structure as
local_embedding_patch_quality.py, but delegates the iterative refinement path
to IterativePALPooler so the script is much smaller and easier to compare
against the original.
"""

from __future__ import annotations

import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.decomposition import PCA
from tabicl import TabICLClassifier

from adaptive_patch_pooling import local_embedding_patch_quality_old as base
from adaptive_patch_pooling.pal_pooler import IterativePALPooler


BUTTERFLY_DATASET_PATH = base.BUTTERFLY_DATASET_PATH
RSNA_DATASET_PATH = base.RSNA_DATASET_PATH
DATASET_PATH = base.DATASET_PATH
FEATURES_DIR = base.FEATURES_DIR


def run_patch_quality_eval(
	features_dir: Path = FEATURES_DIR,
	dataset_path: Path = DATASET_PATH,
	dataset: str = "butterfly",
	backbone: str = "rad-dino",
	n_sample: int = 8,
	n_train: Optional[int] = None,
	n_estimators: int = 1,
	pca_dim: Optional[int] = 128,
	seed: int = 42,
	output_dir: Path = Path("patch_quality_results"),
	patch_size: int = 16,
	patch_group_sizes: list = [1],
	refine: bool = False,
	temperature: float = 1.0,
	batch_size: int = 10,
	weight_method: str = "correct_class_prob",
	ridge_alpha: float = 1.0,
	normalize_features: bool = False,
	max_query_rows: Optional[int] = None,
	use_random_subsampling: bool = False,
	balance_train: bool = False,
	balance_test: bool = False,
	attn_pool: bool = False,
	attn_pool_only: bool = False,
	attn_steps: int = 500,
	attn_lr: float = 1e-3,
	attn_max_step_samples: int = 512,
	attn_num_queries: int = 1,
	attn_num_heads: int = 8,
	device: str = "auto",
	post_refinement_viz: bool = False,
	aoe_class: Optional[str] = None,
	aoe_handling: str = "filter",
	gpu_ridge: bool = False,
	_cli_args: Optional[dict] = None,
) -> None:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	experiment_start = time.perf_counter()
	run_ts = datetime.now(timezone.utc).isoformat()

	(
		train_patches,
		train_labels,
		test_patches,
		test_labels,
		cls_train_feats,
		cls_test_feats,
		idx_to_class,
		train_sub_idx,
	) = base._load_features(
		dataset=dataset,
		features_dir=features_dir,
		n_train=n_train,
		seed=seed,
		backbone=backbone,
	)

	bal_rng = np.random.RandomState(seed + 1)
	bal_train_keep_idx: Optional[np.ndarray] = None
	if balance_train:
		train_patches, train_labels, cls_train_feats, bal_train_keep_idx = base._balance_classes(
			train_patches, train_labels, cls_train_feats, bal_rng
		)
	if balance_test:
		test_patches, test_labels, cls_test_feats, _ = base._balance_classes(
			test_patches, test_labels, cls_test_feats, bal_rng
		)

	_, _, D = train_patches.shape
	n_classes = int(train_labels.max()) + 1
	counts = np.bincount(train_labels.astype(np.int64), minlength=n_classes)
	class_prior = (counts / counts.sum()).astype(np.float32)

	aoe_mask: Optional[np.ndarray] = None
	resolved_aoe_class_idx: Optional[int] = None
	if aoe_class is not None:
		class_to_idx = {v: k for k, v in idx_to_class.items()}
		try:
			aoe_class_idx = int(aoe_class)
		except (ValueError, TypeError):
			aoe_class_idx = None
		if aoe_class_idx is not None:
			if aoe_class_idx not in idx_to_class:
				raise ValueError(f"aoe_class index {aoe_class_idx} not in [0, {n_classes - 1}]")
		else:
			name = str(aoe_class)
			if name not in class_to_idx:
				raise ValueError(
					f"aoe_class '{name}' not found in class names. Available: {sorted(class_to_idx)}"
				)
			aoe_class_idx = class_to_idx[name]
		resolved_aoe_class_idx = aoe_class_idx
		aoe_mask = train_labels == aoe_class_idx
		print(
			f"[aoe] Absence-of-evidence class: '{idx_to_class[aoe_class_idx]}' (index {aoe_class_idx}), "
			f"{int(aoe_mask.sum())} training images excluded from Ridge fitting"
		)

	if attn_pool_only:
		base._run_attn_only(
			train_patches=train_patches,
			train_labels=train_labels,
			test_patches=test_patches,
			test_labels=test_labels,
			D=D,
			output_dir=output_dir,
			attn_steps=attn_steps,
			attn_lr=attn_lr,
			attn_max_step_samples=attn_max_step_samples,
			attn_num_queries=attn_num_queries,
			attn_num_heads=attn_num_heads,
			device=device,
			seed=seed,
			pca_dim=pca_dim,
			n_estimators=n_estimators,
			cli_args=_cli_args,
		)
		base._merge_attn_into_results(output_dir)
		return

	n_stages = len(patch_group_sizes)

	def _broadcast(val, label: str) -> list:
		if isinstance(val, (int, float)):
			return [float(val)] * n_stages
		vals = list(val)
		if len(vals) == 1:
			return vals * n_stages
		if len(vals) != n_stages:
			raise ValueError(
				f"{label}: {len(vals)} value(s) given for {n_stages} stage(s) in --patch-group-sizes; "
				f"pass a single value (broadcast to all stages) or exactly {n_stages} value(s)."
			)
		return vals

	temperatures = _broadcast(temperature, "--temperature")
	ridge_alphas = _broadcast(ridge_alpha, "--ridge-alpha")

	baseline_support_raw = train_patches.astype(np.float32).mean(axis=1)
	pca: Optional[PCA] = None
	if pca_dim is not None:
		n_comp = min(pca_dim, len(train_labels), D)
		pca = PCA(n_components=n_comp, random_state=seed)
		baseline_support = pca.fit_transform(baseline_support_raw).astype(np.float32)
		print(f"[info] PCA: {D}D → {n_comp}D")
	else:
		baseline_support = baseline_support_raw

	cls_acc: Optional[float] = None
	cls_auroc: Optional[float] = None
	if cls_train_feats is not None and cls_test_feats is not None:
		cls_pca: Optional[PCA] = None
		if pca_dim is not None:
			n_comp_cls = min(pca_dim, len(cls_train_feats), cls_train_feats.shape[1])
			cls_pca = PCA(n_components=n_comp_cls, random_state=seed)
			cls_support = cls_pca.fit_transform(cls_train_feats).astype(np.float32)
			cls_test_q = cls_pca.transform(cls_test_feats).astype(np.float32)
		else:
			cls_support = cls_train_feats
			cls_test_q = cls_test_feats
		cls_acc, cls_auroc = base._compute_accuracy_from_features(
			cls_support,
			train_labels,
			cls_test_q,
			test_labels,
			n_estimators=n_estimators,
			seed=seed,
		)

	train_image_paths: list = []
	test_image_paths: list = []
	open_image = None
	if n_sample > 0:
		if dataset == "butterfly":
			train_image_paths, _, idx_to_class = base._get_image_paths(dataset_path, split="train", seed=seed)
			test_image_paths, _, _ = base._get_image_paths(dataset_path, split="test", seed=seed)
		elif dataset == "rsna":
			train_image_paths, _, _ = base._get_rsna_image_paths(dataset_path, features_dir, split="train", backbone=backbone)
			test_image_paths, _, _ = base._get_rsna_image_paths(dataset_path, features_dir, split="test", backbone=backbone)
			open_image = base._dicom_to_pil

		if train_sub_idx is not None:
			train_image_paths = [train_image_paths[i] for i in train_sub_idx]
		if bal_train_keep_idx is not None:
			train_image_paths = [train_image_paths[i] for i in bal_train_keep_idx]

	rng = np.random.RandomState(seed)
	train_sample_idx = rng.choice(len(train_labels), size=min(n_sample, len(train_labels)), replace=False)
	test_sample_idx = rng.choice(len(test_labels), size=min(n_sample, len(test_labels)), replace=False)

	baseline_acc, baseline_auroc = base._compute_accuracy(
		baseline_support,
		train_labels,
		test_patches,
		test_labels,
		pca=pca,
		n_estimators=n_estimators,
		seed=seed,
	)
	if cls_acc is not None:
		print(f"\n[cls-token]  test accuracy: {cls_acc:.4f}  auroc: {cls_auroc:.4f}")
	else:
		print("\n[cls-token]  test accuracy: N/A (files not found)")
	print(f"[mean-pool]  test accuracy: {baseline_acc:.4f}  auroc: {baseline_auroc:.4f}")

	attn_result: Optional[dict] = None
	if attn_pool:
		import torch as _torch
		from attention_pooling_experiments import _pool_with_head as _attn_pool_fn
		from attention_pooling_experiments import train_attention_pooling_head
		from finetune_projection_head import ProjectionTrainingConfig

		if device == "auto":
			_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
		else:
			_device = _torch.device(device)

		attn_cfg = ProjectionTrainingConfig(
			num_steps=attn_steps,
			learning_rate=attn_lr,
			max_step_samples=attn_max_step_samples,
			seed=seed,
			log_every=max(1, attn_steps // 10),
		)
		print(
			f"\n[attn-pool]  Training attention head  (steps={attn_steps}  lr={attn_lr}  "
			f"device={_device}  n_queries={attn_num_queries}  n_heads={attn_num_heads})"
		)

		t_attn_start = time.perf_counter()
		attn_head, attn_history = train_attention_pooling_head(
			train_patches=_torch.from_numpy(train_patches),
			y_train=train_labels,
			val_patches=_torch.from_numpy(test_patches),
			y_val=test_labels,
			embed_dim=D,
			out_dim=None,
			num_queries=attn_num_queries,
			num_heads=attn_num_heads,
			device=_device,
			config=attn_cfg,
		)
		attn_total_time_s = time.perf_counter() - t_attn_start

		attn_best_val_acc_raw = max(attn_history["val_accuracy"]) if attn_history["val_accuracy"] else float("nan")
		attn_time_to_best = attn_history.get("time_to_best_s", float("nan"))
		attn_best_step = attn_history.get("best_val_step", 0)

		print(f"[attn-pool]  Evaluating best checkpoint (step {attn_best_step}) with PCA={pca_dim} ...")
		attn_train_pooled = _attn_pool_fn(attn_head, _torch.from_numpy(train_patches), _device)
		attn_test_pooled = _attn_pool_fn(attn_head, _torch.from_numpy(test_patches), _device)
		if pca_dim is not None:
			n_comp_attn = min(pca_dim, len(train_labels), attn_train_pooled.shape[1])
			attn_pca = PCA(n_components=n_comp_attn, random_state=seed)
			attn_train_pooled = attn_pca.fit_transform(attn_train_pooled).astype(np.float32)
			attn_test_pooled = attn_pca.transform(attn_test_pooled).astype(np.float32)
		attn_test_acc, attn_test_auroc = base._compute_accuracy_from_features(
			attn_train_pooled,
			train_labels,
			attn_test_pooled,
			test_labels,
			n_estimators=n_estimators,
			seed=seed,
		)

		attn_result = {
			"test_acc": round(attn_test_acc, 6),
			"test_auroc": round(attn_test_auroc, 6) if not np.isnan(attn_test_auroc) else None,
			"best_val_acc_raw": round(attn_best_val_acc_raw, 6),
			"best_val_step": attn_best_step,
			"time_to_best_s": attn_time_to_best,
			"total_train_time_s": round(attn_total_time_s, 2),
		}
		print(
			f"[attn-pool]  test acc (PCA={pca_dim}): {attn_test_acc:.4f}  auroc: {attn_test_auroc:.4f}  "
			f"(best train val: {attn_best_val_acc_raw:.4f}  step {attn_best_step}/{attn_steps}  "
			f"time_to_best={attn_time_to_best:.1f}s)"
		)

	if n_sample > 0 and not post_refinement_viz:
		split_configs_orig = [
			("train", train_patches, train_labels, train_image_paths, train_sample_idx),
			("test", test_patches, test_labels, test_image_paths, test_sample_idx),
		]
		baseline_mean_probs = base._run_visual_eval(
			"baseline",
			baseline_support,
			train_labels,
			split_configs_orig,
			idx_to_class,
			pca=pca,
			n_estimators=n_estimators,
			patch_size=patch_size,
			seed=seed,
			output_dir=output_dir,
			temperature=temperatures[0],
			ridge_model=None,
			feature_scaler=None,
			open_image=open_image,
			class_prior=class_prior,
			weight_method=weight_method,
		)
	else:
		baseline_mean_probs = {}

	if not refine:
		base._save_results(
			output_dir=output_dir,
			run_ts=run_ts,
			cli_args=_cli_args,
			total_time_s=time.perf_counter() - experiment_start,
			train_patches=train_patches,
			test_labels=test_labels,
			D=D,
			n_classes=n_classes,
			pca=pca,
			cls_acc=cls_acc,
			cls_auroc=cls_auroc,
			baseline_acc=baseline_acc,
			baseline_auroc=baseline_auroc,
			all_results=[("baseline", baseline_acc, baseline_auroc, baseline_mean_probs, 0.0, 0.0, 0.0, 0.0)],
			attn_result=attn_result,
		)
		return

	pooler = IterativePALPooler(
		tabicl=TabICLClassifier(n_estimators=n_estimators, random_state=seed),
		patch_group_sizes=patch_group_sizes,
		weight_method=weight_method,
		temperature=temperatures,
		ridge_alpha=ridge_alphas,
		max_query_rows=max_query_rows,
		use_random_subsampling=use_random_subsampling,
		pca_dim=pca_dim,
		batch_size=batch_size,
		seed=seed,
		aoe_class=resolved_aoe_class_idx,
		aoe_handling=aoe_handling,
		gpu_ridge=gpu_ridge,
		normalize_features=normalize_features,
	)

	pooler.fit(train_patches, train_labels)

	current_support = baseline_support
	current_pca = pca
	all_results: list[tuple[str, float, float, dict, float, float, float, float]] = [
		("baseline", baseline_acc, baseline_auroc, baseline_mean_probs, 0.0, 0.0, 0.0, 0.0)
	]

	stages = pooler.stages_
	pre_refine_support = current_support
	pre_refine_pca = current_pca
	tag = "baseline"
	stage_temp = temperatures[0]
	eff_patch_sz = patch_size
	train_grouped = train_patches
	test_grouped = test_patches

	for stage_idx, (group_size, stage) in enumerate(zip(patch_group_sizes, stages)):
		stage_temp = temperatures[stage_idx]
		stage_alpha = ridge_alphas[stage_idx]
		group_side = int(round(group_size ** 0.5))
		eff_patch_sz = patch_size * group_side
		tag = f"iter_{stage_idx}_g{group_size}"

		print(
			f"\n[{tag}] group_size={group_size}  ({group_side}×{group_side} patches per group)  "
			f"T={stage_temp}  ridge_alpha={stage_alpha}"
		)

		train_grouped = base.group_patches(train_patches, group_size)
		test_grouped = base.group_patches(test_patches, group_size)

		if n_sample > 0 and not post_refinement_viz:
			split_configs_iter = [
				("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
				("test", test_grouped, test_labels, test_image_paths, test_sample_idx),
			]
			iter_mean_probs = base._run_visual_eval(
				tag,
				current_support,
				train_labels,
				split_configs_iter,
				idx_to_class,
				pca=current_pca,
				n_estimators=n_estimators,
				patch_size=eff_patch_sz,
				seed=seed,
				output_dir=output_dir,
				temperature=stage_temp,
				ridge_model=None,
				feature_scaler=None,
				open_image=open_image,
				class_prior=class_prior,
				weight_method=weight_method,
			)
		else:
			iter_mean_probs = {}

		pre_refine_support = current_support
		pre_refine_pca = current_pca

		if stage.ridge_model_ is not None:
			ridge_path = output_dir / f"ridge_quality_model_{tag}.joblib"
			joblib.dump(stage.ridge_model_, ridge_path)
			print(f"[ridge] Model saved → {ridge_path}")

		if n_sample > 0 and post_refinement_viz and stage.ridge_model_ is not None:
			split_configs_post = [
				("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
				("test", test_grouped, test_labels, test_image_paths, test_sample_idx),
			]
			iter_mean_probs = base._run_visual_eval(
				f"{tag}_post",
				current_support,
				train_labels,
				split_configs_post,
				idx_to_class,
				pca=current_pca,
				n_estimators=n_estimators,
				patch_size=eff_patch_sz,
				seed=seed,
				output_dir=output_dir,
				temperature=stage_temp,
				ridge_model=stage.ridge_model_,
				feature_scaler=stage.feature_scaler_,
				open_image=open_image,
				class_prior=class_prior,
				weight_method=weight_method,
			)

		w_ridge = base._ridge_pool_weights(test_grouped, stage.ridge_model_, stage.feature_scaler_)
		test_repooled = (w_ridge[:, :, None] * test_grouped).sum(axis=1)
		test_query = stage._pca_.transform(test_repooled).astype(np.float32) if stage._pca_ is not None else test_repooled.astype(np.float32)

		t_eval_start = time.perf_counter()
		iter_acc, iter_auroc = base._compute_accuracy_from_features(
			stage._support_projected_,
			train_labels,
			test_query,
			test_labels,
			n_estimators=n_estimators,
			seed=seed,
		)
		eval_time_s = time.perf_counter() - t_eval_start
		fit_time_s = stage.fit_time_s_
		pool_time_s = stage.pool_time_s_
		refine_time_s = fit_time_s + pool_time_s
		print(
			f"[{tag}] test accuracy (quality-pooled queries): {iter_acc:.4f}  auroc: {iter_auroc:.4f}  "
			f"(fit {fit_time_s:.1f}s, pool {pool_time_s:.1f}s, eval {eval_time_s:.1f}s)"
		)

		all_results.append((tag, iter_acc, iter_auroc, iter_mean_probs, refine_time_s, eval_time_s, fit_time_s, pool_time_s))
		current_support = stage._support_projected_
		current_pca = stage._pca_

	if n_sample > 0 and not post_refinement_viz and refine and stages[-1].ridge_model_ is not None:
		split_configs_final = [
			("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
			("test", test_grouped, test_labels, test_image_paths, test_sample_idx),
		]
		base._run_visual_eval(
			f"{tag}_post",
			pre_refine_support,
			train_labels,
			split_configs_final,
			idx_to_class,
			pca=pre_refine_pca,
			n_estimators=n_estimators,
			patch_size=eff_patch_sz,
			seed=seed,
			output_dir=output_dir,
			temperature=stage_temp,
			ridge_model=stages[-1].ridge_model_,
			feature_scaler=stages[-1].feature_scaler_,
			open_image=open_image,
			class_prior=class_prior,
			weight_method=weight_method,
		)

	total_time_s = time.perf_counter() - experiment_start

	col_w = max(len(r[0]) for r in all_results) + 2
	print("\n" + "=" * (col_w + 78))
	print("ITERATIVE REFINEMENT SUMMARY")
	print("=" * (col_w + 70))
	print(
		f"  {'Stage':<{col_w}}  {'Test Acc':>10}  {'AUROC':>8}  {'Δ Acc':>8}  "
		f"{'P(true)/train':>14}  {'P(true)/test':>13}  {'Fit(s)':>8}  {'Pool(s)':>8}  {'Eval(s)':>8}"
	)
	print("-" * (col_w + 78))
	for stage_name, acc, auroc, mean_probs, refine_s, eval_s, fit_s, pool_s in all_results:
		delta_str = "" if stage_name == "baseline" else f"{acc - baseline_acc:+.4f}"
		fit_str = "-" if stage_name == "baseline" else f"{fit_s:.1f}"
		pool_str = "-" if stage_name == "baseline" else f"{pool_s:.1f}"
		eval_str = "-" if stage_name == "baseline" else f"{eval_s:.1f}"
		auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "  N/A"
		print(
			f"  {stage_name:<{col_w}}  {acc:>10.4f}  {auroc_str:>8}  {delta_str:>8}"
			f"  {mean_probs.get('train', float('nan')):>14.3f}"
			f"  {mean_probs.get('test', float('nan')):>13.3f}"
			f"  {fit_str:>8}  {pool_str:>8}  {eval_str:>8}"
		)
	print("=" * (col_w + 78))
	print(f"  Total wall time: {total_time_s:.1f}s")

	base._save_results(
		output_dir=output_dir,
		run_ts=run_ts,
		cli_args=_cli_args,
		total_time_s=total_time_s,
		train_patches=train_patches,
		test_labels=test_labels,
		D=D,
		n_classes=n_classes,
		pca=pca,
		cls_acc=cls_acc,
		cls_auroc=cls_auroc,
		baseline_acc=baseline_acc,
		baseline_auroc=baseline_auroc,
		all_results=all_results,
		attn_result=attn_result,
	)


def run_n_train_sweep(
	n_train_values: list[int],
	base_output_dir: Path,
	**kwargs,
) -> None:
	base_output_dir = Path(base_output_dir)
	base_output_dir.mkdir(parents=True, exist_ok=True)

	sweep_start = time.perf_counter()
	sweep_ts = datetime.now(timezone.utc).isoformat()
	sweep_runs: list[dict] = []

	for n_train in n_train_values:
		run_dir = base_output_dir / f"n_train_{n_train}"
		print(f"\n{'='*60}")
		print(f"  SWEEP  n_train={n_train}  →  {run_dir}")
		print(f"{'='*60}")

		run_cli_args = dict(kwargs.get("_cli_args") or {})
		run_cli_args["n_train"] = n_train
		run_cli_args["output_dir"] = str(run_dir)

		run_patch_quality_eval(
			n_train=n_train,
			output_dir=run_dir,
			**{k: v for k, v in kwargs.items() if k != "_cli_args"},
			_cli_args=run_cli_args,
		)

		results_path = run_dir / "results.json"
		attn_path = run_dir / "attn_pool_results.json"
		run_summary: dict = {"n_train": n_train, "output_dir": str(run_dir)}
		if results_path.exists():
			with results_path.open() as f:
				run_data = json.load(f)
			run_summary["baselines"] = run_data.get("baselines", {})
			run_summary["stages"] = run_data.get("stages", [])
			run_summary["total_time_s"] = run_data.get("total_time_s")
		elif attn_path.exists():
			with attn_path.open() as f:
				attn_data = json.load(f)
			run_summary["baselines"] = {"attn_pool": attn_data.get("attn_pool")}
		sweep_runs.append(run_summary)

	sweep_total = time.perf_counter() - sweep_start
	sweep_record = {
		"sweep_timestamp": sweep_ts,
		"n_train_values": n_train_values,
		"total_sweep_time_s": round(sweep_total, 2),
		"runs": sweep_runs,
	}
	sweep_path = base_output_dir / "sweep_results.json"
	with sweep_path.open("w") as f:
		json.dump(sweep_record, f, indent=2)
	print(f"\n[sweep] Done — {len(n_train_values)} runs in {sweep_total:.1f}s")
	print(f"[sweep] Results → {sweep_path}")


def _parse_args() -> object:
	return base._parse_args()


if __name__ == "__main__":
	sys.stdout.reconfigure(line_buffering=True)
	args = _parse_args()

	if args.n_train_sweep is not None and args.n_train is not None:
		raise SystemExit("error: --n-train-sweep and --n-train are mutually exclusive")

	_dataset_defaults = {"butterfly": BUTTERFLY_DATASET_PATH, "rsna": RSNA_DATASET_PATH}
	dataset_path = args.dataset_path or _dataset_defaults[args.dataset]

	shared_kwargs = dict(
		dataset=args.dataset,
		backbone=args.backbone,
		features_dir=args.features_dir,
		dataset_path=dataset_path,
		n_sample=args.n_sample,
		n_estimators=args.n_estimators,
		pca_dim=None if args.no_pca else args.pca_dim,
		seed=args.seed,
		patch_size=args.patch_size,
		patch_group_sizes=args.patch_group_sizes,
		refine=args.refine,
		temperature=args.temperature,
		batch_size=args.batch_size,
		weight_method=args.weight_method,
		ridge_alpha=args.ridge_alpha,
		normalize_features=args.normalize_features,
		max_query_rows=args.max_query_rows,
		use_random_subsampling=args.use_random_subsampling,
		balance_train=args.balance_train,
		balance_test=args.balance_test,
		attn_pool=args.attn_pool or args.attn_pool_only,
		attn_pool_only=args.attn_pool_only,
		attn_steps=args.attn_steps,
		attn_lr=args.attn_lr,
		attn_max_step_samples=args.attn_max_step_samples,
		attn_num_queries=args.attn_num_queries,
		attn_num_heads=args.attn_num_heads,
		device=args.device,
		post_refinement_viz=args.post_refinement_viz,
		aoe_class=args.aoe_class,
		aoe_handling=args.aoe_handling,
		gpu_ridge=args.gpu_ridge,
	)

	if args.n_train_sweep is not None:
		run_n_train_sweep(
			n_train_values=args.n_train_sweep,
			base_output_dir=args.output_dir,
			_cli_args=vars(args),
			**shared_kwargs,
		)
	else:
		run_patch_quality_eval(
			n_train=args.n_train,
			output_dir=args.output_dir,
			_cli_args=vars(args),
			**shared_kwargs,
		)
