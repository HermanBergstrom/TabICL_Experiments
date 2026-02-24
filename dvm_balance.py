from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DATA_DIR = Path("/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset")
DEFAULT_DVM_MODULE_PATH = Path(
	"/home/hermanb/projects/aip-rahulgk/image_icl_project/DVM_Dataset/dvm_dataset_with_dinov2.py"
)
DEFAULT_OUTPUT = Path(
	"/home/hermanb/projects/aip-rahulgk/hermanb/TabICL_Experiments/plots/dvm_class_balance_hist.png"
)


def _import_load_dvm_dataset(dvm_module_path: Path):
	if not dvm_module_path.exists():
		raise FileNotFoundError(f"DVM module not found: {dvm_module_path}")

	spec = importlib.util.spec_from_file_location("dvm_dataset", str(dvm_module_path))
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load module spec from {dvm_module_path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	if not hasattr(module, "load_dvm_dataset"):
		raise AttributeError(f"Module {dvm_module_path} does not define load_dvm_dataset")

	return module.load_dvm_dataset


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot class balance histogram for full DVM dataset")
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="DVM dataset root directory")
	parser.add_argument(
		"--dvm-module-path",
		type=Path,
		default=DEFAULT_DVM_MODULE_PATH,
		help="Path to dvm_dataset*.py module that provides load_dvm_dataset",
	)
	
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output image path")
	parser.add_argument("--show", action="store_true", help="Display plot window")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	load_dvm_dataset = _import_load_dvm_dataset(args.dvm_module_path)

	train_loader, val_loader, test_loader, metadata = load_dvm_dataset(
		data_dir=str(args.data_dir),
		batch_size=1024,
		num_workers=0,

		use_images=False,
	)

	train_df = train_loader.dataset.data
	val_df = val_loader.dataset.data
	test_df = test_loader.dataset.data
	all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

	if "target_encoded" not in all_df.columns:
		raise ValueError("Expected 'target_encoded' in loaded dataset splits")

	class_counts = all_df["target_encoded"].value_counts().sort_index()

	plt.figure(figsize=(12, 5))
	plt.bar(class_counts.index.astype(str), class_counts.values)
	plt.xlabel("Class index (target_encoded)")
	plt.xticks([])
	plt.ylabel("Sample count")
	num_classes = metadata.get("num_classes", class_counts.shape[0]) if isinstance(metadata, dict) else class_counts.shape[0]
	plt.title(f"DVM Class Balance (Full Dataset) | classes={num_classes}, samples={len(all_df)}")
	plt.tight_layout()

	args.output.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(args.output, dpi=200)
	print(f"Saved histogram to: {args.output}")
	print(f"Total samples: {len(all_df)} | Total classes: {class_counts.shape[0]}")
	print(f"Min/Median/Max class count: {class_counts.min()}/{int(class_counts.median())}/{class_counts.max()}")

	if args.show:
		plt.show()
	else:
		plt.close()


if __name__ == "__main__":
	main()
