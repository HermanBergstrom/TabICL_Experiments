# Feature Extraction Guide

This guide shows how to use `extract_features.py` to extract DINOv3 image features for different datasets.

## Usage

### Basic Command Structure
```bash
python extract_features.py --dataset <dataset-name> [OPTIONS]
```

## Supported Datasets

### 1. PetFinder Adoption Prediction
```bash
# Train split
python extract_features.py --dataset petfinder --split train --batch-size 64

# Test split
python extract_features.py --dataset petfinder --split test --batch-size 64
```

**Output:**
- `extracted_features/petfinder_train_dinov3_features.pt`
- `extracted_features/petfinder_test_dinov3_features.pt`

**Features:** Image features + tabular features (pet characteristics)

---

### 2. COVID-19 Chest X-ray
```bash
python extract_features.py --dataset covid19 --batch-size 64
```

**Output:**
- `extracted_features/covid19_dinov3_features.pt`

**Features:** Image features only (chest X-rays)

---

### 3. Paintings Price Prediction
```bash
python extract_features.py --dataset paintings --batch-size 64
```

**Output:**
- `extracted_features/paintings_dinov3_features.pt`

**Features:** Image features + tabular features (material, dimensions, etc.)

---

### 4. Skin Cancer (HAM10000)
```bash
python extract_features.py --dataset skin-cancer --batch-size 64
```

**Output:**
- `extracted_features/skin-cancer_dinov3_features.pt`

**Features:** Image features + tabular features (age, gender, etc.)

---

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size` | Batch size for feature extraction | 32 |
| `--num-workers` | Number of data loader workers | 4 |
| `--img-size` | Image size for model input | 224 |
| `--device` | Device to use (cuda/cpu) | cuda if available |
| `--data-dir` | Custom dataset directory | `datasets/<dataset-name>` |
| `--output` | Custom output path | `extracted_features/<dataset>_dinov3_features.pt` |
| `--repo-dir` | Path to DINOv3 repo | `../dinov3` |
| `--weights` | Path to model weights | `model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |

## Output Format

All extracted features are saved as PyTorch `.pt` files with the following structure:

```python
{
    "image_features": Tensor,      # Shape: [N, feature_dim]
    "tabular_features": Tensor,    # Shape: [N, num_features] (if available)
    "targets": Tensor,             # Shape: [N] or [N, num_classes]
    "indices": Tensor,             # Original indices
    "sample_ids": List[str],       # Sample identifiers
}
```

## Examples with Custom Options

### High-resolution features with custom output
```bash
python extract_features.py \
    --dataset petfinder \
    --split train \
    --img-size 384 \
    --batch-size 32 \
    --output my_features/petfinder_384.pt
```

### Multiple workers for faster processing
```bash
python extract_features.py \
    --dataset covid19 \
    --num-workers 8 \
    --batch-size 128
```

### CPU-only extraction
```bash
python extract_features.py \
    --dataset paintings \
    --device cpu \
    --batch-size 16
```

## SLURM Job Example

```bash
#!/bin/bash
#SBATCH --account=aip-rahulgk
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH -c 4

# Extract features for all datasets
python extract_features.py --dataset petfinder --split train --batch-size 64
python extract_features.py --dataset petfinder --split test --batch-size 64
python extract_features.py --dataset covid19 --batch-size 64
python extract_features.py --dataset paintings --batch-size 64
python extract_features.py --dataset skin-cancer --batch-size 64
```
