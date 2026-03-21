#!/usr/bin/env python
"""Extract Dinov3 image features for multimodal datasets and save for TabICL.

Example:
  python extract_features.py --dataset petfinder --split train --batch-size 64
  python extract_features.py --dataset covid19 --batch-size 64
  python extract_features.py --dataset paintings --batch-size 64
  python extract_features.py --dataset skin-cancer --batch-size 64
    python extract_features.py --dataset petfinder --backend vertexai --vertex-project your-project-id
    python extract_features.py --dataset covid19 --backend vertexai --vertex-project your-project-id
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image

# Allow imports from repository root when this script is run from feature_extraction/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multimodal_datasets import (
    PetFinderDataset,
    COVID19ChestXrayDataset,
    PaintingsPricePredictionDataset,
    SkinCancerDataset,
)
from tqdm import tqdm


TEXT_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_VERTEX_CONTEXTUAL_TEXT_CHARS = 768#100 #768


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def make_transform(resize_size: int | List[int] = 768) -> v2.Compose:
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def make_vertex_transform(resize_size: int | List[int] = 768) -> v2.Compose:
    """Transform images to uint8 tensors suitable for Vertex image embedding API."""
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_uint8 = v2.ToDtype(torch.uint8, scale=True)
    return v2.Compose([to_tensor, resize, to_uint8])


def tensor_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    """Convert CHW tensor in uint8/float format to PNG bytes for Vertex API."""
    image = image_tensor.detach().cpu()
    if image.dtype != torch.uint8:
        image = torch.clamp(image, 0.0, 1.0)
        image = (image * 255.0).to(torch.uint8)

    if image.dim() != 3:
        raise ValueError(f"Expected CHW image tensor, got shape: {tuple(image.shape)}")

    image_hwc = image.permute(1, 2, 0).numpy()
    pil_image = Image.fromarray(image_hwc)
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()


def encode_text_batch(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    device: str,
) -> torch.Tensor:
    """Encode a batch of texts with BGE and return normalized embeddings."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    outputs = model(**encoded)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.detach().float().cpu()


class IndexedDataset(Dataset):
    """Wraps any base dataset and adds index and ID information."""
    
    def __init__(self, base: Dataset, dataset_name: str):
        self.base = base
        self.dataset_name = dataset_name

    def __len__(self) -> int:  # noqa: D401
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        item["index"] = idx
        
        # Extract ID based on dataset type
        if self.dataset_name == "petfinder":
            item["sample_id"] = self.base.df.iloc[idx]["PetID"]
        elif self.dataset_name == "covid19":
            item["sample_id"] = self.base.df.iloc[idx]["filename"]
        elif self.dataset_name == "paintings":
            item["sample_id"] = self.base.df.iloc[idx]["image_url"].split('/')[-1]
        elif self.dataset_name == "skin-cancer":
            item["sample_id"] = self.base.df.iloc[idx]["img_id"]
        else:
            item["sample_id"] = f"sample_{idx}"
        
        return item


def collate_multimodal(batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """General collate function for multimodal datasets."""
    batch = [b for b in batch if b.get("image") is not None]
    if not batch:
        return None

    images = torch.stack([b["image"] for b in batch])
    
    # Handle tabular features if present
    tab_features = None
    if "features" in batch[0]:
        tab_features = torch.stack([b["features"] for b in batch])
    
    # Handle targets
    targets = None
    if "target" in batch[0]:
        targets = torch.stack([b["target"] for b in batch])

    indices = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    sample_ids = [b["sample_id"] for b in batch]
    texts = [b["text"] for b in batch] if "text" in batch[0] else None

    result = {
        "image": images,
        "indices": indices,
        "sample_ids": sample_ids,
    }
    
    if tab_features is not None:
        result["features"] = tab_features
    if targets is not None:
        result["targets"] = targets
    if texts is not None:
        result["texts"] = texts
    
    return result


def get_dataset(
    dataset_name: str,
    data_dir: Path,
    split: Optional[str],
    transform: v2.Compose,
) -> Dataset:
    """Get the appropriate dataset based on name."""
    if dataset_name == "petfinder":
        if split is None:
            split = "train"
        return PetFinderDataset(
            data_dir=str(data_dir),
            split=split,
            image_transform=transform,
            return_image=True,
            return_text=True,
        )
    elif dataset_name == "covid19":
        return COVID19ChestXrayDataset(
            data_dir=str(data_dir),
            image_transform=transform,
        )
    elif dataset_name == "paintings":
        return PaintingsPricePredictionDataset(
            data_dir=str(data_dir),
            image_transform=transform,
            return_image=True,
        )
    elif dataset_name == "skin-cancer":
        return SkinCancerDataset(
            data_dir=str(data_dir),
            image_transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_features(
    dataset_name: str,
    data_dir: Path,
    split: Optional[str],
    output_path: Path,
    batch_size: int,
    num_workers: int,
    img_size: int,
    repo_dir: Path,
    weights_path: Path,
    device: str,
    backend: str,
    vertex_project: Optional[str],
    vertex_location: str,
    vertex_model_name: str,
    vertex_dimension: Optional[int],
    vertex_service_account_key: Optional[Path],
    max_samples: Optional[int],
    subset_seed: int,
) -> None:
    if backend == "dinov3":
        transform = make_transform(img_size)
    elif backend == "vertexai":
        transform = make_vertex_transform(img_size)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    base_dataset = get_dataset(dataset_name, data_dir, split, transform)
    dataset = IndexedDataset(base_dataset, dataset_name)

    if max_samples is not None and max_samples > 0:
        subset_size = min(max_samples, len(dataset))
        rng = random.Random(subset_seed)
        subset_indices = rng.sample(range(len(dataset)), k=subset_size)
        dataset = torch.utils.data.Subset(dataset, subset_indices)
        print(
            f"Using subset of {subset_size}/{len(base_dataset)} samples "
            f"(seed={subset_seed})."
        )
    
    # Extract column names from the dataset if available
    column_names = None
    if hasattr(base_dataset, 'get_feature_columns'):
        column_names = base_dataset.get_feature_columns()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
        collate_fn=collate_multimodal,
    )

    model = None
    vertex_image_cls = None
    if backend == "dinov3":
        model = torch.hub.load(str(repo_dir), "dinov3_vitb16", source="local", weights=str(weights_path))
        model.eval()
        model.to(device)
    else:
        try:
            import vertexai
            from vertexai.vision_models import Image as VertexImage
            from vertexai.vision_models import MultiModalEmbeddingModel
        except ImportError as exc:
            raise ImportError(
                "Vertex backend requires google-cloud-aiplatform. "
                "Install with: pip install google-cloud-aiplatform"
            ) from exc

        credentials = None
        resolved_project = vertex_project
        if vertex_service_account_key is not None:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                str(vertex_service_account_key)
            )
            # Match vertex_test behavior while making project explicit when omitted.
            if not resolved_project and credentials.project_id:
                resolved_project = credentials.project_id

        if not resolved_project:
            raise ValueError(
                "Vertex project is required. Pass --vertex-project or provide "
                "--vertex-service-account-key with a project_id field."
            )

        vertexai.init(
            project=resolved_project,
            location=vertex_location,
            experiment=None,
            staging_bucket=None,
            credentials=credentials,
            encryption_spec_key_name=None,
        )
        model = MultiModalEmbeddingModel.from_pretrained(vertex_model_name)
        vertex_image_cls = VertexImage

    image_features: List[torch.Tensor] = []
    tabular_features: List[torch.Tensor] = []
    text_features: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    indices_list: List[torch.Tensor] = []
    sample_ids_list: List[str] = []
    vertex_truncated_texts = 0

    text_tokenizer = None
    text_model = None
    if dataset_name == "petfinder" and backend == "dinov3":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Text embedding for PetFinder requires transformers. "
                "Install with: pip install transformers"
            ) from exc

        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBED_MODEL_NAME)
        text_model = AutoModel.from_pretrained(TEXT_EMBED_MODEL_NAME)
        text_model.eval()
        text_model.to(device)

    use_amp = backend == "dinov3" and device.startswith("cuda")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.inference_mode():
        for batch in tqdm(loader):

            if batch is None:
                continue

            if backend == "dinov3":
                images = batch["image"].to(device, non_blocking=True)
                with autocast_ctx:
                    feats = model(images)
                image_features.append(feats.detach().float().cpu())
            else:
                batch_embeddings: List[torch.Tensor] = []
                batch_text_embeddings: List[torch.Tensor] = []
                batch_texts = batch.get("texts")
                for sample_idx, image_tensor in enumerate(batch["image"]):

                    #print(f"Processing sample_id={batch['sample_ids'][sample_idx]}")
                    #if batch['sample_ids'][sample_idx] != "2ac4dd8c2":
                    #    continue

                    png_bytes = tensor_to_png_bytes(image_tensor)
                    try:
                        vertex_image = vertex_image_cls(image_bytes=png_bytes)
                    except TypeError:
                        vertex_image = vertex_image_cls(png_bytes)
                    contextual_text = None if batch_texts is None else batch_texts[sample_idx]
                    contextual_text = remove_emoji(contextual_text) if contextual_text is not None else None
                    #breakpoint()
                    if contextual_text is not None and len(contextual_text) > MAX_VERTEX_CONTEXTUAL_TEXT_CHARS:
                        #original_word_count = len(contextual_text.split())
                        contextual_text = contextual_text[:MAX_VERTEX_CONTEXTUAL_TEXT_CHARS]
                        #cropped_word_count = len(contextual_text.split())
                        #print(
                        #    f"Truncated contextual_text for sample_id={batch['sample_ids'][sample_idx]}: "
                        #    f"{original_word_count} -> {cropped_word_count} words"
                        #)
                        vertex_truncated_texts += 1
                        
                    try:
                        if vertex_dimension is not None:
                            if contextual_text is not None:
                                embedding_response = model.get_embeddings(
                                    image=vertex_image,
                                    dimension=vertex_dimension,
                                    contextual_text=contextual_text,
                                )
                            else:
                                embedding_response = model.get_embeddings(
                                    image=vertex_image,
                                    dimension=vertex_dimension,
                                )
                        else:
                            if contextual_text is not None:
                                embedding_response = model.get_embeddings(
                                    image=vertex_image,
                                    contextual_text=contextual_text,
                                )
                            else:
                                embedding_response = model.get_embeddings(image=vertex_image)
                    except Exception:
                        passed_text = contextual_text
                        passed_chars = 0 if passed_text is None else len(passed_text)
                        passed_words = 0 if passed_text is None else len(passed_text.split())
                        print(
                            f"Vertex get_embeddings failed for sample_id={batch['sample_ids'][sample_idx]}"
                        )
                        print(
                            f"Passed contextual_text stats: chars={passed_chars}, words={passed_words}"
                        )
                        print("Passed contextual_text:")
                        print("<None>" if passed_text is None else passed_text)
                        raise
                    batch_embeddings.append(
                        torch.tensor(embedding_response.image_embedding, dtype=torch.float32)
                    )
                    if contextual_text is not None and embedding_response.text_embedding is not None:
                        batch_text_embeddings.append(
                            torch.tensor(embedding_response.text_embedding, dtype=torch.float32)
                        )
                image_features.append(torch.stack(batch_embeddings, dim=0))
                if batch_text_embeddings:
                    text_features.append(torch.stack(batch_text_embeddings, dim=0))

            indices_list.append(batch["indices"].detach().cpu())
            sample_ids_list.extend(batch["sample_ids"])

            if text_model is not None and "texts" in batch:
                text_features.append(
                    encode_text_batch(
                        batch["texts"],
                        text_tokenizer,
                        text_model,
                        device,
                    )
                )

            # Tabular features are optional
            if "features" in batch:
                tabular_features.append(batch["features"].detach().float().cpu())

            # Targets may not exist for test splits
            if "targets" in batch and batch["targets"] is not None:
                targets_list.append(batch["targets"].detach().cpu())

    output = {
        "image_features": torch.cat(image_features, dim=0),
        "indices": torch.cat(indices_list, dim=0),
        "sample_ids": sample_ids_list,
    }
    
    if tabular_features:
        output["tabular_features"] = torch.cat(tabular_features, dim=0)
        if column_names is not None:
            output["column_names"] = column_names

    if text_features:
        output["text_features"] = torch.cat(text_features, dim=0)
        output["text_model_name"] = TEXT_EMBED_MODEL_NAME
    
    if targets_list:
        output["targets"] = torch.cat(targets_list, dim=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f"Saved features to: {output_path}")
    print(f"Image features shape: {output['image_features'].shape}")
    if "tabular_features" in output:
        print(f"Tabular features shape: {output['tabular_features'].shape}")
        if "column_names" in output:
            print(f"Column names: {output['column_names']}")
    if "text_features" in output:
        print(f"Text features shape: {output['text_features'].shape}")
        print(f"Text model: {output['text_model_name']}")
    if backend == "vertexai" and vertex_truncated_texts > 0:
        print(
            f"Vertex contextual_text truncations: {vertex_truncated_texts} "
            f"(max chars={MAX_VERTEX_CONTEXTUAL_TEXT_CHARS})"
        )
    if "targets" in output:
        print(f"Targets shape: {output['targets'].shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract image features for multimodal datasets using Dinov3 or Vertex AI."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["petfinder", "covid19", "paintings", "skin-cancer"],
        required=True,
        help="Dataset to extract features from",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to dataset directory (defaults to datasets/<dataset-name>)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default=None,
        help="Dataset split (only applicable for petfinder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for features (defaults depend on backend)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["dinov3", "vertexai"],
        default="dinov3",
        help="Feature extraction backend",
    )
    parser.add_argument("--repo-dir", type=Path, default=Path("../dinov3"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("model_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--vertex-project",
        type=str,
        default=None,
        help="GCP project ID for Vertex AI (required unless already configured)",
    )
    parser.add_argument(
        "--vertex-location",
        type=str,
        default="us-central1",
        help="Vertex AI region",
    )
    parser.add_argument(
        "--vertex-model-name",
        type=str,
        default="multimodalembedding",
        help="Vertex multimodal embedding model name",
    )
    parser.add_argument(
        "--vertex-dimension",
        type=int,
        default=None,
        help="Optional embedding dimension for Vertex model",
    )
    parser.add_argument(
        "--vertex-service-account-key",
        type=Path,
        default=None,
        help="Path to service account JSON key for Vertex authentication",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional number of randomly sampled images to process for quick tests",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Random seed for subset sampling when --max-samples is used",
    )
    
    args = parser.parse_args()
    
    # Set default data_dir if not provided
    if args.data_dir is None:
        dataset_dir_map = {
            "petfinder": "petfinder-adoption-prediction",
            "covid19": "covid19-chest-xray",
            "paintings": "paintings-price-prediction",
            "skin-cancer": "skin-cancer",
        }
        args.data_dir = Path("datasets") / dataset_dir_map[args.dataset]
    
    # Set default output path if not provided
    if args.output is None:
        split_suffix = f"_{args.split}" if args.split else ""
        if args.backend == "dinov3":
            args.output = Path(f"extracted_features/{args.dataset}{split_suffix}_dinov3_features.pt")
        else:
            args.output = Path(f"extracted_features/{args.dataset}{split_suffix}_vertexai_features.pt")

    if args.backend == "vertexai" and args.vertex_service_account_key is not None:
        args.vertex_service_account_key = args.vertex_service_account_key.expanduser()
    
    return args


def main() -> None:
    args = parse_args()
    extract_features(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        repo_dir=args.repo_dir,
        weights_path=args.weights,
        device=args.device,
        backend=args.backend,
        vertex_project=args.vertex_project,
        vertex_location=args.vertex_location,
        vertex_model_name=args.vertex_model_name,
        vertex_dimension=args.vertex_dimension,
        vertex_service_account_key=args.vertex_service_account_key,
        max_samples=args.max_samples,
        subset_seed=args.subset_seed,
    )



if __name__ == "__main__":
    main()
