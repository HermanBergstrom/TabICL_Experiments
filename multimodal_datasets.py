"""
Dataloaders for 5 Kaggle datasets:
1. Paintings Price Prediction - Multimodal (tabular + images)
2. COVID-19 Chest X-ray - Image classification
3. Skin Cancer (HAM10000) - Image classification with tabular features
4. PetFinder Adoption Prediction - Tabular + images
5. CBIS-DDSM Breast Cancer - Medical images with metadata
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer


class PaintingsPricePredictionDataset(Dataset):
    """
    Paintings Price Prediction Dataset.
    Multimodal dataset with tabular features (material, styles, dimensions, etc.) and images.
    Target: price prediction (regression)
    """
    
    def __init__(
        self,
        data_dir: str = "datasets/paintings-price-prediction",
        image_transform: Optional[transforms.Compose] = None,
        return_image: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / "paintings_data.csv"
        self.image_dir = self.data_dir / "images"
        
        self.df = pd.read_csv(self.csv_path)
        
        #Drop 'styles' column, already encoded as multi-label columns
        if 'styles' in self.df.columns:
            self.df.drop(columns=['styles'], inplace=True)

        multi_label_cols = ['material']
        
        #Remove any parantheses (and the content inside) in 'material' column
        self.df['material'] = self.df['material'].str.replace(r'\(.*?\)', '', regex=True)

        for col in multi_label_cols:
            mlb = MultiLabelBinarizer()
            dummies = mlb.fit_transform(
                self.df[col].apply(lambda x: x.split(','))
            )
            styles_df = pd.DataFrame(
                dummies,
                columns=[f"{col}_{s}" for s in mlb.classes_]
            )
            self.df = pd.concat([self.df, styles_df], axis=1)
            self.df.drop(columns=[col], inplace=True)

    
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.return_image = return_image
        
    def __len__(self):
        return len(self.df)
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return [col for col in self.df.columns if col not in ['image_url', 'price']]
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Get image
        image_name = row['image_url'].split('/')[-1]
        image_path = self.image_dir / image_name
        
        data = {}
        if self.return_image and image_path.exists():
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            data['image'] = image
        
        # Get tabular features (exclude image_url and price for features)
        features = row.drop(['image_url', 'price']).values.astype(np.float32)


        data['features'] = torch.tensor(features, dtype=torch.float32)
        
        # Target
        data['target'] = torch.tensor(row['price'], dtype=torch.float32)
        
        return data


class COVID19ChestXrayDataset(Dataset):
    """
    COVID-19 Chest X-ray Dataset.
    Binary classification: COVID-19 vs. other findings
    Contains clinical metadata and X-ray images.
    """
    
    def __init__(
        self,
        data_dir: str = "datasets/covid19-chest-xray",
        image_transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "metadata.csv"
        self.image_dir = self.data_dir / "images"
        
        self.df = pd.read_csv(self.metadata_path)
        
        # Create binary labels: 1 for COVID-19, 0 for all other findings
        self.df['label'] = (self.df['finding'] == 'COVID-19').astype(int)
        
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        image_path = self.image_dir / row['filename']
        data = {}
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            data['image'] = image
        
        # Extract clinical features
        clinical_features = {
            'age': row['age'],
            'sex': 1 if row['sex'] == 'M' else 0,
            'RT_PCR_positive': 1 if row['RT_PCR_positive'] == 'Y' else 0,
        }
        data['clinical_features'] = clinical_features
        data['target'] = torch.tensor(row['label'], dtype=torch.long)
        
        return data


class SkinCancerDataset(Dataset):
    """
    Skin Cancer (HAM10000) Dataset.
    Multiclass classification with tabular metadata and dermatoscopy images.
    7 classes: nevi, melanoma, etc.
    """
    
    def __init__(
        self,
        data_dir: str = "datasets/skin-cancer",
        image_transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "metadata.csv"
        
        self.df = pd.read_csv(self.metadata_path)

        #Get unique diagnostic labels
        self.diagnostic_labels = self.df['diagnostic'].unique().tolist()
        self.diagnosis_to_id = {label: idx for idx, label in enumerate(self.diagnostic_labels)}
        self.id_to_diagnosis = {idx: label for label, idx in self.diagnosis_to_id.items()}
        
        #TODO: Double check handling of missing values
        # Encode categorical columns
        categorical_cols = ['smoke', 'drink', 'gender', 'background_father', 
        'background_mother', 'fitspatrick', 'region', 'pesticide', 'skin_cancer_history', 
        'cancer_history', 'has_piped_water','has_sewage_system', 'itch', 'grew', 'hurt', 
        'changed', 'bleed', 'elevation']

        #Perform mean imputation on diameter_1 and diameter_2 columns
        self.df['diameter_1'].fillna(self.df['diameter_1'].mean(), inplace=True)
        self.df['diameter_2'].fillna(self.df['diameter_2'].mean(), inplace=True)

        ##With missing
        self.df = pd.get_dummies(self.df, columns=categorical_cols, prefix=categorical_cols, dummy_na=True)
        
        #print(df.head())

        self.image_dirs = [
            self.data_dir / "imgs_part_1",
            self.data_dir / "imgs_part_2",
            self.data_dir / "imgs_part_3",
        ]
        
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        exclude_cols = ['img_id', 'patient_id', 'diagnostic', 'lesion_id']
        return [col for col in self.df.columns if col not in exclude_cols]
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Find image
        image_name = row['img_id']
        image_path = None
        for img_dir in self.image_dirs:
            candidate = img_dir / image_name
            if candidate.exists():
                image_path = candidate
                break
        
        data = {}
        if image_path:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            data['image'] = image
        
        # Tabular features
        feature_cols = self.get_feature_columns()
        
        features = row[feature_cols].values.astype(np.float32)
        data['features'] = torch.tensor(features, dtype=torch.float32)
        
        # Target
        diagnostic = row['diagnostic']
        target = self.diagnosis_to_id[diagnostic]
        data['target'] = torch.tensor(target, dtype=torch.long)
        
        return data


class PetFinderDataset(Dataset):
    """
    PetFinder Adoption Prediction Dataset.
    Multimodal dataset with pet characteristics and adoption images.
    Target: Adoption speed (ordinal classification: 0-4)
    """
    
    def __init__(
        self,
        data_dir: str = "datasets/petfinder-adoption-prediction",
        split: str = "train",  # "train" or "test"
        image_transform: Optional[transforms.Compose] = None,
        return_image: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.return_image = return_image
        
        if split == "train":
            self.csv_path = self.data_dir / "train" / "train.csv"
            self.image_dir = self.data_dir / "train_images"
            self.metadata_dir = self.data_dir / "train_metadata"
        else:
            self.csv_path = self.data_dir / "test" / "test.csv"
            self.image_dir = self.data_dir / "test_images"
            self.metadata_dir = self.data_dir / "test_metadata"
        
        self.df = pd.read_csv(self.csv_path)
        
        # Load label mappings
        breed_map = pd.read_csv(self.data_dir / "breed_labels.csv")
        color_map = pd.read_csv(self.data_dir / "color_labels.csv")
        state_map = pd.read_csv(self.data_dir / "state_labels.csv")

        self.breed_dict = dict(zip(breed_map['BreedID'], breed_map['BreedName']))
        self.color_dict = dict(zip(color_map['ColorID'], color_map['ColorName']))
        self.state_dict = dict(zip(state_map['StateID'], state_map['StateName']))
        
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return ['Type', 'Age', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health']
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pet_id = row['PetID']
        
        data = {}
        
        # Get primary image
        if self.return_image:
            image_files = list(self.image_dir.glob(f"{pet_id}*.jpg"))
            if image_files:
                image_path = image_files[0]
                image = Image.open(image_path).convert('RGB')
                image = self.image_transform(image)
                data['image'] = image
        
        # Tabular features
        feature_dict = {
            'Type': row['Type'] - 1,  # Cat=1, Dog=2 -> 0, 1
            'Age': row['Age'],
            'Gender': row['Gender'] - 1,  # Male=1, Female=2 -> 0, 1
            'MaturitySize': row['MaturitySize'],
            'FurLength': row['FurLength'],
            'Vaccinated': row['Vaccinated'],
            'Dewormed': row['Dewormed'],
            'Sterilized': row['Sterilized'],
            'Health': row['Health'],
        }
        
        features = np.array([feature_dict[k] for k in feature_dict.keys()], dtype=np.float32)
        data['features'] = torch.tensor(features, dtype=torch.float32)
        
        # Target (only for train split)
        if 'AdoptionSpeed' in row:
            data['target'] = torch.tensor(row['AdoptionSpeed'], dtype=torch.long)
        
        return data


#TODO: Broken, redo
class CBISDDSMBreastCancerDataset(Dataset):
    """
    CBIS-DDSM Breast Cancer Dataset.
    Binary classification: Benign vs. Malignant
    Mammography images with clinical metadata.
    """
    
    def __init__(
        self,
        data_dir: str = "datasets/cbis-ddsm-breast-cancer-image-dataset",
        split: str = "train",  # "train" or "test"
        lesion_type: str = "mass",  # "mass" or "calc"
        image_transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.lesion_type = lesion_type
        
        csv_prefix = "mass" if lesion_type == "mass" else "calc"
        if split == "train":
            self.metadata_path = (
                self.data_dir / "csv" / f"{csv_prefix}_case_description_train_set.csv"
            )
        else:
            self.metadata_path = (
                self.data_dir / "csv" / f"{csv_prefix}_case_description_test_set.csv"
            )
        
        self.df = pd.read_csv(self.metadata_path)
        self.jpeg_dir = self.data_dir / "jpeg"
        
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Construct image path
        image_folder = row['image file path'].replace('CBIS-DDSM/jpeg/', '')
        image_path = self.jpeg_dir / image_folder
        
        data = {}
        if image_path.exists():
            image = Image.open(image_path)
            image = self.image_transform(image)
            data['image'] = image
        
        # Metadata features
        feature_dict = {
            'breast_density': row['breast density'],
        }
        
        # Try to convert age
        try:
            age = float(feature_dict['patient_age'])
        except:
            age = 0
        
        features = np.array([age, feature_dict['breast_density']], dtype=np.float32)
        data['features'] = torch.tensor(features, dtype=torch.float32)
        
        # Target: Benign (0) or Malignant (1)
        pathology = str(row['pathology']).lower()
        target = 1 if 'malignant' in pathology else 0
        data['target'] = torch.tensor(target, dtype=torch.long)
        
        return data


def get_dataloader(
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Convenient function to get dataloaders for any of the 5 datasets.
    
    Args:
        dataset_name: One of ['paintings', 'covid19', 'skin_cancer', 'petfinder', 'cbis_ddsm']
        batch_size: Batch size for dataloader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the dataset
        **kwargs: Additional arguments to pass to dataset constructor
    
    Returns:
        DataLoader object
    """
    
    dataset_map = {
        'paintings': PaintingsPricePredictionDataset,
        'covid19': COVID19ChestXrayDataset,
        'skin_cancer': SkinCancerDataset,
        'petfinder': PetFinderDataset,
        'cbis_ddsm': CBISDDSMBreastCancerDataset,
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_map.keys())}")
    
    dataset_class = dataset_map[dataset_name]
    dataset = dataset_class(**kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


if __name__ == "__main__":
    # Example usage
    print("Testing dataloaders...\n")
    
    # Paintings dataset
    print("1. Paintings Price Prediction Dataset")
    paintings_dl = get_dataloader('paintings', batch_size=4)
    batch = next(iter(paintings_dl))
    print(f"   Sample batch keys: {batch.keys()}")
    if 'image' in batch:
        print(f"   Image shape: {batch['image'].shape}")
    print(f"   Features shape: {batch['features'].shape}")
    print(f"   Target shape: {batch['target'].shape}\n")

    # COVID-19 dataset
    print("2. COVID-19 Chest X-ray Dataset")
    covid_dl = get_dataloader('covid19', batch_size=4)
    batch = next(iter(covid_dl))
    print(f"   Sample batch keys: {batch.keys()}")
    if 'image' in batch:
        print(f"   Image shape: {batch['image'].shape}")
    print(f"   Clinical features: {list(batch['clinical_features'].keys())}\n")
    
    # Skin Cancer dataset
    print("3. Skin Cancer Dataset")
    skin_dl = get_dataloader('skin_cancer', batch_size=4)
    batch = next(iter(skin_dl))
    print(f"   Sample batch keys: {batch.keys()}")
    if 'image' in batch:
        print(f"   Image shape: {batch['image'].shape}")
    print(f"   Features shape: {batch['features'].shape}")
    print(f"   Target shape: {batch['target'].shape}\n")
    
    # PetFinder dataset
    print("4. PetFinder Adoption Prediction Dataset")
    petfinder_dl = get_dataloader('petfinder', batch_size=4)
    batch = next(iter(petfinder_dl))
    print(f"   Sample batch keys: {batch.keys()}")
    if 'image' in batch:
        print(f"   Image shape: {batch['image'].shape}")
    print(f"   Features shape: {batch['features'].shape}")
    if 'target' in batch:
        print(f"   Target shape: {batch['target'].shape}\n")
    
    # CBIS-DDSM dataset
    print("5. CBIS-DDSM Breast Cancer Dataset")
    cbis_dl = get_dataloader('cbis_ddsm', batch_size=4)
    batch = next(iter(cbis_dl))
    print(f"   Sample batch keys: {batch.keys()}")
    if 'image' in batch:
        print(f"   Image shape: {batch['image'].shape}")
    print(f"   Features shape: {batch['features'].shape}")
    print(f"   Target shape: {batch['target'].shape}")
