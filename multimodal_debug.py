"""
Simple test script to fetch and inspect 1 item from each of the 5 datasets.
"""

from multimodal_datasets import (
    PaintingsPricePredictionDataset,
    COVID19ChestXrayDataset,
    SkinCancerDataset,
    PetFinderDataset,
    CBISDDSMBreastCancerDataset,
)


def test_paintings_dataset():
    """Test Paintings Price Prediction Dataset"""
    print("\n" + "="*60)
    print("Testing Paintings Price Prediction Dataset")
    print("="*60)
    
    dataset = PaintingsPricePredictionDataset()
    print(f"Dataset size: {len(dataset)}")
    
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    if 'image' in item:
        print(f"Image shape: {item['image'].shape}")
    print(f"Features shape: {item['features'].shape}")
    print(f"Target (price): {item['target'].item():.2f}")


def test_covid19_dataset():
    """Test COVID-19 Chest X-ray Dataset"""
    print("\n" + "="*60)
    print("Testing COVID-19 Chest X-ray Dataset")
    print("="*60)
    
    dataset = COVID19ChestXrayDataset()
    print(f"Dataset size: {len(dataset)}")
    
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    if 'image' in item:
        print(f"Image shape: {item['image'].shape}")
    print(f"Clinical features: {item['clinical_features']}")
    print(f"Target: {item['target']}")


def test_skin_cancer_dataset():
    """Test Skin Cancer Dataset"""
    print("\n" + "="*60)
    print("Testing Skin Cancer Dataset")
    print("="*60)
    
    dataset = SkinCancerDataset()
    print(f"Dataset size: {len(dataset)}")
    
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    if 'image' in item:
        print(f"Image shape: {item['image'].shape}")
    print(f"Features shape: {item['features'].shape}")
    print(f"Target (class): {item['target'].item()}")


def test_petfinder_dataset():
    """Test PetFinder Adoption Prediction Dataset"""
    print("\n" + "="*60)
    print("Testing PetFinder Adoption Prediction Dataset")
    print("="*60)
    
    dataset = PetFinderDataset(split="train")
    print(f"Dataset size: {len(dataset)}")
    
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    if 'image' in item:
        print(f"Image shape: {item['image'].shape}")
    print(f"Features shape: {item['features'].shape}")
    if 'target' in item:
        print(f"Target (adoption speed): {item['target'].item()}")


def test_cbis_ddsm_dataset():
    """Test CBIS-DDSM Breast Cancer Dataset"""
    print("\n" + "="*60)
    print("Testing CBIS-DDSM Breast Cancer Dataset")
    print("="*60)
    
    dataset = CBISDDSMBreastCancerDataset(split="train", lesion_type="mass")
    print(f"Dataset size: {len(dataset)}")
    
    item = dataset[0]
    print(f"Item keys: {item.keys()}")
    if 'image' in item:
        print(f"Image shape: {item['image'].shape}")
    print(f"Features shape: {item['features'].shape}")
    print(f"Target (benign/malignant): {item['target'].item()}")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Testing All 5 Kaggle Datasets")
    print("#"*60)
    
    try:
        test_paintings_dataset()
    except Exception as e:
        print(f"Error testing Paintings dataset: {e}")
    
    try:
        test_covid19_dataset()
    except Exception as e:
        print(f"Error testing COVID-19 dataset: {e}")
    
    try:
        test_skin_cancer_dataset()
    except Exception as e:
        print(f"Error testing Skin Cancer dataset: {e}")
    
    try:
        test_petfinder_dataset()
    except Exception as e:
        print(f"Error testing PetFinder dataset: {e}")
    
    try:
        test_cbis_ddsm_dataset()
    except Exception as e:
        print(f"Error testing CBIS-DDSM dataset: {e}")
    
    print("\n" + "#"*60)
    print("# Testing Complete")
    print("#"*60 + "\n")
