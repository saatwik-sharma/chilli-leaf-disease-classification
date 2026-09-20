import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATASET_ROOT = "Chilli Leaf Disease Image Dataset for Classificati"
# Use the verified duplicate-aware split produced by create_final_split.py.
# Do NOT change this back to dataset_splits.json or dataset_splits_clean.json.
SPLITS_FILE = "dataset_splits_final.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_test_transform

class LeafDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item["path"]
        label = item["label"]
        
        # Open and convert RGBA / RGB / Grayscale to standard 3-channel RGB
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def build_or_load_splits(dataset_root=DATASET_ROOT, splits_file=SPLITS_FILE, random_state=42):
    """Load the pre-verified duplicate-aware split from JSON.

    The split is produced once by create_final_split.py and must NOT be
    regenerated here.  We raise an explicit error if the file is missing
    so we never silently fall back to a naive stratified split.
    """
    if not os.path.exists(splits_file):
        raise FileNotFoundError(
            f"Verified split file '{splits_file}' not found.\n"
            "Run create_final_split.py first to generate it."
        )

    print(f"Loading verified duplicate-aware split from {splits_file} ...")
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    n_test  = len(splits["test"])
    total   = n_train + n_val + n_test
    print(f"  Train: {n_train} ({n_train/total*100:.1f}%)  "
          f"Val: {n_val} ({n_val/total*100:.1f}%)  "
          f"Test: {n_test} ({n_test/total*100:.1f}%)  "
          f"Total: {total}")
    return splits

def get_dataloaders(batch_size=32, num_workers=2, dataset_root=DATASET_ROOT, splits_file=SPLITS_FILE):  # noqa: E501
    splits = build_or_load_splits(dataset_root, splits_file)
    train_transform, val_test_transform = get_transforms()

    train_ds = LeafDataset(splits["train"], transform=train_transform)
    val_ds = LeafDataset(splits["val"], transform=val_test_transform)
    test_ds = LeafDataset(splits["test"], transform=val_test_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, splits

if __name__ == "__main__":
    print("Testing dataset splitting and data loader creation...")
    train_l, val_l, test_l, splits = get_dataloaders(batch_size=16, num_workers=0)
    print(f"Classes: {splits['classes']}")
    print(f"Train batches: {len(train_l)}, Val batches: {len(val_l)}, Test batches: {len(test_l)}")
    
    # Check one batch
    for images, labels in train_l:
        print(f"Sample train batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Image tensor range: min={images.min().item():.3f}, max={images.max().item():.3f}")
        print(f"Sample labels: {labels[:8].tolist()}")
        break
    print("Dataset verification complete.")
