import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DENOMINATIONS = [10000, 20000, 50000, 100000, 200000, 500000]
CLASS_TO_IDX = {denom: idx for idx, denom in enumerate(DENOMINATIONS)}
IDX_TO_CLASS = {idx: denom for idx, denom in enumerate(DENOMINATIONS)}

def get_transforms(img_size=224):
    """
    Returns train and eval transforms.
    Ensures NO CenterCrop is used to preserve corner numbers.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Bỏ hue để bảo toàn 100% màu sắc nhận diện polymer
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform

class BanknoteDataset(Dataset):
    """
    Dataset loader for Vietnamese Polymer Banknotes from DATA/
    """
    def __init__(self, root_dir="DATA", split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        
        for denom in DENOMINATIONS:
            denom_folder = self.root_dir / str(denom)
            if not denom_folder.exists():
                continue
            class_idx = CLASS_TO_IDX[denom]
            for img_name in sorted(os.listdir(denom_folder)):
                if img_name.lower().endswith(".png"):
                    rel_path = f"{split}/{denom}/{img_name}"
                    img_path = denom_folder / img_name
                    self.samples.append((str(img_path), rel_path, class_idx, denom))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, rel_path, class_idx, denom = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_idx, denom, rel_path

def get_dataloaders(root_dir="DATA", batch_size=64, img_size=224, num_workers=2):
    train_tf, eval_tf = get_transforms(img_size)
    
    train_ds = BanknoteDataset(root_dir=root_dir, split="train", transform=train_tf)
    dev_ds   = BanknoteDataset(root_dir=root_dir, split="dev",   transform=eval_tf)
    test_ds  = BanknoteDataset(root_dir=root_dir, split="test",  transform=eval_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, dev_loader, test_loader

