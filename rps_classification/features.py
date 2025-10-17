import argparse
from pathlib import Path
from typing import Tuple
import shutil
import random


from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# canonical transforms shared by train/eval

def make_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf

def _copy_subset(filepaths, dst_dir: Path):
    for src in filepaths:
        rel = Path(src).parent.name # class name
        out_dir = dst_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_dir / Path(src).name)

def prepare_splits(raw_dir: Path, out_dir: Path, val_ratio: float, test_ratio: float, seed: int = 42):
    assert raw_dir.exists(), f"Raw dir not found: {raw_dir}"
    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    rng = random.Random(seed)

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    for cls in classes:
        files = list((raw_dir / cls).glob("*.jpg")) + list((raw_dir / cls).glob("*.png"))
        rng.shuffle(files)
        n = len(files)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]
        _copy_subset(train_files, out_dir / "train" / cls)
        _copy_subset(val_files, out_dir / "val" / cls)
        _copy_subset(test_files, out_dir / "test" / cls)


def make_loaders(processed_dir: Path, batch_size: int, num_workers: int, img_size: int):
    train_tf, eval_tf = make_transforms(img_size)
    train_ds = datasets.ImageFolder(processed_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(processed_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(processed_dir / "test", transform=eval_tf)


    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl