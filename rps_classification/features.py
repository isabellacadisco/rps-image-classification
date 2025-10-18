from pathlib import Path
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from .config import PATHS, Settings

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize(img_size, img_size),#interpolation=InterpolationMode.BICUBIC, antialias=True),  # lato corto
        #transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    return train_tf, test_tf

def make_loaders(cfg: Settings, processed_dir: Path = PATHS.DATA_PROC):
    train_tf, test_tf = build_transforms(cfg.img_size)
    train_ds = datasets.ImageFolder(processed_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(processed_dir / "val",   transform=test_tf)
    test_ds  = datasets.ImageFolder(processed_dir / "test",  transform=test_tf)

    def mkloader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                          persistent_workers=(cfg.num_workers>0))
    return mkloader(train_ds, True), mkloader(val_ds, False), mkloader(test_ds, False)
