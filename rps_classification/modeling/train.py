import argparse
from pathlib import Path
from typing import Tuple


import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm


from rps_classification.config import TrainConfig
from rps_classification.dataset import make_loaders

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    elif name == "resnet18": #TODO: eliminare!!
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")
    
def _evaluate(model: nn.Module, loader, device) -> Tuple[float, float]:
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y)
    preds = logits.argmax(1)
    acc = accuracy_score(y_true, preds)
    loss = nn.CrossEntropyLoss()(logits, y_true).item()
    return loss, acc

def train(cfg: TrainConfig, evaluate_only: bool = False, ckpt_path: str | None = None, report_dir: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dl, val_dl, test_dl = make_loaders(Path(cfg.data_dir), cfg.batch_size, cfg.num_workers, cfg.img_size)
    num_classes = len(train_dl.dataset.classes)
    model = build_model(cfg.model, num_classes).to(device)


    if evaluate_only:
        assert ckpt_path is not None, "--evaluate requires --ckpt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        loss, acc = _evaluate(model, test_dl, device)
        print(f"Test loss: {loss:.4f} | Test acc: {acc:.4f}")
        Path(report_dir or "./reports").mkdir(parents=True, exist_ok=True)
        with open(Path(report_dir or "./reports")/"test_metrics.txt", "w") as f:
            f.write(f"loss={loss}\nacc={acc}\n")
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.mixed_precision)

    best_val_acc = 0.0
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path("./reports")/"figures").mkdir(parents=True, exist_ok=True)


    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.mixed_precision):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=loss.item())


        val_loss, val_acc = _evaluate(model, val_dl, device)
        print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")


        # checkpointing
        torch.save(model.state_dict(), out_dir / "latest.pt")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

    # final test
    test_loss, test_acc = _evaluate(model, test_dl, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


    # save classification report
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(1).cpu())
            all_true.append(y)
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_true)


    report = classification_report(y_true, y_pred, target_names=test_dl.dataset.classes, output_dict=True)
    df = pd.DataFrame(report).T
    df.to_csv(Path("./reports")/"classification_report.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data/processed")
    p.add_argument("--out", type=str, default="./models")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "resnet18"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--report-dir", type=str, default="./reports")
    args = p.parse_args()


    cfg = TrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        img_size=args.img_size,
        model=args.model,
        seed=args.seed,
        mixed_precision=not args.no_amp,
    )
train(cfg, evaluate_only=args.evaluate, ckpt_path=args.ckpt, report_dir=args.report_dir)