import argparse, json, math, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets
from ..config import PATHS, Settings
from ..features import build_transforms, make_loaders, ListDataset
import pandas as pd, json as js, torch.nn as nn

import random, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ----------------- MODELS -----------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=3, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

class TinyCNN(nn.Module):
    def __init__(self, num_classes=3, img_size=192):
        super().__init__()
        img_size = int(cfg.img_size)  # Ensure img_size is an integer
        
        print("[DEBUG] Initial img_size:", img_size)  # Debugging line

        feature_map_size = img_size // 8  # Calculate the size after 3 MaxPool2d layers
        if feature_map_size <= 0:
            raise ValueError(f"Invalid img_size: {img_size}. Ensure img_size is large enough to pass through the network.")
        print(f"[DEBUG] Feature map size: {feature_map_size}x{feature_map_size}")  # Debugging line
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 64x64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),# 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2) # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_map_size * feature_map_size, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class MediumCNN(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3), nn.ReLU(),
            nn.Conv2d(32,32,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(),
            nn.Conv2d(64,64,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class LargeCNN(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self,x): return self.classifier(self.pool(self.features(x)))

def build_model(model_id, num_classes, dropout):
    if model_id=="small":  return SmallCNN(num_classes, dropout)
    if model_id=="medium": return MediumCNN(num_classes, dropout)
    if model_id=="large":  return LargeCNN(num_classes, dropout)
    
    if model_id=="tiny":  return TinyCNN(num_classes, dropout)

    raise ValueError(f"Unknown model_id: {model_id}")

# ----------------- TRAIN / EVAL -----------------
def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def one_epoch(model, loader, crit, opt=None, device="cpu"):
    is_train = opt is not None
    model.train(is_train)
    loss_sum, correct, n = 0.0, 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        if is_train: opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        if is_train:
            loss.backward()
            opt.step()
        loss_sum += loss.item()*x.size(0)
        correct += (logits.argmax(1)==y).sum().item()
        n += x.size(0)
    return loss_sum/n, correct/n

def fit_once(model_id, params, train_loader, val_loader, device, epochs=20, patience=5, dropout=0.3):
    model = build_model(model_id, params["num_classes"], params.get("dropout",0.3)).to(device)

    opt = optim.Adam(model.parameters(), lr=params["lr"])
    crit = nn.CrossEntropyLoss()
    best = {"val_acc": -1, "state": None}
    patience_left = patience
    history = []
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = one_epoch(model, train_loader, crit, opt, device)
        va_loss, va_acc = one_epoch(model, val_loader, crit, None, device)
        history.append({"epoch":ep, "train_loss":tr_loss, "train_acc":tr_acc, "val_loss":va_loss, "val_acc":va_acc})
        if va_acc > best["val_acc"]:
            best = {"val_acc": va_acc, "state": model.state_dict()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0: break
    model.load_state_dict(best["state"])
    return model, history, best["val_acc"]

# ----------------- CV HELPERS -----------------
def _list_paths_labels(dirpath: Path):
    ds = datasets.ImageFolder(dirpath)
    paths = [p[0] for p in ds.samples]
    labels = [int(p[1]) for p in ds.samples]
    classes = ds.classes
    return np.array(paths), np.array(labels), classes

def _make_loader_from_lists(paths, labels, batch, img_size, shuffle, workers, pin):
    # usa le stesse trasformazioni di val/test (augment SOLO nel fold train via flag)
    train_tf, test_tf = build_transforms(img_size)
    tfm = train_tf if shuffle else test_tf

    lst_ds = ListDataset(paths, labels, tfm)

    return torch.utils.data.DataLoader(lst_ds, batch_size=batch, shuffle=shuffle,
                                       num_workers=workers, pin_memory=pin,
                                       persistent_workers=(workers>0))

# ----------------- MAIN LOGIC -----------------
def arch_cv(k, cfg: Settings):

    # sovrascrivo num workers a 0 
    #cfg.num_workers = 0 #TODO: sistemato ListDataset, se non si risolve rimettere a zero 
    # oppure provare altra soluzione con PathDataset

    set_seeds(cfg.seed)
    X_tr, y_tr, _ = _list_paths_labels(PATHS.DATA_PROC / "train")
    X_va, y_va, _ = _list_paths_labels(PATHS.DATA_PROC / "val")
    X = np.concatenate([X_tr, X_va]); y = np.concatenate([y_tr, y_va])

    # SKF garantisce che la proporzione delle classi sia il più possibile simile tra tutti i fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for model_id in ["small","medium","large"]:
        fold_acc = []
        # Stratified K-Fold CV
        for tr_idx, va_idx in skf.split(X,y):
            tr_loader = _make_loader_from_lists(X[tr_idx], y[tr_idx], cfg.batch, cfg.img_size, True, cfg.num_workers, cfg.pin_memory)
            va_loader = _make_loader_from_lists(X[va_idx], y[va_idx], cfg.batch, cfg.img_size, False, cfg.num_workers, cfg.pin_memory)
            model, hist, acc = fit_once(model_id, {"num_classes":3,"lr":cfg.lr}, tr_loader, va_loader, device, epochs=min(cfg.epochs,8), patience=3)
            fold_acc.append(acc)
        results.append({"model_id": model_id, "mean_val_acc": float(np.mean(fold_acc)), "std_val_acc":float(np.std(fold_acc)), "folds": [float(a) for a in fold_acc]})
        print(f"[ARCH CV] {model_id}: mean={np.mean(fold_acc):.4f}")
    best = max(results, key=lambda r: r["mean_val_acc"])
    json.dump(results, open(PATHS.MODELS/"arch_cv_results.json","w"), indent=2)
    json.dump(best, open(PATHS.MODELS/"best_arch.json","w"), indent=2)
    print("[BEST ARCH]", best)

def grid_cv(k, grid, cfg: Settings, arch=None):
    set_seeds(cfg.seed)

    # sovrascrivo num workers a 0 
    #cfg.num_workers = 0 #TODO: sistemato ListDataset, se non si risolve rimettere a zero 
    # oppure provare altra soluzione con PathDataset

    # best_arch è il model_id della migliore architettura trovata in arch_cv
    # siccome la migliore architettura potrebbe essere "large" che richiede più memoria,
    # rendo possibile specificare manualmente il model_id da usare per il grid search
    if arch is None:
        best_arch = json.load(open(PATHS.MODELS/"best_arch.json"))["model_id"]
    else:
        best_arch = arch

    X_tr, y_tr, _ = _list_paths_labels(PATHS.DATA_PROC / "train")
    X_va, y_va, _ = _list_paths_labels(PATHS.DATA_PROC / "val")
    X = np.concatenate([X_tr, X_va]); y = np.concatenate([y_tr, y_va])

    # SKF garantisce che la proporzione delle classi sia il più possibile simile tra tutti i fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for lr in grid["lr"]: #TODO: rendere generico e dinamico 
        for batch in grid["batch"]:  #TODO: rendere generico e dinamico
            for dropout in grid["dropout"]:  #TODO: rendere generico e dinamico 
                for epochs in grid.get("epochs",[cfg.epochs]): #TODO: rendere generico e dinamico
                    fold_acc = []
                    # Stratified K-Fold CV
                    for tr_idx, va_idx in skf.split(X,y):
                        tr_loader = _make_loader_from_lists(X[tr_idx], y[tr_idx], batch, cfg.img_size, True, cfg.num_workers, cfg.pin_memory)
                        va_loader = _make_loader_from_lists(X[va_idx], y[va_idx], batch, cfg.img_size, False, cfg.num_workers, cfg.pin_memory)
                        params = {"num_classes":3, "lr":lr, "dropout":dropout}
                        _, _, acc = fit_once(best_arch, params, tr_loader, va_loader, device, epochs=epochs, patience=cfg.patience)
                        fold_acc.append(acc)
                    results.append({"arch":best_arch,"params":{"lr":lr,"batch":batch,"dropout":dropout,"epochs":epochs},"mean_val_acc":float(np.mean(fold_acc)), "std_val_acc":float(np.std(fold_acc)), "folds": [float(a) for a in fold_acc]})
                    print(f"[GRID] {best_arch} {results[-1]['params']} -> {results[-1]['mean_val_acc']:.4f}")
    best = max(results, key=lambda r: r["mean_val_acc"])
    json.dump(results, open(PATHS.MODELS/"grid_cv_results.json","w"), indent=2)
    json.dump(best, open(PATHS.MODELS/f"best_params_{arch}.json","w"), indent=2)
    print("[BEST PARAMS]", best)

def retrain_and_eval(cfg: Settings, exp_name="final", arch=None):
    set_seeds(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if arch is None:
        best_arch = json.load(open(PATHS.MODELS/"best_arch.json"))["model_id"]
        best_params = json.load(open(PATHS.MODELS/"best_params.json"))["params"]
    else:
        best_arch = arch
        best_params = json.load(open(PATHS.MODELS/f"best_params_{arch}.json"))["params"]
  
    # loader standard (augment su train)
    train_loader, val_loader, test_loader = make_loaders(cfg)
    # unisci train+val
    full_train = torch.utils.data.ConcatDataset([train_loader.dataset, val_loader.dataset])
    full_loader = torch.utils.data.DataLoader(full_train, batch_size=best_params["batch"], shuffle=True,
                                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                                              persistent_workers=(cfg.num_workers>0))
    model = build_model(best_arch, 3, best_params.get("dropout",0.3)).to(device)
    opt = optim.Adam(model.parameters(), lr=best_params["lr"])
    crit = nn.CrossEntropyLoss()
    # simple early-stopping on train loss
    best_loss, patience_left = float("inf"), cfg.patience
    for ep in range(best_params.get("epochs", cfg.epochs)):
        tr_loss, tr_acc = one_epoch(model, full_loader, crit, opt, device)
        if tr_loss < best_loss - 1e-4:
            best_loss, best_state = tr_loss, model.state_dict()
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left == 0: break
    model.load_state_dict(best_state)
    # test eval
    test_loss, test_acc = one_epoch(model, test_loader, crit, None, device)
    # per report: classification report & CM
    y_true, y_pred = [], []
    classes = [l.strip() for l in (PATHS.DATA_PROC/"classes.txt").read_text().splitlines()]
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            logits = model(x)
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(y.tolist())
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm  = confusion_matrix(y_true, y_pred).tolist()

    exp = PATHS.MODELS / exp_name
    exp.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), exp/"final.pt")
    json.dump({"test_loss":test_loss, "test_acc":test_acc}, open(exp/"final_test_metrics.json","w"), indent=2)
    json.dump(rep, open(exp/"classification_report.json","w"), indent=2)
    json.dump({"confusion_matrix":cm, "classes":classes}, open(exp/"confusion_matrix.json","w"), indent=2)
    print("[FINAL TEST]", {"loss":test_loss, "acc":test_acc})

def parse_grid(grid_str: str):
    # es: "lr=1e-3,3e-4 batch=32,64 dropout=0.2,0.4 epochs=12"
    out = {}
    for token in grid_str.split():
        k, v = token.split("=")
        vals = v.split(",")
        out[k] = [float(x) if any(c in x for c in ".e") else int(x) for x in vals]
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="baseline_small")
    ap.add_argument("--model_id", type=str, choices=["small","medium","large", "tiny"], default="small")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--do_arch_cv", type=bool, default=False)
    ap.add_argument("--do_grid_cv", type=bool, default=False)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--grid", type=str, default="")
    ap.add_argument("--final_eval", type=bool, default=False)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--arch", type=str, default=None)

    args = ap.parse_args()

    cfg = Settings(epochs=args.epochs, batch=args.batch, lr=args.lr)
    if args.do_arch_cv:
        arch_cv(args.k, cfg)
    elif args.do_grid_cv:
        grid = parse_grid(args.grid) if args.grid else {"lr":[1e-3,3e-4], "batch":[32,64], "dropout":[0.2,0.4], "epochs":[12]}
        grid_cv(args.k, grid, cfg)
    elif args.final_eval:
        retrain_and_eval(cfg, exp_name="final_best")
    else:
        # semplice train su train/val e test a fine (per baseline rapida)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, val_loader, test_loader = make_loaders(cfg)
        model, hist, _ = fit_once(args.model_id, {"num_classes":3,"lr":cfg.lr}, train_loader, val_loader, device, epochs=cfg.epochs, patience=cfg.patience)
        # salva log + test
        out = PATHS.MODELS / args.exp
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out/"model.pt")
        
        pd.DataFrame(hist).to_csv(out/"training_log.csv", index=False)
        loss, acc = one_epoch(model, test_loader, nn.CrossEntropyLoss(), None, device)
        js.dump({"test_loss":loss, "test_acc":acc}, open(out/"test_metrics.json","w"), indent=2)
        print("[BASELINE TEST]", {"loss":loss, "acc":acc})
