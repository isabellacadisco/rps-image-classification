import argparse, json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from ..config import PATHS, Settings
from .train import build_model

def predict(imgs, model_path, model_id="medium", img_size=192):
    classes = [l.strip() for l in (PATHS.DATA_PROC/"classes.txt").read_text().splitlines()]
    model = build_model(model_id, num_classes=len(classes), dropout=0.0)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    out = []
    with torch.no_grad():
        for p in imgs:
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0)
            y = model(x).softmax(1)[0]
            conf, idx = float(y.max().item()), int(y.argmax().item())
            out.append({"path": str(p), "pred": classes[idx], "conf": conf})
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="medium")
    ap.add_argument("--imgs", type=str, nargs="+", required=True)
    args = ap.parse_args()
    res = predict([Path(p) for p in args.imgs], Path(args.model_path), args.model_id)
    for r in res:
        print(f"{r['path']} -> {r['pred']} ({r['conf']:.3f})")
