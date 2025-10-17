import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd


from rps_classification.modeling.train import build_model




def predict_folder(images_dir: Path, ckpt: Path, out_csv: Path, img_size: int = 128, class_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if class_names is None:
        class_names = ["paper", "rock", "scissors"]  # adjust if needed

    model = build_model("simple_cnn", num_classes=len(class_names))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)


    rows = []
    with torch.no_grad():
        for fp in files:
            img = Image.open(fp).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            pred_idx = int(torch.argmax(logits, dim=1))
            rows.append({
                "file": fp.name,
                "pred": class_names[pred_idx],
                **{f"p_{name}": probs[i] for i, name in enumerate(class_names)}
            })


    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="./reports/predictions.csv")
    p.add_argument("--img-size", type=int, default=128)
    args = p.parse_args()


    predict_folder(Path(args.images), Path(args.ckpt), Path(args.out), img_size=args.img_size)