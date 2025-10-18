from pathlib import Path
import json, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from .config import PATHS

def plot_history(csv_log: Path, out_prefix: Path):
    df = pd.read_csv(csv_log)
    plt.figure(); plt.plot(df["epoch"], df["train_loss"], label="train"); plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(); plt.legend(); plt.savefig(out_prefix.with_name(out_prefix.stem+"_loss.png"), dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); plt.plot(df["epoch"], df["train_acc"], label="train"); plt.plot(df["epoch"], df["val_acc"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(); plt.legend(); plt.savefig(out_prefix.with_name(out_prefix.stem+"_acc.png"), dpi=150, bbox_inches="tight"); plt.close()

def plot_cm(cm_json: Path, out_png: Path):
    d = json.load(open(cm_json)); cm = d["confusion_matrix"]; classes = d["classes"]
    plt.figure(figsize=(4,4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

if __name__ == "__main__":
    exp_dir = PATHS.MODELS / "baseline_small"
    if (exp_dir / "training_log.csv").exists():
        plot_history(exp_dir / "training_log.csv", PATHS.FIGURES / "baseline_small")

    # TODO: non farlo così manuale, soluzione sporca 
    exp_dir = PATHS.MODELS / "best_model_large"
    if (exp_dir / "training_log.csv").exists():
        plot_history(exp_dir / "training_log.csv", PATHS.FIGURES / "best_model_large")

    final_cm = PATHS.MODELS / "final_best" / "confusion_matrix.json"
    if final_cm.exists():
        plot_cm(final_cm, PATHS.FIGURES / "final_confusion_matrix.png")
