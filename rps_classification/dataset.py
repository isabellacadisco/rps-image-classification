import argparse, csv, shutil, uuid, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from .config import PATHS, Settings

IMG_EXTS = {".png", ".jpg", ".jpeg"}

def _iter_imgs(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def make_split(src: Path, out: Path, val: float, test: float, seed: int):
    # reset processed
    for split in ["train","val","test"]:
        d = out / split
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    (out / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

    rows = []
    for cls in classes:
        imgs = sorted(_iter_imgs(src / cls))
        train_imgs, temp = train_test_split(imgs, test_size=val+test, random_state=seed, shuffle=True)
        rel = test / (val + test)
        val_imgs, test_imgs = train_test_split(temp, test_size=rel, random_state=seed, shuffle=True)

        for split, arr in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            (out / split / cls).mkdir(parents=True, exist_ok=True)
            for p in arr:
                dst = out / split / cls / p.name
                if dst.exists(): 
                    stem, suf = dst.stem, dst.suffix
                    dst = dst.with_name(f"{stem}__{uuid.uuid4().hex[:8]}{suf}")
                shutil.copy2(p, dst)

                # fix percorso relativo per manifest
                root_abs = PATHS.ROOT.resolve()  # root del progetto
                dst_abs = dst.resolve()          # percorso assoluto dell'immagine copiata
                try:
                    rel = os.path.relpath(dst_abs, root_abs)
                except ValueError:
                    # es: drive diverso su Windows -> fallback assoluto
                    rel = str(dst_abs)
                rel = rel.replace("\\", "/")  # normalizza separatori per compatibilità
                rows.append((split, rel, cls))

    PATHS.REFS.mkdir(parents=True, exist_ok=True)
    for split in ["train","val","test","all"]:
        with open(PATHS.REFS / f"manifest_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["split","path","label"])
            for s, p, l in rows:
                if split == "all" or s == split: w.writerow([s,p,l])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=str(PATHS.DATA_RAW))
    ap.add_argument("--out", type=str, default=str(PATHS.DATA_PROC))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make_split(Path(args.src), Path(args.out), args.val, args.test, args.seed)
    print("Split creato in", args.out)
