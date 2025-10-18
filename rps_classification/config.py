from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_RAW: Path = ROOT / "data" / "raw"      # dati copiati da data/data_rps
    DATA_PROC: Path = ROOT / "data" / "processed"    # train, val, test split 
    REFS: Path = ROOT / "references" / "splits"
    MODELS: Path = ROOT / "models"
    REPORTS: Path = ROOT / "reports"
    FIGURES: Path = REPORTS / "figures"

@dataclass
class Settings:
    seed: int = 42
    img_size: int = 192    # cambi qui o nella CLI
    num_classes: int = 3
    lr: float = 1e-3
    batch: int = 64
    epochs: int = 20
    patience: int = 5
    num_workers: int = 2
    pin_memory: bool = True

PATHS = Paths()
for d in [PATHS.REFS, PATHS.MODELS, PATHS.FIGURES]:
    d.mkdir(parents=True, exist_ok=True)
