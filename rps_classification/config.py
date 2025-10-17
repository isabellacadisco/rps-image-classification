from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


@dataclass
class TrainConfig:
    data_dir: str = "./data/processed"
    out_dir: str = "./models"
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    img_size: int = 128
    model: str = "simple_cnn" # switchable in train.py
    seed: int = 42
    mixed_precision: bool = True


    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            yaml.safe_dump(asdict(self), f)


    @staticmethod
    def load(path: str | Path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return TrainConfig(**data)