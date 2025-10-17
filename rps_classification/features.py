from dataclasses import dataclass
from typing import Dict


@dataclass
class FeatureReport:
    class_counts: Dict[str, int]
    img_size_used: int


# Extend here if you want to compute per-channel means/stds and persist to YAML.