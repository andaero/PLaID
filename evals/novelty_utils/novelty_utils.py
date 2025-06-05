"""
This file depends on and heavily modifies code from Meta's flowllm repository, which is MIT-licensed.
The original license is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint



CrysArrayListType = list[dict[str, np.ndarray]]

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")



def save_metrics_only_overwrite_newly_computed(
    path: Path, metrics: dict[str, float]
) -> None:
    # only overwrite metrics computed in the new run.
    if Path(path).exists():
        with open(path, "r") as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(metrics)
            else:
                with open(path, "w") as f:
                    json.dump(metrics, f)
        if isinstance(written_metrics, dict):
            with open(path, "w") as f:
                json.dump(written_metrics, f)
    else:
        with open(path, "w") as f:
            json.dump(metrics, f)
