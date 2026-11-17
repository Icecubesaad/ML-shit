"""Step 1 (free): train/val L1 history from the recorded DM-AHR training JSONs.

Reads only local logs. Reports the last-20k validation-L1 trend and the train/val
gap. Never reads the test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "artifacts" / "logs"
DIAG = ROOT / "artifacts" / "diagnosis"
OUT = ROOT / "artifacts" / "logs" / "iteration_l1_history.csv"


def load_small(name: str) -> dict:
    p = LOGS / name
    return json.loads(p.read_text()) if p.exists() else {}


def slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    a = load_small("dmahr_train.json")
    b = load_small("dmahr_v2_train.json")
    hist_path = DIAG / "dmahr_v2_train_log.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    rows = []

    for label, d in (("dmahr_train", a), ("dmahr_v2_train", b)):
        rows.append({"log": label, "field": "summary", "iterations": d.get("iterations"),
                     "best_val_l1": d.get("best_val"), "seconds": d.get("seconds"),
                     "has_history": "history" in d})

    h = hist.get("history", [])
    if h:
        df = pd.DataFrame(h).sort_values("iteration")
        last = df[df["iteration"] > df["iteration"].max() - 20000]
        last20 = last.tail(20)
        val = last20["val_l1"].to_numpy(dtype=float)
        gap = (last20["val_l1"] - last20["train_l1"]).to_numpy(dtype=float)
        rows.append({"log": "dmahr_v2_train_log", "field": "last_20k_val_l1_slope_per_1k",
                     "value": slope(val) * 1000, "n_points": len(val),
                     "first": float(val[0]), "last": float(val[-1]),
                     "min": float(val.min()), "min_iteration": int(last20["iteration"].iloc[int(val.argmin())])})
        rows.append({"log": "dmahr_v2_train_log", "field": "last_10k_val_l1_slope_per_1k",
                     "value": slope(val[-10:]) * 1000, "n_points": 10,
                     "first": float(val[-10]), "last": float(val[-1])})
        rows.append({"log": "dmahr_v2_train_log", "field": "train_val_gap_mean_last_20k",
                     "value": float(gap.mean()), "n_points": len(gap),
                     "first": float(gap[0]), "last": float(gap[-1])})
        # Full-history trend (all recorded points, 1k..20k).
        vall = df["val_l1"].to_numpy(dtype=float)
        rows.append({"log": "dmahr_v2_train_log", "field": "full_val_l1_slope_per_1k",
                     "value": slope(vall) * 1000, "n_points": len(vall),
                     "first": float(vall[0]), "last": float(vall[-1]),
                     "min": float(vall.min()), "min_iteration": int(df["iteration"].iloc[int(vall.argmin())])})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
