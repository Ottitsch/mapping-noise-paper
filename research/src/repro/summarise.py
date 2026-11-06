import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2] / "results" / "repro"
rows = []
for d in sorted(R.glob("*/")):
    if not (d / "metrics.json").exists():
        continue
    m = json.loads((d / "metrics.json").read_text())
    t = pd.read_csv(d / "test_metrics.csv")
    for r in t.itertuples():
        rows.append(dict(run=d.name, data=m["data"], train=m["train_cond"], test=r.cond, mae=r.mae, rmse=r.rmse,
                         minutes=m["seconds_total"] / 60))
df = pd.DataFrame(rows)
df.to_csv(R / "runs.csv", index=False)
df["group"] = df.run.str.replace(r"(_s\d|_heidfold\d)$", "", regex=True)
g = df.groupby(["group", "test"]).agg(n=("mae", "size"), mae=("mae", "mean"), mae_sd=("mae", "std"),
                                      rmse=("rmse", "mean"), minutes=("minutes", "mean"))
g.to_csv(R / "summary.csv")
print(g.to_string())
