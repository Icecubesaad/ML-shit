"""Estimated Modal spend for the client-asks task, from captured job logs.

Each `_run` wrapper prints "EXIT <code> in <seconds>s". Container start-up and image pulls are not in
that number, so a fixed overhead per job is added. Prices are the per-second rates on
modal.com/pricing fetched 2026-11-13 (A10G is not listed; the A10 rate is used). This is an
estimate, not an invoice.

usage: 66_spend_log.py --logs artifacts/logs/modal_runs
Each log file name is "<job>__<gpu>.log" and may contain several EXIT lines.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PRICE_PER_SECOND = {"A100-80GB": 0.000694, "A100-40GB": 0.000583, "L4": 0.000222, "A10G": 0.000306}
OVERHEAD_SECONDS = 90


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="artifacts/logs/modal_runs")
    ap.add_argument("--out", default="reports/spend_log.csv")
    ap.add_argument("--registry", default=None)
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.logs).glob("*.log")):
        job, _, gpu = f.stem.partition("__")
        text = f.read_text(errors="ignore")
        for i, m in enumerate(re.finditer(r"EXIT (\d+) in ([0-9.]+)s", text)):
            secs = float(m.group(2)) + OVERHEAD_SECONDS
            rows.append({"job": job, "gpu": gpu, "run": i, "exit_code": int(m.group(1)),
                         "seconds_with_overhead": round(secs, 1),
                         "usd_per_second": PRICE_PER_SECOND[gpu],
                         "est_usd": round(secs * PRICE_PER_SECOND[gpu], 3)})
    for r in rows:
        r["scope"] = "this session (job logs)"
    if args.registry and Path(args.registry).exists():
        import json  # noqa: PLC0415

        # Earlier sessions: registry rows carry GPU names only sometimes; an unnamed "cuda" row is priced
        # at the A100-80GB rate, so the subtotal is an upper bound, not an invoice.
        for i, line in enumerate(Path(args.registry).read_text().splitlines()):
            d = json.loads(line)
            if d.get("gpu") is None or str(d.get("timestamp", "")) >= "2026-11-14T00":
                continue  # this session's jobs are counted from their job logs
            g = str(d["gpu"])
            rate = PRICE_PER_SECOND["A100-40GB"] if "40GB" in g else PRICE_PER_SECOND["A100-80GB"]
            secs = float(d.get("seconds", 0)) + OVERHEAD_SECONDS
            rows.append({"job": d.get("job", "?"), "gpu": g, "run": i, "exit_code": 0,
                         "seconds_with_overhead": round(secs, 1), "usd_per_second": rate,
                         "est_usd": round(secs * rate, 3), "scope": "earlier sessions (run registry, upper bound)"})
    df = pd.DataFrame(rows)
    df["running_total_usd"] = df["est_usd"].cumsum().round(3)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print("total estimated $", round(df["est_usd"].sum(), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
