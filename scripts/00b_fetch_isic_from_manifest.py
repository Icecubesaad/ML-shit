"""Fetch ISIC originals from the frozen manifest's S3 URLs, with hash verification.

Used when the ISIC API listing endpoint is unreachable from the compute environment:
the frozen ``data/manifests/data_manifest.csv`` already carries every image_id and its
SHA-256, so no metadata API is needed. Files that already exist and hash-correct are
skipped, making the fetch resumable. Only the 10,015-image C0 cohort in the manifest is
downloaded; no extra ISIC collection image is fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.paths import isic_images_dir, manifest_dir, raw_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
S3 = "https://isic-archive.s3.amazonaws.com/images/{iid}.jpg"


def fetch_one(iid: str, want: str, out_dir: Path) -> dict:
    dest = out_dir / f"{iid}.jpg"
    if dest.exists():
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if not want or got == want:
            return {"image_id": iid, "status": "cached", "sha256": got}
    last = None
    for attempt in range(4):
        try:
            r = requests.get(S3.format(iid=iid), timeout=120)
            r.raise_for_status()
            blob = r.content
            got = hashlib.sha256(blob).hexdigest()
            if want and got != want:
                return {"image_id": iid, "status": "hash_mismatch",
                        "sha256": got, "expected": want}
            dest.write_bytes(blob)
            return {"image_id": iid, "status": "downloaded", "sha256": got,
                    "bytes": len(blob)}
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            time.sleep(2 * (attempt + 1))
    return {"image_id": iid, "status": "failed", "error": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_csv(manifest_dir() / "data_manifest.csv")
    rows = man[["image_id", "sha256"]].drop_duplicates("image_id")
    if args.limit:
        rows = rows.head(args.limit)
    out_dir = isic_images_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("fetching %d manifest images from S3 into %s", len(rows), out_dir)

    with timed("fetch_isic_from_manifest", log) as tmr:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            recs = list(ex.map(lambda r: fetch_one(r.image_id, r.sha256, out_dir),
                               rows.itertuples()))
    rec_path = raw_dir() / "isic_212" / "records.jsonl"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    with rec_path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    counts: dict[str, int] = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    bad = [r for r in recs if r["status"] in ("failed", "hash_mismatch")]
    if bad:
        (raw_dir() / "isic_212" / "fetch_failures.json").write_text(json.dumps(bad, indent=2))
    record_run("fetch_isic_from_manifest", gpu=None, seconds=tmr["seconds"],
               n=len(recs), **counts)
    print(json.dumps({"n": len(recs), "counts": counts, "seconds": tmr["seconds"]}, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
