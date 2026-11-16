"""Read-only Modal discovery for the real-hair validation experiment.

Separate Modal app and isolated experiment namespace. This module only READS the
existing project volume; it never writes checkpoints, datasets or result files.
No environment variables, credentials or secrets are printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "dm-ahr-real-hair-validation"
VOLUME_NAME = "dmahr-vit-vol"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME)  # existing volume; no create_if_missing

base_image = modal.Image.debian_slim(python_version="3.11").env({"PYTHONUNBUFFERED": "1"})
torch_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==1.26.4")
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)

CKPT_EXTS = (".pth", ".pt", ".ckpt", ".safetensors")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(image=base_image, cpu=4, memory=8192, timeout=3600, volumes={"/vol": vol})
def discover_volumes(hash_checkpoints: bool = True) -> dict:
    """Walk the mounted project volume for checkpoint files and report sizes/hashes."""
    vol.reload()
    root = Path("/vol")
    found = []
    top = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel_dir = str(Path(dirpath).relative_to(root))
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        if depth <= 2:
            top.append({"dir": rel_dir, "files": len(filenames), "subdirs": len(dirnames)})
        for name in filenames:
            if name.lower().endswith(CKPT_EXTS):
                p = Path(dirpath) / name
                try:
                    st = p.stat()
                    rec = {"path": str(p), "bytes": st.st_size,
                           "sha256": _sha256(p) if hash_checkpoints else None}
                except Exception as exc:  # noqa: BLE001
                    rec = {"path": str(p), "error": repr(exc)}
                found.append(rec)
    found.sort(key=lambda r: r["path"])
    result = {"volume": VOLUME_NAME, "root": str(root),
              "n_checkpoint_files": len(found), "checkpoints": found,
              "top_directories": sorted(top, key=lambda r: r["dir"])}
    out = root / "artifacts" / "real_hair_validation" / "discovery_output.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"DISCOVERY WRITTEN {out} with {len(found)} checkpoint files", flush=True)
    return result


@app.function(image=torch_image, gpu="A100-80GB", timeout=600, volumes={"/vol": vol})
def probe_a100_80gb() -> dict:
    import torch
    return {"gpu_requested": "A100-80GB", "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


@app.function(image=torch_image, gpu="A100-40GB", timeout=600, volumes={"/vol": vol})
def probe_a100_40gb() -> dict:
    import torch
    return {"gpu_requested": "A100-40GB", "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


@app.function(image=torch_image, gpu="L4", timeout=600, volumes={"/vol": vol})
def probe_l4() -> dict:
    import torch
    return {"gpu_requested": "L4", "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


@app.function(image=torch_image, gpu="A10G", timeout=600, volumes={"/vol": vol})
def probe_a10g() -> dict:
    import torch
    return {"gpu_requested": "A10G", "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


@app.local_entrypoint()
def probe_primary_gpu():
    """Verify the GPU we would actually use can be allocated."""
    print(probe_a100_80gb.remote(), flush=True)
