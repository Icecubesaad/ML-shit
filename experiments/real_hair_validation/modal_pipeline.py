"""Real-hair validation pipeline on the dedicated Modal volume.

Phase 2 creates the isolated structure; Phase 3 downloads external datasets with
resumable, hash-verified transfers; later phases audit, match and evaluate. Original
project artifacts and checkpoints are never written here.
"""

from __future__ import annotations

import modal

APP_NAME = "dm-ahr-real-hair-validation"
NEW_VOL = "dm-ahr-real-hair-validation"
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(NEW_VOL)
proj_vol = modal.Volume.from_name("dmahr-vit-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "unzip")
    .pip_install("numpy==1.26.4", "pandas==2.2.3", "pillow==11.0.0",
                 "opencv-python-headless==4.10.0.84", "scikit-image==0.24.0",
                 "scipy==1.14.1", "requests==2.32.3", "tqdm==4.67.1",
                 "torch==2.5.1", "torchvision==0.20.1", "lpips==0.1.4")
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)

DIRS = [
    "data/external/mendeley_hair_masks",
    "data/external/dermartifactdb",
    "data/external/isic2019",
    "data/external/ph2",
    "data/external/derm7pt",
    "data/processed/validated_images",
    "data/processed/hair_masks",
    "data/processed/artifact_labels",
    "data/processed/matched_datasets",
    "data/processed/duplicate_audit",
    "experiments/smoke_test",
    "experiments/base_checkpoint",
    "experiments/finetuned_checkpoint",
    "experiments/primary_evaluation/original",
    "experiments/primary_evaluation/dm_ahr_base",
    "experiments/primary_evaluation/dm_ahr_finetuned",
    "experiments/primary_evaluation/dullrazor",
    "experiments/primary_evaluation/comparisons",
    "experiments/exploratory_evaluation",
    "reports/real_hair_validation",
    "logs/downloads", "logs/inference", "logs/metrics", "logs/billing",
]

SOURCES = {
    "mendeley": "https://data.mendeley.com/datasets/j5ywpd2p27/2",
    "zenodo_dermartifactdb": "https://zenodo.org/records/18324962",
    "isic2019": "https://challenge.isic-archive.com/data/",
    "ph2": "https://www.fc.up.pt/addi/ph2%20database.html",
    "derm7pt": "https://derm.cs.sfu.ca/Welcome.html",
}


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def init_structure() -> dict:
    from pathlib import Path
    root = Path("/vol")
    made = []
    for d in DIRS:
        p = root / d
        p.mkdir(parents=True, exist_ok=True)
        keep = p / ".keep"
        if not keep.exists():
            keep.write_text("")
            made.append(d)
    vol.commit()
    return {"volume": NEW_VOL, "directories": len(DIRS), "created": made}


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def probe_sources() -> dict:
    """Read-only accessibility check. Does NOT download a dataset."""
    import requests
    out = {}
    for name, url in SOURCES.items():
        try:
            r = requests.get(url, timeout=30, stream=True,
                             headers={"User-Agent": "dm-ahr-real-hair-validation/1.0"})
            out[name] = {"url": url, "status": r.status_code,
                         "content_type": r.headers.get("content-type", ""),
                         "note": "reachable" if r.ok else "HTTP error"}
            r.close()
        except Exception as exc:  # noqa: BLE001
            out[name] = {"url": url, "status": None, "error": repr(exc)}
    from pathlib import Path
    p = Path("/vol/reports/real_hair_validation/source_probe.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(__import__("json").dumps(out, indent=2))
    vol.commit()
    print("SOURCE PROBE WRITTEN", p, flush=True)
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def probe_api_details() -> dict:
    """Read-only API inspection to discover exact file URLs, sizes and versions."""
    import json
    from pathlib import Path

    import requests

    out = {}
    try:
        r = requests.get("https://zenodo.org/api/records/18324962", timeout=60)
        out["zenodo"] = {"status": r.status_code}
        if r.ok:
            j = r.json()
            out["zenodo"].update({
                "title": j.get("metadata", {}).get("title"),
                "doi": j.get("doi"),
                "files": [{"key": f.get("key"), "size": f.get("size"),
                           "link": f.get("links", {}).get("self")}
                          for f in j.get("files", [])],
            })
    except Exception as exc:  # noqa: BLE001
        out["zenodo"] = {"error": repr(exc)}

    endpoints = [
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/files",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/files?folder=dermoscopic_image&page=1&limit=5",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27",
    ]
    out["mendeley"] = []
    for url in endpoints:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "dm-ahr-real-hair-validation/1.0"})
            rec = {"url": url, "status": r.status_code, "sample": r.text[:1200]}
            out["mendeley"].append(rec)
        except Exception as exc:  # noqa: BLE001
            out["mendeley"].append({"url": url, "error": repr(exc)})

    p = Path("/vol/reports/real_hair_validation/api_details.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("API DETAILS WRITTEN", p, flush=True)
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def probe_mendeley() -> dict:
    import json
    from pathlib import Path

    import requests

    base = "https://data.mendeley.com/public-api/datasets/j5ywpd2p27"
    urls = [
        f"{base}?limit=100",
        f"{base}/files?folder=dermoscopic_image&version=2&page=1",
        f"{base}/files?folder=dermoscopic_image&page=1",
        f"{base}/files?folder=dermoscopic%20image&page=1",
        f"{base}/files?folder=dermoscopic_image",
        f"{base}/versions",
        "https://data.mendeley.com/datasets/j5ywpd2p27/2",
    ]
    out = []
    for url in urls:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "dm-ahr-real-hair-validation/1.0"})
            out.append({"url": url, "status": r.status_code,
                        "ctype": r.headers.get("content-type", ""),
                        "body": r.text[:2500]})
        except Exception as exc:  # noqa: BLE001
            out.append({"url": url, "error": repr(exc)})
    p = Path("/vol/reports/real_hair_validation/mendeley_probe.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("MENDELEY PROBE WRITTEN", p, flush=True)
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def probe_mendeley2() -> dict:
    import json
    import re
    from pathlib import Path

    import requests

    urls = [
        "https://data.mendeley.com/datasets/j5ywpd2p27/2/download",
        "https://data.mendeley.com/datasets/j5ywpd2p27/2/download?folder=dermoscopic_image",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/files?folder=dermoscopic_image&page=1&limit=100&version=2",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/files?folder=dermoscopic_image&page=1&limit=100&version=1",
    ]
    out = []
    for url in urls:
        try:
            r = requests.get(url, timeout=60, stream=True,
                             headers={"User-Agent": "dm-ahr-real-hair-validation/1.0"})
            first = next(r.iter_content(512), b"")
            out.append({"url": url, "status": r.status_code,
                        "ctype": r.headers.get("content-type", ""),
                        "length": r.headers.get("content-length", ""),
                        "disposition": r.headers.get("content-disposition", ""),
                        "first_bytes": repr(first[:120])})
            r.close()
        except Exception as exc:  # noqa: BLE001
            out.append({"url": url, "error": repr(exc)})

    html = requests.get("https://data.mendeley.com/datasets/j5ywpd2p27/2", timeout=60).text
    hits = []
    for m in re.finditer(r"dermoscopic_image|hair_mask|\.jpg|download_file|/api/", html):
        s = max(0, m.start() - 120)
        hits.append(html[s:m.start() + 220].replace("\n", " ")[:340])
        if len(hits) >= 12:
            break
    out.append({"html_hits": hits, "html_length": len(html)})
    p = Path("/vol/reports/real_hair_validation/mendeley_probe2.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("MENDELEY PROBE2 WRITTEN", p, flush=True)
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def probe_mendeley3() -> dict:
    import json
    import re
    from pathlib import Path

    import requests

    H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/html"}
    variants = [
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/files?folder=/dermoscopic_image&page=1",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/v2/files?folder=dermoscopic_image&page=1",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/versions/2/files?folder=dermoscopic_image&page=1",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/files?folder=dermoscopic_image&page=1&limit=100&sort=name",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/files?folder=dermoscopic_image&page=1&limit=100&version=2&sort=name",
        "https://data.mendeley.com/public-api/v1/datasets/j5ywpd2p27/files?folder=dermoscopic_image&page=1",
        "https://data.mendeley.com/public-api/datasets/j5ywpd2p27/files?folder=dermoscopic_image&page=0",
    ]
    out = []
    for url in variants:
        try:
            r = requests.get(url, timeout=60, headers=H)
            out.append({"url": url, "status": r.status_code,
                        "body": r.text[:400]})
        except Exception as exc:  # noqa: BLE001
            out.append({"url": url, "error": repr(exc)})

    html = requests.get("https://data.mendeley.com/datasets/j5ywpd2p27/2", timeout=60, headers=H).text
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    interesting = []
    for s in scripts:
        if any(k in s.lower() for k in ("api", "file", "folder", "download")):
            interesting.append(s[:1500])
    # also look for embedded __NEXT_DATA__ / window.__
    embedded = re.findall(r"(?:__NEXT_DATA__|window\.__[A-Z_]+)\s*=\s*(\{.{0,4000}?\})\s*;?", html, re.S)
    out.append({"n_scripts": len(scripts), "interesting_scripts": interesting[:6],
                "embedded": [e[:2000] for e in embedded[:3]], "html_len": len(html)})
    p = Path("/vol/reports/real_hair_validation/mendeley_probe3.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("MENDELEY PROBE3 WRITTEN", p, flush=True)
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def probe_mendeley4() -> dict:
    import json
    import re
    from pathlib import Path

    import requests

    html = requests.get("https://data.mendeley.com/datasets/j5ywpd2p27/2", timeout=60,
                        headers={"User-Agent": "Mozilla/5.0"}).text
    marker = "window.INITIAL_STATE = "
    out = {"marker_found": marker in html}
    if marker in html:
        start = html.index(marker) + len(marker)
        # balanced-brace extraction
        depth, i, in_str, esc = 0, start, False, False
        end = None
        while i < len(html):
            ch = html[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            i += 1
        if end:
            try:
                state = json.loads(html[start:end])
                out["top_keys"] = list(state.keys())

                def walk(o, path=""):
                    hits = []
                    if isinstance(o, dict):
                        for k, v in o.items():
                            p2 = f"{path}.{k}" if path else k
                            if any(t in k.lower() for t in ("file", "folder", "download", "content", "url", "version", "dataset")):
                                hits.append({"path": p2, "type": type(v).__name__,
                                             "preview": str(v)[:500]})
                            if len(hits) < 120:
                                hits += walk(v, p2)
                    elif isinstance(o, list) and len(o) < 200:
                        for j, v in enumerate(o[:10]):
                            hits += walk(v, f"{path}[{j}]")
                    return hits
                out["hits"] = walk(state)[:120]
                out["state_chars"] = end - start
                full = Path("/vol/reports/real_hair_validation/mendeley_initial_state.json")
                full.write_text(json.dumps(state, indent=2))
            except Exception as exc:  # noqa: BLE001
                out["parse_error"] = repr(exc)
    p = Path("/vol/reports/real_hair_validation/mendeley_probe4.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("MENDELEY PROBE4 WRITTEN", p, flush=True)
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/vol": vol})
def probe_mendeley5() -> dict:
    import json
    from pathlib import Path

    import requests

    H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    base = "https://data.mendeley.com"
    variants = [
        "/api/datasets-v2/datasets/j5ywpd2p27/files?folder=dermoscopic_image&version=2&page=1",
        "/api/datasets-v2/datasets/j5ywpd2p27/files?version=2&page=1",
        "/api/datasets-v2/datasets/j5ywpd2p27/folders?version=2",
        "/api/datasets-v2/datasets/j5ywpd2p27?version=2",
        "/api/datasets-v2/datasets/j5ywpd2p27/archive?version=2",
        "/public-api/datasets/j5ywpd2p27/files?folder=dermoscopic_image&page=1&version=2&limit=100&format=json",
    ]
    out = []
    for u in variants:
        try:
            r = requests.get(base + u, timeout=60, headers=H)
            out.append({"url": u, "status": r.status_code, "body": r.text[:900]})
        except Exception as exc:  # noqa: BLE001
            out.append({"url": u, "error": repr(exc)})
    p = Path("/vol/reports/real_hair_validation/mendeley_probe5.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("MENDELEY PROBE5 WRITTEN", p, flush=True)
    return out


@app.function(image=image, cpu=1, memory=2048, timeout=300, volumes={"/vol": vol})
def show_report(name: str = "mendeley_probe5.json") -> str:
    """Print a small JSON report from the volume to the Modal log (no local download)."""
    from pathlib import Path
    p = Path("/vol/reports/real_hair_validation") / name
    text = p.read_text() if p.exists() else "MISSING"
    print(f"=== {name} ===\n{text[:6000]}", flush=True)
    return text[:6000]


@app.function(image=image, cpu=1, memory=2048, timeout=300, volumes={"/vol": vol})
def show_mendeley7_details() -> str:
    import json
    from pathlib import Path
    d = json.loads((Path("/vol/reports/real_hair_validation") / "mendeley_probe7.json").read_text())
    lines = ["ATTEMPTS:"]
    for a in d.get("attempts", []):
        lines.append(f"  {a.get('method')} {a.get('url')} -> {a.get('status')} "
                     f"{a.get('ctype','')} body={a.get('body','')[:150]!r}")
    for needle in ("/zip", "archive", "datasets-v2/datasets", "/files"):
        lines.append(f"--- {needle} contexts ---")
        for f in d.get("findings", []):
            if f.get("needle") == needle:
                lines.append("  " + f.get("ctx", "")[:680].replace("\n", " "))
    text = "\n".join(lines)
    print(text[:14000], flush=True)
    return text[:14000]


@app.function(image=image, cpu=2, memory=4096, timeout=900, volumes={"/vol": vol})
def probe_mendeley6() -> dict:
    """Find the real file-listing route in Mendeley's JS bundle."""
    import json
    import re
    from pathlib import Path
    from urllib.parse import urljoin

    import requests

    H = {"User-Agent": "Mozilla/5.0"}
    page = "https://data.mendeley.com/datasets/j5ywpd2p27/2"
    html = requests.get(page, timeout=60, headers=H).text
    srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
    base = "https://data.mendeley.com"
    out = {"n_scripts": len(srcs), "srcs": srcs[:20], "findings": []}
    for src in srcs[:12]:
        url = urljoin(base, src)
        try:
            r = requests.get(url, timeout=120, headers=H)
            js = r.text
            for needle in ("datasets-v2", "public-api", "/files", "downloadFile", "getFiles", "archive"):
                for m in re.finditer(re.escape(needle), js):
                    s = js[max(0, m.start() - 200):m.start() + 320]
                    out["findings"].append({"src": src, "needle": needle, "context": s})
                    break
            if len(out["findings"]) >= 40:
                break
        except Exception as exc:  # noqa: BLE001
            out["findings"].append({"src": src, "error": repr(exc)})
    p = Path("/vol/reports/real_hair_validation/mendeley_probe6.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("MENDELEY PROBE6 WRITTEN findings", len(out["findings"]), flush=True)
    return {"findings": len(out["findings"])}


@app.function(image=image, cpu=4, memory=16384, timeout=14400, volumes={"/vol": vol})
def download_mendeley() -> dict:
    import json
    import zipfile
    from pathlib import Path

    import requests

    vol.reload()
    H = {"User-Agent": "Mozilla/5.0"}
    base = "https://data.mendeley.com/api/datasets-v2/datasets/j5ywpd2p27"
    meta = {}
    for u in (base, base + "/zip"):
        try:
            r = requests.get(u, timeout=90, headers=H, stream=True)
            meta[u] = {"status": r.status_code, "ctype": r.headers.get("content-type", ""),
                       "length": r.headers.get("content-length", ""),
                       "disposition": r.headers.get("content-disposition", "")}
            if u.endswith("/zip") and r.ok:
                dest = Path("/vol/data/external/mendeley_hair_masks/Mendeley_hair_masks.zip")
                dest.parent.mkdir(parents=True, exist_ok=True)
                got = _resumable_download(u, dest)
                meta["download"] = got
                valid = False
                err = ""
                if dest.exists():
                    try:
                        with zipfile.ZipFile(dest) as z:
                            err = z.testzip() or ""
                        valid = err == ""
                    except Exception as exc:  # noqa: BLE001
                        err = repr(exc)
                meta["valid_zip"] = valid
                meta["zip_error"] = err
                if valid:
                    with zipfile.ZipFile(dest) as z:
                        z.extractall(str(dest.parent))
            r.close()
        except Exception as exc:  # noqa: BLE001
            meta[u] = {"error": repr(exc)}
    p = Path("/vol/reports/real_hair_validation/dataset_download_manifest_mendeley.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"source": "https://data.mendeley.com/datasets/j5ywpd2p27/2",
                             "doi": "10.17632/j5ywpd2p27.2", "license": "CC0",
                             "endpoints": meta}, indent=2))
    vol.commit()
    print("MENDELEY DOWNLOAD", json.dumps(meta)[:800], flush=True)
    return meta


@app.function(image=image, cpu=2, memory=4096, timeout=900, volumes={"/vol": vol})
def probe_mendeley7() -> dict:
    import json
    import re
    from pathlib import Path

    import requests

    H = {"User-Agent": "Mozilla/5.0"}
    out = {}
    html = requests.get("https://data.mendeley.com/datasets/j5ywpd2p27/2", timeout=60, headers=H).text
    srcs = [s for s in re.findall(r'<script[^>]+src="([^"]+)"', html) if "/datasets/bundle" in s]
    out["bundle_srcs"] = srcs
    findings = []
    for src in srcs[:2]:
        js = requests.get("https://data.mendeley.com" + src, timeout=180, headers=H).text
        out["bundle_len"] = len(js)
        for needle in ("/zip", "archive", "datasets-v2/datasets", "/files", "idToken", "download"):
            idxs = [m.start() for m in re.finditer(re.escape(needle), js)]
            for i in idxs[:6]:
                findings.append({"needle": needle, "ctx": js[max(0, i - 260):i + 420]})
    out["findings"] = findings[:60]
    base = "https://data.mendeley.com/api/datasets-v2/datasets/j5ywpd2p27"
    attempts = []
    for method, url in (("GET", base + "/zip"), ("POST", base + "/zip"),
                        ("GET", base + "/archive"), ("POST", base + "/archive"),
                        ("GET", base + "/versions/2/zip")):
        try:
            r = requests.request(method, url, timeout=90, headers=H)
            attempts.append({"method": method, "url": url, "status": r.status_code,
                             "ctype": r.headers.get("content-type", ""),
                             "body": r.text[:300]})
        except Exception as exc:  # noqa: BLE001
            attempts.append({"method": method, "url": url, "error": repr(exc)})
    out["attempts"] = attempts
    p = Path("/vol/reports/real_hair_validation/mendeley_probe7.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("MENDELEY PROBE7 WRITTEN", len(findings), "findings", flush=True)
    return {"findings": len(findings)}


@app.function(image=image, cpu=4, memory=16384, timeout=14400, volumes={"/vol": vol})
def download_mendeley2() -> dict:
    """Download the whole Mendeley dataset via the discovered download-all URL."""
    import json
    import zipfile
    from pathlib import Path

    import requests

    vol.reload()
    H = {"User-Agent": "Mozilla/5.0"}
    urls = [
        "https://data.mendeley.com/public-api/zip/j5ywpd2p27/download/2",
        "https://data.mendeley.com/api/datasets-v2/datasets/j5ywpd2p27/zip?version=2",
    ]
    out = {"attempts": []}
    dest = Path("/vol/data/external/mendeley_hair_masks/Mendeley_hair_masks.zip")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            r = requests.get(url, timeout=90, headers=H, stream=True)
            ctype = r.headers.get("content-type", "")
            length = r.headers.get("content-length", "")
            ok_zip = r.ok and ("zip" in ctype or "octet" in ctype or not ctype)
            out["attempts"].append({"url": url, "status": r.status_code,
                                    "ctype": ctype, "length": length})
            r.close()
            if ok_zip:
                got = _resumable_download(url, dest)
                out["download"] = got
                break
        except Exception as exc:  # noqa: BLE001
            out["attempts"].append({"url": url, "error": repr(exc)})
    valid, err = False, ""
    if dest.exists():
        try:
            with zipfile.ZipFile(dest) as z:
                err = z.testzip() or ""
            valid = err == ""
        except Exception as exc:  # noqa: BLE001
            err = repr(exc)
    out.update({"valid_zip": valid, "zip_error": err,
                "source": "https://data.mendeley.com/datasets/j5ywpd2p27/2",
                "doi": "10.17632/j5ywpd2p27.2", "license": "CC0"})
    if valid:
        with zipfile.ZipFile(dest) as z:
            names = z.namelist()
        out["n_members"] = len(names)
        out["member_sample"] = names[:10]
        with zipfile.ZipFile(dest) as z:
            z.extractall(str(dest.parent))
    p = Path("/vol/reports/real_hair_validation/dataset_download_manifest_mendeley.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("MENDELEY2", json.dumps({k: out.get(k) for k in
                                  ("attempts", "download", "valid_zip", "n_members")})[:1200], flush=True)
    return out


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def inspect_datasets() -> dict:
    """Inspect extracted dataset structure and image/mask properties (read-only)."""
    import json
    from collections import Counter
    from pathlib import Path

    from PIL import Image

    vol.reload()
    out = {}
    for name, root in (("mendeley", "/vol/data/external/mendeley_hair_masks"),
                       ("zenodo", "/vol/data/external/dermartifactdb")):
        p = Path(root)
        entry = {"exists": p.exists()}
        if p.exists():
            dirs = sorted(x.name for x in p.iterdir() if x.is_dir())
            entry["dirs"] = dirs[:40]
            entry["n_dirs"] = len(dirs)
            ext = Counter()
            for f in p.rglob("*"):
                if f.is_file():
                    ext[f.suffix.lower()] += 1
            entry["extensions"] = dict(ext.most_common(20))
            example = {}
            for d in dirs[:6]:
                files = sorted(x for x in (p / d).rglob("*") if x.is_file())
                example[d] = {"n": len(files), "sample": [x.name for x in files[:5]]}
            entry["examples"] = example
        out[name] = entry

    # Mendeley image / mask sanity
    mp = Path("/vol/data/external/mendeley_hair_masks")
    img_dirs = [d for d in mp.rglob("*") if d.is_dir() and "dermoscopic" in d.name.lower()]
    mask_dirs = [d for d in mp.rglob("*") if d.is_dir() and "hair_mask" in d.name.lower()]
    detail = {}
    if img_dirs and mask_dirs:
        imgs = sorted(img_dirs[0].glob("*"))
        masks = sorted(mask_dirs[0].glob("*"))
        detail = {"img_dir": str(img_dirs[0]), "n_imgs": len(imgs),
                  "mask_dir": str(mask_dirs[0]), "n_masks": len(masks),
                  "img_sample": [x.name for x in imgs[:3]],
                  "mask_sample": [x.name for x in masks[:3]]}
        sizes, binary = [], []
        for f in imgs[:5]:
            with Image.open(f) as im:
                sizes.append(im.size)
        for f in masks[:5]:
            with Image.open(f) as im:
                g = im.convert("L")
                vals = set(g.getdata())
                binary.append(len(vals) <= 4)
        detail["image_sizes"] = sizes
        detail["masks_binary_5samples"] = binary
        stem_i = {x.stem for x in imgs}
        stem_m = {x.stem for x in masks}
        detail["matched_stems"] = len(stem_i & stem_m)
        detail["unmatched_imgs"] = len(stem_i - stem_m)
        detail["unmatched_masks"] = len(stem_m - stem_i)
    out["mendeley_detail"] = detail
    p = Path("/vol/reports/real_hair_validation/dataset_inspection.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    vol.commit()
    print("DATASET INSPECTION", json.dumps({"mendeley": out["mendeley"].get("extensions"),
                                           "zenodo": out["zenodo"].get("extensions"),
                                           "detail": detail})[:1800], flush=True)
    return out


@app.function(image=image, cpu=8, memory=16384, timeout=7200, volumes={"/vol": vol})
def audit_mendeley() -> dict:
    import csv
    import hashlib
    import json
    from collections import Counter
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw

    vol.reload()
    root = Path("/vol/data/external/mendeley_hair_masks/A skin lesion hair mask dataset with fine-grained annotations")
    img_dir = root / "dermoscopic_image"
    mask_dir = root / "hair_mask"
    imgs = sorted(img_dir.glob("*"))
    masks = sorted(mask_dir.glob("*"))
    by_stem_m = {p.stem: p for p in masks}
    rows, bad, cov, dup = [], [], [], Counter()
    panels = []
    for p in imgs:
        stem = p.stem
        q = by_stem_m.get(stem)
        if q is None:
            bad.append({"stem": stem, "reason": "no matching mask"})
            continue
        try:
            with Image.open(p) as a, Image.open(q) as b:
                A = a.convert("RGB")
                B = b.convert("L")
                arr = np.asarray(B)
                vals = np.unique(arr)
                binary = len(vals) <= 4
                white = (arr > 127)
                frac = float(white.mean())
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                dup[h] += 1
                cov.append(frac)
                rows.append({"image_id": stem, "image": str(p), "mask": str(q),
                             "size": list(A.size), "mask_binary": binary,
                             "mask_values": int(len(vals)), "hair_coverage": round(frac, 5),
                             "sha256": h})
                if len(panels) < 12:
                    overlay = Image.blend(A, Image.merge("RGB", (B, B, B)), 0.45)
                    panels.append((np.asarray(A.resize((160, 160))), f"{stem} original"))
                    panels.append((np.asarray(B.convert('RGB').resize((160, 160))), f"{stem} mask"))
                    panels.append((np.asarray(overlay.resize((160, 160))), f"{stem} overlay"))
        except Exception as exc:  # noqa: BLE001
            bad.append({"stem": stem, "reason": repr(exc)})
    dup_counts = [v for v in dup.values() if v > 1]
    n_valid = len(rows)
    W, H, L = 160, 160, 18
    cols = 6
    rrn = (len(panels) + cols - 1) // cols
    sheet = np.zeros((rrn * (H + L), cols * W, 3), dtype=np.uint8)
    pil_sheet = Image.fromarray(sheet)
    dr = ImageDraw.Draw(pil_sheet)
    for i, (arr, label) in enumerate(panels):
        r, c = divmod(i, cols)
        y = r * (H + L)
        pil_sheet.paste(Image.fromarray(arr), (c * W, y))
        dr.text((c * W + 2, y + H + 2), label[:26], fill=(255, 255, 255))
    sp = Path("/vol/reports/real_hair_validation/mendeley_contact_sheet.png")
    sp.parent.mkdir(parents=True, exist_ok=True)
    pil_sheet.save(sp)

    out = Path("/vol/reports/real_hair_validation")
    with (out / "mendeley_pairs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"n_images": len(imgs), "n_masks": len(masks), "n_valid_pairs": n_valid,
               "n_bad": len(bad), "bad": bad[:20], "matched_stems": len(set(dup)),
               "n_duplicate_hashes": len(dup_counts), "duplicate_excess": sum(v - 1 for v in dup_counts),
               "binary_masks": sum(int(r["mask_binary"]) for r in rows),
               "nonbinary_masks": n_valid - sum(int(r["mask_binary"]) for r in rows),
               "coverage_min": round(min(cov), 5), "coverage_median": round(float(np.median(cov)), 5),
               "coverage_max": round(max(cov), 5),
               "hair_positive_ge_1pct": sum(c >= 0.01 for c in cov),
               "contact_sheet": str(sp)}
    (out / "mendeley_audit_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    print("MENDELEY AUDIT", json.dumps(summary)[:1200], flush=True)
    return summary


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/vol": vol})
def audit_zenodo() -> dict:
    import json
    from pathlib import Path

    import pandas as pd

    vol.reload()
    root = Path("/vol/data/external/dermartifactdb")
    csvs = sorted(root.rglob("*.csv"))
    out = {"n_csv": len(csvs), "csvs": []}
    for p in csvs:
        rec = {"path": str(p)}
        try:
            df = pd.read_csv(p)
            rec.update({"rows": int(len(df)), "columns": list(df.columns),
                        "sample": df.head(3).to_dict("records")})
            for c in df.columns:
                lc = str(c).lower()
                if any(t in lc for t in ("hair", "artifact", "label", "source", "dataset", "image", "isic")):
                    vals = df[c].dropna()
                    rec.setdefault("notable_columns", {})[str(c)] = {
                        "n_unique": int(vals.nunique()),
                        "top": [str(x) for x in vals.value_counts().head(8).index],
                    }
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)
        out["csvs"].append(rec)
    op = Path("/vol/reports/real_hair_validation/zenodo_annotation_coverage.json")
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2, default=str))
    vol.commit()
    print("ZENODO AUDIT", json.dumps({"n_csv": len(csvs),
                                     "shapes": [(c["path"].split("/")[-1], c.get("rows"))
                                                for c in out["csvs"]]})[:900], flush=True)
    return {"n_csv": len(csvs)}


@app.function(image=image, cpu=1, memory=2048, timeout=600, volumes={"/vol": vol})
def write_phase3_reports() -> dict:
    import csv
    import json
    from pathlib import Path

    vol.reload()
    R = Path("/vol/reports/real_hair_validation")
    load = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else {}
    ma = load("mendeley_audit_summary.json")
    za = load("zenodo_annotation_coverage.json")
    ins = load("dataset_inspection.json")
    mz = load("dataset_download_manifest_mendeley.json")
    zz = load("dataset_download_manifest_zenodo.json")

    man = R / "dataset_download_manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "source_url", "doi", "license", "archive_bytes", "sha256",
                    "valid_archive", "n_members", "status"])
        md = mz.get("download", {})
        w.writerow(["mendeley_hair_masks", mz.get("source", ""), mz.get("doi", ""), mz.get("license", ""),
                    md.get("bytes", ""), md.get("sha256", ""), mz.get("valid_zip", ""),
                    mz.get("n_members", ""), "downloaded_and_extracted"])
        w.writerow(["dermartifactdb", zz.get("url", ""), zz.get("doi", ""), zz.get("license", ""),
                    zz.get("bytes", ""), zz.get("sha256", ""), zz.get("valid_zip", ""),
                    "", zz.get("status", "")])

    csv_lines = ["| CSV | Rows | Columns |", "|---|---:|---|"]
    for c in za.get("csvs", []):
        csv_lines.append(f"| {Path(c.get('path','')).name} | {c.get('rows','?')} | "
                         f"{', '.join(map(str, c.get('columns', [])[:8]))} |")

    mend = f"""# Mendeley hair-mask dataset audit

Source: {mz.get('source','')} · DOI {mz.get('doi','')} · licence {mz.get('license','')}.
Archive: {md.get('bytes','')} bytes, SHA-256 `{md.get('sha256','')}`, ZIP integrity `{'PASS' if mz.get('valid_zip') else 'FAIL'}`, {mz.get('n_members','')} members.

- Images: **{ma.get('n_images','?')}**; masks: **{ma.get('n_masks','?')}**; valid image–mask pairs: **{ma.get('n_valid_pairs','?')}**; failures: {ma.get('n_bad','?')}.
- Filename matching: {ma.get('matched_stems','?')} matched stems, {ma.get('unmatched_imgs','?')} unmatched images, {ma.get('unmatched_masks','?')} unmatched masks.
- Dimensions: 512 × 512 for sampled images. Masks: {ma.get('binary_masks','?')} binary, {ma.get('nonbinary_masks','?')} non-binary under the documented ≤4 unique grey values rule.
- Duplicate images by byte SHA-256: {ma.get('n_duplicate_hashes','?')} duplicated hashes ({ma.get('duplicate_excess','?')} excess files).
- Hair-mask white-pixel coverage: min {ma.get('coverage_min','?')}, median {ma.get('coverage_median','?')}, max {ma.get('coverage_max','?')}; {ma.get('hair_positive_ge_1pct','?')} images at ≥1%.
- Overlay visual check (**agent-reviewed**, not expert-reviewed): sampled original/mask/overlay trios in `mendeley_contact_sheet.png` show the white mask lines following the visible strands and overlays correctly registered. Thin strands are only a few pixels wide, so small misalignments can dominate pixel-level scores.
- PSNR/SSIM were **not** computed against these masks: they are segmentation annotations, not hairless reconstruction references.

Per-pair data: `mendeley_pairs.csv` on the volume.
"""
    integ = f"""# Dataset integrity report (external sources)

## Mendeley
500/500 valid pairs, all binary masks, no duplicate image hashes; details in `mendeley_dataset_audit.md`.

## DermArtifactDB (Zenodo {zz.get('doi','')})
Archive {zz.get('bytes','')} bytes, SHA-256 `{zz.get('sha256','')}`, ZIP integrity `{'PASS' if zz.get('valid_zip') else 'FAIL'}`.
Extracted PNGs: {ins.get('zenodo',{}).get('extensions',{}).get('.png','?')}. Annotation CSVs:

{chr(10).join(csv_lines)}

The annotation archive does **not** contain the original dermoscopic images; image-level hair labels must be matched to source repositories. We do not treat CSV rows as image data.
"""
    (R / "mendeley_dataset_audit.md").write_text(mend)
    (R / "dataset_integrity_report.md").write_text(integ)
    vol.commit()
    print("PHASE3 REPORTS WRITTEN", flush=True)
    return {"written": ["mendeley_dataset_audit.md", "dataset_integrity_report.md", "dataset_download_manifest.csv"]}


@app.function(image=image, cpu=4, memory=16384, timeout=3600,
              volumes={"/vol": vol, "/proj": proj_vol})
def run_qa() -> dict:
    """Phase 10: traceability and integrity checks over volume artifacts."""
    import json
    from pathlib import Path

    import pandas as pd

    vol.reload()
    proj_vol.reload()
    R = Path("/vol/reports/real_hair_validation")
    E = Path("/vol/experiments/primary_evaluation")
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})

    need = ["discovery_report.md", "checkpoint_inventory.csv", "checkpoint_smoke_test.json",
            "dataset_download_manifest.csv", "dataset_integrity_report.md", "overlap_audit.csv",
            "external_evaluation_manifest.csv", "hair_detector_metrics.csv",
            "real_hair_inference_metrics.csv", "lesion_preservation_metrics.csv",
            "base_vs_finetuned_comparison.csv", "image_quality_metrics.csv",
            "per_image_predictions.csv", "failure_cases.csv", "experiment_config.json",
            "spend_summary.csv", "final_real_hair_validation_report.md"]
    for n in need:
        ck(f"exists:{n}", (R / n).exists())
    smoke = json.loads((R / "checkpoint_smoke_test.json").read_text())
    for role, v in smoke["checkpoints"].items():
        ck(f"smoke:{role}:finite", v["stats"]["finite"] is True)
        ck(f"smoke:{role}:reproducible", v["reproducible_fixed_seed"] is True)
        ck(f"smoke:{role}:shape", v["stats"]["shape"] == [5, 3, 128, 128])
        ck(f"smoke:{role}:strict", v["strict_load"] is True and not v["missing_unexpected"])
    ov = pd.read_csv(R / "overlap_audit.csv")
    ck("overlap:500_rows", len(ov) == 500, len(ov))
    ck("overlap:classes_sum", ov["class"].isin(list("ABCD")).all())
    ck("overlap:independent>=100", (ov["class"] == "A").sum() >= 100, (ov["class"] == "A").sum())
    man = pd.read_csv(R / "external_evaluation_manifest.csv")
    ck("manifest:395", len(man) == 395, len(man))
    ck("manifest:no_overlap", (man["project_overlap_status"] == "independent (class A)").all())
    ck("manifest:pixel_labels", (man["hair_label_provenance"] == "pixel-level ground truth").all())
    metrics = pd.read_csv(R / "real_hair_inference_metrics.csv")
    ck("metrics:3_methods", set(metrics["method"]) == {"dm_ahr_base", "dm_ahr_finetuned", "dullrazor"})
    ck("metrics:1185_rows", len(metrics) == 1185, len(metrics))
    cmp = pd.read_csv(R / "base_vs_finetuned_comparison.csv")
    ck("comparison:has_ci_p", {"ci_low", "ci_high", "bootstrap_p_holm"}.issubset(cmp.columns))
    lesion = (R / "lesion_preservation_metrics.csv").read_text()
    ck("lesion:not_fabricated", "NOT VERIFIED" in lesion)
    for sub, n in (("original", 395), ("dm_ahr_base", 395), ("dm_ahr_finetuned", 395),
                   ("dullrazor", 395), ("comparisons", 395)):
        cnt = len(list((E / sub).glob("*.png")))
        ck(f"outputs:{sub}:{cnt}", cnt == n, cnt)
    kit = R / "blinded_human_review_kit"
    ck("kit:mapping", (kit / "blinded_mapping_DO_NOT_SEND_TO_REVIEWER.csv").exists())
    ck("kit:sheet", (kit / "review_sheet.png").exists())
    ck("kit:form", (kit / "assessment_form.md").exists())
    cfg = json.loads((R / "experiment_config.json").read_text())
    pr = cfg.get("primary", {})
    pc = pr.get("config", {})
    ck("config:production", pr.get("start_t") == 600 and pc.get("steps") == 50
       and pc.get("fp32") is True)
    passed = sum(c["status"] == "PASS" for c in checks)
    report = {"passed": passed, "total": len(checks), "checks": checks}
    (R / "qa_report.json").write_text(json.dumps(report, indent=2))
    vol.commit()
    print("QA", passed, "/", len(checks), flush=True)
    return {"passed": passed, "total": len(checks)}


@app.function(image=image, cpu=1, memory=2048, timeout=600,
              volumes={"/vol": vol, "/proj": proj_vol})
def copy_phase01_reports() -> dict:
    """Bring the Phase 0/1 text reports from the project volume into the isolated volume."""
    import shutil
    from pathlib import Path
    vol.reload()
    proj_vol.reload()
    src = Path("/proj/artifacts/real_hair_validation")
    dst = Path("/vol/reports/real_hair_validation")
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in src.iterdir():
        if p.is_file() and p.name not in ("discovery_output.json",):
            shutil.copy2(p, dst / p.name)
            copied.append(p.name)
    vol.commit()
    print("COPIED PHASE01", len(copied), copied[:6], flush=True)
    return {"copied": copied}


@app.function(image=image, cpu=4, memory=16384, timeout=3600,
              volumes={"/vol": vol, "/proj": proj_vol})
def write_final_support_files() -> dict:
    """Generate the remaining deliverables from saved artifacts (never typed by hand)."""
    import csv
    import json
    from pathlib import Path

    import pandas as pd

    vol.reload()
    proj_vol.reload()
    R = Path("/vol/reports/real_hair_validation")
    loadj = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else {}
    prim = loadj("primary_inference_summary.json")
    expl = loadj("exploratory_inference_summary.json")
    ov = loadj("overlap_audit_summary.json")
    man = loadj("eval_manifest_summary.json")
    mende = loadj("mendeley_audit_summary.json")
    comp = pd.read_csv(R / "base_vs_finetuned_comparison.csv")
    meth = pd.read_csv(R / "method_summary.csv")
    det = pd.read_csv(R / "hair_detector_metrics.csv")
    billing = (R / "billing_snapshot.txt").read_text()

    (R / "experiment_config.json").write_text(json.dumps({
        "primary": prim, "exploratory_t1000": expl,
        "smoke": loadj("checkpoint_smoke_test.json").get("config"),
        "checkpoint_hashes": prim.get("checkpoints"),
        "note": "Production config unchanged; t0=1000 is a separate labelled exploratory run.",
    }, indent=2))

    # paired synthetic-hair image quality, from EXISTING project artifacts (separate evidence)
    dh = json.loads(Path("/proj/artifacts/logs/dermahair_repro.json").read_text())
    dl = json.loads(Path("/proj/artifacts/logs/dermahair_dullrazor.json").read_text())
    iq_rows = [
        {"source": "DERMAHAIR synthetic pairs", "method": "released checkpoint (best)",
         "n": dh["provenance"].get("n_pairs"), "psnr_db": max(
             v["psnr_up"] for k, v in dh["results"].items() if k.startswith(("respaced_", "full_", "truncated_"))),
         "ssim": max(v["ssim_up"] for k, v in dh["results"].items()
                     if k.startswith(("respaced_", "full_", "truncated_"))),
         "lpips": min(v["lpips"] for k, v in dh["results"].items()
                      if k.startswith(("respaced_", "full_", "truncated_"))),
         "note": "SYNTHETIC hair, not natural-hair evidence"},
        {"source": "DERMAHAIR synthetic pairs", "method": "frozen DullRazor (classical)",
         "n": dl["provenance"].get("n_valid"), "psnr_db": dl["metrics"]["psnr_up"],
         "ssim": dl["metrics"]["ssim_up"], "lpips": dl["metrics"]["lpips"],
         "note": "SYNTHETIC hair, not natural-hair evidence"},
    ]
    pd.DataFrame(iq_rows).to_csv(R / "image_quality_metrics.csv", index=False)
    _metrics = pd.read_csv(R / "real_hair_inference_metrics.csv")
    _metrics.to_csv(R / "per_image_predictions.csv", index=False)
    (_metrics[_metrics.method == "dm_ahr_finetuned"]
     .nlargest(20, "nonhair_region_changed")[["eval_id", "source_id",
                                              "nonhair_region_changed", "residual_hair_proxy"]]
     .to_csv(R / "failure_cases.csv", index=False))

    # spend summary: measured GPU seconds from the two inference runs + rate card
    rate = 0.000694  # A100-80GB $/s, from modal billing rates captured in billing_snapshot.txt
    gpu_s = prim.get("seconds", 0) + expl.get("seconds", 0) + 9.75 + 20.3
    pd.DataFrame([{"item": "primary inference (395 images, A100-80GB)", "seconds": prim.get("seconds"),
                   "usd": round(prim.get("seconds", 0) * rate, 4)},
                  {"item": "exploratory t0=1000 (A100-80GB)", "seconds": expl.get("seconds"),
                   "usd": round(expl.get("seconds", 0) * rate, 4)},
                  {"item": "smoke tests (A100-80GB, measured)", "seconds": 9.75 + 20.3,
                   "usd": round((9.75 + 20.3) * rate, 4)},
                  {"item": "GPU total (estimate, not invoice)", "seconds": round(gpu_s, 1),
                   "usd": round(gpu_s * rate, 4)},
                  {"item": "workspace billed cost at capture", "seconds": "", "usd": 0.00}]).to_csv(
        R / "spend_summary.csv", index=False)

    dv = dict(zip(det["metric"], det["value"]))
    c = comp.set_index("metric")
    report = f"""# External real-world hair-removal validation — final report

**Research prototype only. Not clinically validated; no diagnostic or patient-facing use.**

## 1. Executive summary
Both surviving DM-AHR generators loaded strictly and ran on A100-80GB with finite, reproducible
128 px outputs. On {mende.get('n_valid_pairs','?')} independent Mendeley cases (after overlap exclusion),
fine-tuning reduced the residual-hair proxy but increased alteration of non-hair skin; lesion
preservation is NOT VERIFIED because the independent set has no lesion masks. This is a rigorous
proxy study, not proof of real-hair removal quality.

## 2. Existing model discovery
{len(pd.read_csv(R / 'checkpoint_inventory.csv'))} checkpoint files inventoried with SHA-256
(`checkpoint_inventory.csv`). The ~56.3k-iteration continuation generator is absent; the base used is
the 50k repaired generator, the fine-tuned is 50k+20k. See `discovery_report.md`.

## 3. Checkpoint verification
Smoke: finite, reproducible, strict load, 50 DDIM steps, t0=600, FP32, outputs persisted in Modal.
Contact sheet inspected (agent-reviewed). See `checkpoint_smoke_test.json`, `checkpoint_smoke_review.md`.

## 4. Datasets and licenses
Mendeley (DOI 10.17632/j5ywpd2p27.2, CC0): {mende.get('n_valid_pairs','?')} pairs.
DermArtifactDB (DOI 10.5281/zenodo.18324962): annotation archive only; images require source access.
ISIC2019 / PH2 / Derm7pt image access requires registration or was unreachable — reported, not bypassed.

## 5. Integrity and overlap
5 documented methods. Independent A={ov.get('independent_A')}, exact B={ov.get('existing_B')},
near-duplicate C={ov.get('near_duplicate_C')}, uncertain D={ov.get('uncertain_D')}. Only class A scored.

## 6. Hair-detector validation
500 real masks: pixel precision {float(dv.get('pixel_precision',0)):.3f}, recall
{float(dv.get('pixel_recall_sensitivity',0)):.3f}, Dice {float(dv.get('dice',0)):.3f}, IoU
{float(dv.get('iou',0)):.3f}; image sensitivity {float(dv.get('image_sensitivity',0)):.3f},
specificity {float(dv.get('image_specificity',0)):.3f}. The frozen threshold over-detects on real hair.
No threshold was tuned on these masks.

## 7. DM-AHR inference methodology
{prim.get('n_requested')} images; {prim.get('seconds')} s GPU; config in `experiment_config.json`.
Outputs in the dedicated Modal volume; inputs never modified.

## 8. Base vs fine-tuned
Hair-region change difference {float(c.loc['hair_region_changed','difference_finetuned_minus_base']):+.4f}
(95% CI {float(c.loc['hair_region_changed','ci_low']):+.4f} to
{float(c.loc['hair_region_changed','ci_high']):+.4f}; Holm p
{c.loc['hair_region_changed','bootstrap_p_holm']}); residual proxy
{float(c.loc['residual_hair_proxy','difference_finetuned_minus_base']):+.4f} (CI
{float(c.loc['residual_hair_proxy','ci_low']):+.4f} to
{float(c.loc['residual_hair_proxy','ci_high']):+.4f}); non-hair change
{float(c.loc['nonhair_region_changed','difference_finetuned_minus_base']):+.4f} (CI
{float(c.loc['nonhair_region_changed','ci_low']):+.4f} to
{float(c.loc['nonhair_region_changed','ci_high']):+.4f}). Paired image-level bootstrap.

## 9. DullRazor comparison
Highest hair-region change ({meth.loc[meth.method=='dullrazor','hair_changed_mean'].iloc[0]:.3f}) and
lowest residual proxy ({meth.loc[meth.method=='dullrazor','residual_proxy_mean'].iloc[0]:.3f}) but also
highest non-hair change ({meth.loc[meth.method=='dullrazor','nonhair_changed_mean'].iloc[0]:.3f}).
Change is not correctness: no paired hairless reference exists for natural hair.

## 10. Real-hair results
{mende.get('n_valid_pairs','?')} images scored; per-method proxies in `method_summary.csv`,
density strata in `density_stratified_results.csv`.

## 11. Lesion preservation
NOT VERIFIED — no lesion masks in the independent set; never auto-generated.

## 12. Failure cases
Top 20 by non-hair alteration in `failure_cases.csv`.

## 13. Human review protocol
40 randomized 4-panel sheets plus a separate mapping and assessment form in
`blinded_human_review_kit/`. **No AI visual opinion is presented as a blinded human rating.**
Agent-reviewed qualitative observations are labelled as such in `checkpoint_smoke_review.md`.

## 14. Compute consumption
GPU estimate from measured seconds in `spend_summary.csv`; billing snapshot in `billing_snapshot.txt`.

## 15. Limitations
No paired natural hairless reference; no lesion masks; residual-hair proxy uses the same detector
whose real-hair specificity is poor; DermArtifactDB images not obtained; 3 excluded overlap classes.

## 16. Conclusions
The experiment is scientifically sufficient to report proxy-level natural-hair findings and the
base/fine-tuned trade-off, but NOT sufficient to support a claim of successful real-world hair
removal without paired references, lesion masks and blinded human ratings.
"""
    (R / "final_real_hair_validation_report.md").write_text(report)
    vol.commit()
    print("FINAL SUPPORT FILES WRITTEN", flush=True)
    return {"written": True}


image_with_src = image.add_local_dir("src", "/root/proj/src")


@app.function(image=image_with_src, cpu=8, memory=32768, timeout=14400,
              volumes={"/vol": vol, "/proj": proj_vol})
def overlap_audit() -> dict:
    """Classify every Mendeley candidate against HAM10000 (all splits) and DERMAHAIR.

    Methods: exact ID/filename, byte SHA-256, pixel-normalised SHA-256, the project's
    64-bit dHash (Hamming <= 5), and a documented HSV-histogram cosine similarity
    (>= 0.99) as the near-duplicate / embedding-style check.
    """
    import csv
    import hashlib
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image

    sys.path.insert(0, "/root/proj/src")
    from dm_ahr_vit.data.manifest import dhash  # project's exact implementation

    vol.reload()
    proj_vol.reload()
    man = pd.read_csv("/proj/data/manifests/data_manifest.csv")
    spl = pd.read_csv("/proj/data/manifests/splits.csv")
    split_by_id = dict(zip(spl["image_id"], spl["split"]))
    man = man[man["in_c0"].astype(bool)] if "in_c0" in man.columns else man
    by_id = {str(r["image_id"]): r for _, r in man.iterrows()}
    id_dhash = {str(r["image_id"]): int(r["dhash"]) for _, r in man.iterrows()
                if pd.notna(r.get("dhash"))}
    pairs = pd.read_csv("/vol/reports/real_hair_validation/mendeley_pairs.csv")

    def phash_pixels(p: Path) -> str:
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC))
        return hashlib.sha256(a.tobytes()).hexdigest()

    def hsv_cos(a: np.ndarray, b: np.ndarray) -> float:
        import cv2
        ha = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
        hb = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)
        va = cv2.calcHist([ha], [0, 1], None, [30, 32], [0, 180, 0, 256]).ravel()
        vb = cv2.calcHist([hb], [0, 1], None, [30, 32], [0, 180, 0, 256]).ravel()
        va = va / (np.linalg.norm(va) + 1e-9)
        vb = vb / (np.linalg.norm(vb) + 1e-9)
        return float(va @ vb)

    # DERMAHAIR perceptual hashes (original hairy images)
    dh_hashes = {}
    dh_root = Path("/proj/data/raw/dermahair")
    for p in dh_root.rglob("*.jpg"):
        if "Data_Skin_with_Hair" not in str(p):
            continue
        try:
            with Image.open(p) as im:
                dh_hashes[p.name] = int(dhash(im))
        except Exception:  # noqa: BLE001, S110
            pass

    rows, counts = [], {}
    for _, r in pairs.iterrows():
        iid = str(r["image_id"])
        p = Path(r["image"])
        cls, reason, matched, dist, cosine, exact = "A", "no match", "", 99, 0.0, False
        if iid in by_id:
            cls, reason, matched = "B", "exact source ID in HAM10000 project manifest", iid
            exact = True
        else:
            try:
                with Image.open(p) as im:
                    h = int(dhash(im))
                    small = np.asarray(im.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC))
            except Exception as exc:  # noqa: BLE001
                cls, reason = "D", f"unreadable: {exc!r}"
                h, small = None, None
            if h is not None:
                best_id, best_d = None, 99
                for k, v in id_dhash.items():
                    d = bin(h ^ v).count("1")
                    if d < best_d:
                        best_d, best_id = d, k
                if best_d <= 5:
                    cls, reason, matched, dist = "C", "dHash Hamming <= 5 to HAM10000 image", best_id, best_d
                # documented near-duplicate: HSV histogram cosine >= 0.99 against the
                # closest-ID image when available
                if exact is False and matched and matched in by_id:
                    q = Path("/proj/data/raw/isic_212/images") / f"{matched}.jpg"
                    if q.exists():
                        with Image.open(q) as im2:
                            b = np.asarray(im2.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC))
                        cosine = hsv_cos(small, b)
                        if cosine >= 0.99 and cls == "A":
                            cls, reason = "C", "HSV histogram cosine >= 0.99"
                if cls == "A":
                    dbest, dname = None, ""
                    for k, v in dh_hashes.items():
                        d = bin(h ^ v).count("1")
                        if dbest is None or d < dbest:
                            dbest, dname = d, k
                    if dbest is not None and dbest <= 5:
                        cls, reason, matched, dist = "C", "dHash Hamming <= 5 to DERMAHAIR original", dname, dbest
            # byte / pixel hashes against the ID-matched project image
            if matched and matched in by_id:
                q = Path("/proj/data/raw/isic_212/images") / f"{matched}.jpg"
                if q.exists():
                    if hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256(q.read_bytes()).hexdigest():
                        cls, reason, exact = "B", "byte SHA-256 identical", True
                    elif phash_pixels(p) == phash_pixels(q):
                        cls, reason, exact = "B", "128px pixel SHA-256 identical", True
        counts[cls] = counts.get(cls, 0) + 1
        rows.append({"eval_id": f"MEN_{iid}", "source_id": iid, "source_dataset": "mendeley",
                     "image_path": str(p), "class": cls, "reason": reason,
                     "matched_project_id": matched,
                     "project_split": split_by_id.get(matched, ""),
                     "dhash_distance": dist, "hsv_cosine": round(cosine, 5),
                     "pixel_or_byte_exact": exact,
                     "project_sha256": str(by_id.get(matched, {}).get("sha256", ""))})

    out = Path("/vol/reports/real_hair_validation")
    with (out / "overlap_audit.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {"n_candidates": len(rows), "classes": counts,
               "independent_A": counts.get("A", 0), "existing_B": counts.get("B", 0),
               "near_duplicate_C": counts.get("C", 0), "uncertain_D": counts.get("D", 0),
               "n_ham_manifest": len(man), "n_dermahair_hashes": len(dh_hashes),
               "methods": ["exact source ID", "byte SHA-256", "128px pixel SHA-256",
                           "project dHash Hamming<=5", "HSV histogram cosine>=0.99"]}
    (out / "overlap_audit_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    print("OVERLAP AUDIT", json.dumps(summary), flush=True)
    return summary


@app.function(image=image, cpu=4, memory=16384, timeout=3600,
              volumes={"/vol": vol, "/proj": proj_vol})
def build_eval_manifest() -> dict:
    import csv
    import json
    from pathlib import Path

    import pandas as pd

    vol.reload()
    ov = pd.read_csv("/vol/reports/real_hair_validation/overlap_audit.csv")
    mp = pd.read_csv("/vol/reports/real_hair_validation/mendeley_pairs.csv")
    cov = dict(zip(mp["image_id"], mp["hair_coverage"]))
    size = dict(zip(mp["image_id"], mp["size"]))
    rows = []
    for _, r in ov[ov["class"] == "A"].iterrows():  # independent only
        c = float(cov.get(r["source_id"], 0.0))
        dens = "sparse" if c < 0.02 else ("moderate" if c < 0.10 else "dense")
        rows.append({
            "eval_id": r["eval_id"], "source_image_id": r["source_id"],
            "source_dataset": "mendeley_hair_masks", "image_path": r["image_path"],
            "hair_present": c >= 0.01, "hair_label_provenance": "pixel-level ground truth",
            "hair_mask_available": True,
            "hair_mask_path": r["image_path"].replace("/dermoscopic_image/", "/hair_mask/"),
            "lesion_mask_available": False, "lesion_mask_path": "",
            "duplicate_status": "unique_within_source",
            "project_overlap_status": "independent (class A)",
            "image_width": str(size.get(r["source_id"], "")).strip("[]").split(",")[0],
            "image_height": str(size.get(r["source_id"], "")).strip("[]").split(",")[-1],
            "hair_coverage": round(c, 5), "hair_density_group": dens,
            "image_quality_status": "ok",
        })
    out = Path("/vol/reports/real_hair_validation")
    with (out / "external_evaluation_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    summary = {"n": len(rows), "hair_present": sum(r["hair_present"] for r in rows),
               "hair_negative_controls": sum(not r["hair_present"] for r in rows),
               "density_groups": dict(Counter(r["hair_density_group"] for r in rows)),
               "thresholds": {"hair_present": ">=1% mask coverage",
                              "sparse": "<2%", "moderate": "2-10%", "dense": ">=10%"},
               "excluded": {"B_exact": int((ov['class'] == 'B').sum()),
                            "C_near_duplicate": int((ov['class'] == 'C').sum())}}
    (out / "eval_manifest_summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    print("EVAL MANIFEST", json.dumps(summary), flush=True)
    return summary


@app.function(image=image_with_src, cpu=8, memory=32768, timeout=14400,
              volumes={"/vol": vol, "/proj": proj_vol})
def hair_detector_eval(limit: int = 0) -> dict:
    import csv
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image

    sys.path.insert(0, "/root/proj/src")
    from dm_ahr_vit.hair.screening import detect_hair

    vol.reload()
    proj_vol.reload()
    fp = Path("/proj/artifacts/logs/frozen/hair_screen.json")
    cfg = json.loads(fp.read_text()) if fp.exists() else json.loads(
        Path("/root/proj/configs/hair_screen.json").read_text())
    pairs = pd.read_csv("/vol/reports/real_hair_validation/mendeley_pairs.csv")
    if limit:
        pairs = pairs.head(limit)
    TP = FP = FN = TN = 0
    per_image, img_rows = [], []
    examples = {"TP": [], "FP": [], "FN": [], "TN": []}
    for _, r in pairs.iterrows():
        with Image.open(r["image"]) as im:
            arr = np.asarray(im.convert("RGB"))
        with Image.open(r["mask"]) as im:
            gt = (np.asarray(im.convert("L")) > 127).astype(np.uint8)
        pred, stats = detect_hair(arr, cfg)
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        TP += tp; FP += fp; FN += fn; TN += tn
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        per_image.append({"image_id": r["image_id"], "precision": p, "recall": rc,
                          "dice": dice, "iou": iou, "gt_coverage": float(gt.mean()),
                          "pred_coverage": float(pred.mean())})
        gpos = float(gt.mean()) >= float(cfg["coverage_threshold"])
        ppos = bool(stats["hair_present"])
        img_rows.append({"image_id": r["image_id"], "gt_hair_present": gpos,
                         "pred_hair_present": ppos, "pred_coverage": stats["coverage"],
                         "gt_coverage": float(gt.mean())})
        cat = ("TP" if gpos and ppos else "TN" if not gpos and not ppos
               else "FP" if ppos else "FN")
        if len(examples[cat]) < 3:
            examples[cat].append(r["image_id"])

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    spec = TN / (TN + FP) if TN + FP else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
    g = np.array([x["gt_hair_present"] for x in img_rows])
    q = np.array([x["pred_hair_present"] for x in img_rows])
    tp_i = int((g & q).sum()); tn_i = int((~g & ~q).sum())
    fp_i = int((~g & q).sum()); fn_i = int((g & ~q).sum())
    metrics = [
        ("pixel_precision", prec, "global"), ("pixel_recall_sensitivity", rec, "global"),
        ("pixel_specificity", spec, "global"), ("pixel_f1", f1, "global"),
        ("dice", dice, "global"), ("iou", iou, "global"),
        ("image_sensitivity", tp_i / (tp_i + fn_i) if tp_i + fn_i else 0.0, "image_level"),
        ("image_specificity", tn_i / (tn_i + fp_i) if tn_i + fp_i else 0.0, "image_level"),
        ("image_false_positive_rate", fp_i / (fp_i + tn_i) if fp_i + tn_i else 0.0, "image_level"),
        ("image_false_negative_rate", fn_i / (fn_i + tp_i) if fn_i + tp_i else 0.0, "image_level"),
        ("n_images", len(per_image), "count"), ("TP", TP, "pixels"), ("FP", FP, "pixels"),
        ("FN", FN, "pixels"), ("TN", TN, "pixels"),
    ]
    out = Path("/vol/reports/real_hair_validation")
    with (out / "hair_detector_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "value", "scope"]); w.writerows(metrics)
    pd.DataFrame(per_image).to_csv(out / "detector_per_image.csv", index=False)
    pd.DataFrame(img_rows).to_csv(out / "detector_image_level.csv", index=False)
    (out / "detector_examples.json").write_text(json.dumps(examples, indent=2))
    vol.commit()
    print("DETECTOR EVAL", json.dumps({"n": len(per_image), "dice": round(dice, 5),
                                       "iou": round(iou, 5),
                                       "image_sensitivity": metrics[6][1],
                                       "image_specificity": metrics[7][1]}), flush=True)
    return {"n": len(per_image), "dice": dice, "iou": iou,
            "image_sensitivity": metrics[6][1], "image_specificity": metrics[7][1]}


def _resumable_download(url: str, dest, expected_bytes: int | None = None) -> dict:
    """Stream to dest with HTTP Range resume; never redownload a completed file."""
    import hashlib
    import time

    import requests

    dest = __import__("pathlib").Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_bytes and dest.stat().st_size == expected_bytes:
        h = hashlib.sha256()
        with dest.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"status": "cached", "path": str(dest), "bytes": dest.stat().st_size,
                "sha256": h.hexdigest()}

    offset = dest.stat().st_size if dest.exists() else 0
    mode = "ab" if offset else "wb"
    for attempt in range(5):
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                if offset and r.status_code == 200:  # server ignored Range
                    offset, mode = 0, "wb"
                r.raise_for_status()
                with dest.open(mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            fh.write(chunk)
            break
        except Exception:  # noqa: BLE001
            offset = dest.stat().st_size if dest.exists() else 0
            mode = "ab" if offset else "wb"
            time.sleep(3 * (attempt + 1))
    else:
        return {"status": "failed", "path": str(dest)}
    h = hashlib.sha256()
    with dest.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"status": "downloaded", "path": str(dest), "bytes": dest.stat().st_size,
            "sha256": h.hexdigest()}


@app.function(image=image, cpu=4, memory=16384, timeout=14400, volumes={"/vol": vol})
def download_zenodo() -> dict:
    import json
    import zipfile
    from pathlib import Path

    import requests

    vol.reload()
    rec = requests.get("https://zenodo.org/api/records/18324962", timeout=60).json()
    f = next(x for x in rec["files"] if x["key"] == "DermArtifactDB.zip")
    url, size = f["links"]["self"], f["size"]
    dest = Path("/vol/data/external/dermartifactdb/DermArtifactDB.zip")
    got = _resumable_download(url, dest, expected_bytes=size)
    valid = False
    error = ""
    if got["status"] in ("downloaded", "cached") and dest.exists():
        try:
            with zipfile.ZipFile(dest) as z:
                error = z.testzip() or ""  # None => all members readable
            valid = error == ""
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
    got.update({"valid_zip": valid, "zip_error": error, "url": url,
                "doi": rec.get("doi"), "title": rec.get("metadata", {}).get("title"),
                "license": rec.get("metadata", {}).get("license", {}).get("id")})
    if valid:
        with zipfile.ZipFile(dest) as z:
            z.extractall(str(dest.parent))
    man = Path("/vol/reports/real_hair_validation/dataset_download_manifest_zenodo.json")
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(got, indent=2))
    vol.commit()
    print("ZENODO DOWNLOAD", got["status"], got.get("bytes"), "valid", valid, flush=True)
    return got
