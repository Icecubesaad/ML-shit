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
