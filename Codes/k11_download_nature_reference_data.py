#!/usr/bin/env python
"""Download the authoritative observational inputs used by Figures 1-4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PAPER_DIR / "Data/Nature_real_rebuild/raw"

SOURCES = {
    "noaa_ersstv5_sst_monthly.nc": {
        "url": "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc",
        "provider": "NOAA Physical Sciences Laboratory",
        "product": "NOAA Extended Reconstructed SST version 5, monthly mean",
    },
    "noaa_olr_monthly_v03r00_197901_202606.nc": {
        "url": (
            "https://archive.data.noaa.gov/climatedatarecords/UMD_ESSIC/OLR_CDR/Monthly/"
            "OLR-M-CDR_01B-06/OLR-Monthly_v03r00_s197901_e202606.nc"
        ),
        "provider": "NOAA National Centers for Environmental Information",
        "product": "Monthly Outgoing Longwave Radiation CDR version 03 revision 00",
    },
    "cpc_ersst5_nino_monthly_1991_2020_base.txt": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
        "provider": "NOAA Climate Prediction Center",
        "product": "ERSSTv5 monthly Nino indices, 1991-2020 base period",
    },
    "cpc_nino34_detrended.txt": {
        "url": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt",
        "provider": "NOAA Climate Prediction Center",
        "product": "Detrended monthly Nino3.4 index",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "OA-MHW-real-figure-rebuild/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target, length=16 * 1024 * 1024)
    temporary.replace(output)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, source in SOURCES.items():
        output = args.output_dir / filename
        if args.force or not output.is_file() or output.stat().st_size == 0:
            print(f"[Download] {source['url']} -> {output}", flush=True)
            download(str(source["url"]), output)
        record = {
            **source,
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        }
        records.append(record)
        print(f"[Verified] {filename}: {record['bytes']:,} bytes {record['sha256']}", flush=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Files are immutable inputs; rerun with --force to refresh from the recorded URLs.",
        "sources": records,
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
