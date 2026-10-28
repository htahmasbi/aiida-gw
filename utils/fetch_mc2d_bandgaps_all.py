#!/usr/bin/env python3
"""Fetch band gaps for all MC2D structures from Materials Cloud API."""
import argparse
import csv
import glob
import json
import logging
import os
import time
from pathlib import Path

import requests

LOGGER = logging.getLogger("fetch_mc2d_bandgaps_all")
MC2D_API = "https://mcxd-api.materialscloud.org/mc2d/pbe-v1/core_base"
HEADERS = {"User-Agent": "aiida-gw-bandgap-fetcher/1.0", "Accept": "application/json"}


def find_bandgap(data):
    """Recursively search for band_gap value."""
    if isinstance(data, dict):
        if "band_gap" in data and isinstance(data["band_gap"], dict):
            if "value" in data["band_gap"]:
                return data["band_gap"]["value"]
        for v in data.values():
            result = find_bandgap(v)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_bandgap(item)
            if result is not None:
                return result
    return None


def fetch_bandgap(mc2d_id, timeout=30):
    url = f"{MC2D_API}/{mc2d_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return find_bandgap(data)
    except Exception as e:
        LOGGER.debug(f"Failed {mc2d_id}: {e}")
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", default=".", help="Directory containing mc2d_*elements*.json files")
    parser.add_argument("--out", default="mc2d_bandgaps_all.csv", help="Output CSV file")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between requests")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of structures (for testing)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Find all mc2d json files
    patterns = [
        os.path.join(args.json_dir, "mc2d_*elements*.json"),
        os.path.join(args.json_dir, "mc2d*.json"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    LOGGER.info(f"Found {len(files)} JSON files: {[os.path.basename(f) for f in files]}")

    # Load all structures
    all_structs = []
    seen_ids = set()
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    mc2d_id = item.get("id")
                    if mc2d_id in seen_ids:
                        continue
                    seen_ids.add(mc2d_id)
                    all_structs.append(item)
        except Exception as e:
            LOGGER.warning(f"Failed to load {fpath}: {e}")

    if args.limit:
        all_structs = all_structs[:args.limit]
    LOGGER.info(f"Total unique structures: {len(all_structs)}")

    rows = []
    for i, item in enumerate(all_structs):
        mc2d_id = item.get("id", "")
        formula = item.get("formula", "")
        LOGGER.info(f"[{i+1}/{len(all_structs)}] {mc2d_id} {formula}")
        bandgap = fetch_bandgap(mc2d_id)
        rows.append({
            "mc2d_id": mc2d_id,
            "formula": formula,
            "nelements": item.get("nelements", ""),
            "bandgap": bandgap if bandgap is not None else "NOT_FOUND",
        })
        if args.sleep > 0:
            time.sleep(args.sleep)

    rows.sort(key=lambda x: (str(x["bandgap"]), x["formula"]))
    with open(args.out, "w", newline="", encoding="utf-utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mc2d_id", "formula", "nelements", "bandgap"])
        w.writeheader()
        w.writerows(rows)

    found = sum(1 for r in rows if r["bandgap"] != "NOT_FOUND")
    LOGGER.info(f"Done. Saved {len(rows)} rows to {args.out} ({found} bandgaps found)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
