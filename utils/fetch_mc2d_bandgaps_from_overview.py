#!/usr/bin/env python3
"""Fetch band gaps for all MC2D structures from Materials Cloud overview API."""
import argparse
import csv
import glob
import json
import logging
import os

import requests
from pymatgen.core import Composition

LOGGER = logging.getLogger("fetch_mc2d_bandgaps_overview")
MC2D_OVERVIEW = "https://mcxd-api.materialscloud.org/mc2d/pbe-v1/overview"
HEADERS = {"User-Agent": "aiida-gw-bandgap-fetcher/1.0", "Accept": "application/json"}


def _norm_formula(formula):
    if not formula:
        return None
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return str(formula).strip()


def fetch_overview(timeout=60):
    LOGGER.info(f"Fetching overview from {MC2D_OVERVIEW}")
    resp = requests.get(MC2D_OVERVIEW, headers=HEADERS, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    LOGGER.info(f"Got {len(data)} entries from overview")
    return data


def build_lookup(overview):
    lookup = {}
    for entry in overview:
        formula = entry.get("formula")
        bg = entry.get("bg")
        if not formula:
            continue
        rf = _norm_formula(formula)
        if rf:
            if rf not in lookup or (lookup[rf] is None and bg is not None):
                lookup[rf] = bg
        lookup.setdefault(formula, bg)
    return lookup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", default=".", help="Directory containing mc2d_*elements*.json files")
    parser.add_argument("--out", default="mc2d_bandgaps_all.csv", help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of structures (for testing)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overview = fetch_overview()
    lookup = build_lookup(overview)
    LOGGER.info(f"Built lookup for {len(lookup)} unique formulas")

    patterns = [
        os.path.join(args.json_dir, "mc2d_*elements*.json"),
        os.path.join(args.json_dir, "mc2d*.json"),
    ]
    files = []
    seen_files = set()
    for p in patterns:
        for f in glob.glob(p):
            if f not in seen_files:
                seen_files.add(f)
                files.append(f)
    files = sorted(files)
    LOGGER.info(f"Found {len(files)} JSON files: {[os.path.basename(f) for f in files]}")

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
        rf = _norm_formula(formula)
        bandgap = None
        if rf and rf in lookup:
            bandgap = lookup[rf]
        elif formula in lookup:
            bandgap = lookup[formula]
        if bandgap is None:
            bandgap = "NOT_FOUND"
        rows.append({
            "mc2d_id": mc2d_id,
            "formula": formula,
            "reduced_formula": rf or "",
            "nelements": item.get("nelements", ""),
            "bandgap": bandgap,
        })
        if (i + 1) % 100 == 0:
            LOGGER.info(f"Processed {i + 1}/{len(all_structs)}")

    rows.sort(key=lambda x: (x["reduced_formula"] or x["formula"], x["mc2d_id"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mc2d_id", "formula", "reduced_formula", "nelements", "bandgap"])
        w.writeheader()
        w.writerows(rows)

    found = sum(1 for r in rows if r["bandgap"] != "NOT_FOUND")
    LOGGER.info(f"Done. Saved {len(rows)} rows to {args.out} ({found}/{len(rows)} bandgaps found)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
