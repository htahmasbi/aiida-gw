#!/usr/bin/env python3
"""Fetch band gaps for all MC2D structures from Materials Cloud overview API."""
import argparse
import csv
import glob
import json
import logging
import os
import time

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


def fetch_core_base(mc2d_id, timeout=30):
    url = f"https://mcxd-api.materialscloud.org/mc2d/pbe-v1/core_base/{mc2d_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception as e:
        LOGGER.debug(f"Failed {mc2d_id}: {e}")
        return None


def build_mappings(overview, opt_set, sleep=0.05):
    """Build UUID->bandgap by only checking core_base for entries that could match our OPTIMADE IDs."""
    uuid_to_bg = {}
    uuid_to_mc2d = {}
    by_rf = {}
    for entry in overview:
        formula = entry.get("formula")
        rf = _norm_formula(formula)
        if rf:
            by_rf.setdefault(rf, []).append(entry)

    LOGGER.info(f"Checking core_base for {len(overview)} entries to find matches to {len(opt_set)} OPTIMADE structures...")
    for i, entry in enumerate(overview):
        mc2d_id = entry.get("id")
        if not mc2d_id:
            continue
        data = fetch_core_base(mc2d_id)
        if not data:
            continue
        gen = data.get("general", {}) if isinstance(data, dict) else {}
        relaxed_uuid = gen.get("structure_relaxed_uuid")
        extracted_uuid = gen.get("structure_extracted_uuid")
        bg = None
        try:
            if isinstance(data.get("electronic"), dict):
                bg = data["electronic"].get("band_gap", {}).get("value")
        except Exception:
            pass
        if bg is None:
            bg = entry.get("bg")
        if bg is None:
            continue
        if relaxed_uuid and relaxed_uuid in opt_set:
            uuid_to_bg[relaxed_uuid] = bg
            uuid_to_mc2d[relaxed_uuid] = mc2d_id
        if extracted_uuid and extracted_uuid in opt_set and extracted_uuid not in uuid_to_bg:
            uuid_to_bg[extracted_uuid] = bg
            uuid_to_mc2d[extracted_uuid] = mc2d_id
        if (i + 1) % 50 == 0:
            LOGGER.info(f"Checked {i + 1}/{len(overview)} (found {len(uuid_to_bg)} matches so far)")
        if sleep > 0:
            time.sleep(sleep)
    LOGGER.info(f"UUID matches: {len(uuid_to_bg)}")
    return uuid_to_bg, uuid_to_mc2d, by_rf


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", default=".", help="Directory containing mc2d_*elements*.json files")
    parser.add_argument("--out", default="mc2d_bandgaps_all.csv", help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of structures (for testing)")
    parser.add_argument("--sleep", type=float, default=0.03, help="Delay between core_base requests")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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

    opt_set = set(s.get("id") for s in all_structs if s.get("id"))
    overview = fetch_overview()
    uuid_to_bg, uuid_to_mc2d, by_rf = build_mappings(overview, opt_set, sleep=args.sleep)

    rows = []
    for i, item in enumerate(all_structs):
        opt_id = item.get("id", "")
        formula = item.get("formula", "")
        rf = _norm_formula(formula)
        bandgap = None
        matched_mc2d = ""
        match_reason = ""

        if opt_id in uuid_to_bg and uuid_to_bg[opt_id] is not None:
            bandgap = uuid_to_bg[opt_id]
            matched_mc2d = uuid_to_mc2d.get(opt_id, "")
            match_reason = "uuid_match"
        else:
            candidates = by_rf.get(rf, []) if rf else []
            if len(candidates) == 1:
                cand = candidates[0]
                bandgap = cand.get("bg")
                matched_mc2d = cand.get("id", "")
                match_reason = "formula_unique"
            elif len(candidates) > 1:
                bandgaps_list = []
                mc2d_list = []
                for c in candidates:
                    if c.get("bg") is not None:
                        bandgaps_list.append(str(c.get("bg")))
                    if c.get("id"):
                        mc2d_list.append(str(c.get("id")))
                if bandgaps_list:
                    bandgap = ";".join(bandgaps_list)
                    matched_mc2d = ";".join(mc2d_list)
                    match_reason = "ambiguous"
                else:
                    bandgap = None
                    match_reason = "not_found"
            else:
                bandgap = None
                match_reason = "not_found"

        if bandgap is None:
            bandgap = "NOT_FOUND"
            matched_mc2d = ""
        else:
            bandgap = str(bandgap)

        rows.append({
            "mc2d_id": opt_id,
            "formula": formula,
            "reduced_formula": rf or "",
            "nelements": item.get("nelements", ""),
            "bandgap": bandgap,
            "matched_mc2d_id": matched_mc2d,
            "match_reason": match_reason,
        })
        if (i + 1) % 100 == 0:
            LOGGER.info(f"Processed {i + 1}/{len(all_structs)}")

    rows.sort(key=lambda x: (x["reduced_formula"] or x["formula"], x["mc2d_id"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mc2d_id", "formula", "reduced_formula", "nelements", "bandgap", "matched_mc2d_id", "match_reason"])
        w.writeheader()
        w.writerows(rows)

    found = sum(1 for r in rows if r["bandgap"] != "NOT_FOUND" and ";" not in str(r["bandgap"]))
    ambiguous = sum(1 for r in rows if ";" in str(r["bandgap"]))
    LOGGER.info(f"Done. Saved {len(rows)} rows to {args.out} ({found} unique matches, {ambiguous} ambiguous)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
