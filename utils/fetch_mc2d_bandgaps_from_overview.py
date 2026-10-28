#!/usr/bin/env python3
"""Fetch band gaps for all MC2D structures from Materials Cloud overview API."""
import argparse
import csv
import glob
import json
import logging
import os

import requests
from pymatgen.core import Composition, Structure

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


def _get_opt_structure(item):
    try:
        struct_dict = item.get("structure")
        if struct_dict:
            return Structure.from_dict(struct_dict)
    except Exception:
        pass
    return None


def _get_overview_structure(mc2d_id):
    try:
        url = f"https://mcxd-api.materialscloud.org/mc2d/pbe-v1/core_base/{mc2d_id}"
        resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Try to extract structure
        def find(d):
            if isinstance(d, dict):
                if "lattice" in d and ("sites" in d or "cartesian_site_positions" in d):
                    try:
                        return Structure.from_dict(d)
                    except Exception:
                        pass
                for v in d.values():
                    r = find(v)
                    if r is not None:
                        return r
            elif isinstance(d, list):
                for x in d:
                    r = find(x)
                    if r is not None:
                        return r
            return None
        return find(data)
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", default=".", help="Directory containing mc2d_*elements*.json files")
    parser.add_argument("--out", default="mc2d_bandgaps_all.csv", help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of structures (for testing)")
    parser.add_argument("--match-by-structure", action="store_true", default=True, help="Match by structure when multiple entries share formula")
    parser.add_argument("--no-match-by-structure", dest="match_by_structure", action="store_false")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overview = fetch_overview()

    # Group overview by reduced formula
    by_rf = {}
    for entry in overview:
        formula = entry.get("formula")
        rf = _norm_formula(formula)
        if not rf:
            continue
        by_rf.setdefault(rf, []).append(entry)

    LOGGER.info(f"Built groups for {len(by_rf)} unique formulas")

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
        bandgap = "NOT_FOUND"
        matched_mc2d = ""
        matched_reason = ""

        candidates = by_rf.get(rf, []) if rf else []
        if not candidates and formula:
            # try exact
            for entries in by_rf.values():
                for e in entries:
                    if e.get("formula") == formula:
                        candidates.append(e)

        if len(candidates) == 0:
            bandgap = "NOT_FOUND"
        elif len(candidates) == 1:
            bandgap = candidates[0].get("bg")
            matched_mc2d = candidates[0].get("id", "")
            matched_reason = "formula_unique"
        else:
            # Multiple candidates - try structure matching
            opt_struct = _get_opt_structure(item)
            best = None
            best_score = -1
            if opt_struct is not None and args.match_by_structure:
                for cand in candidates:
                    cand_id = cand.get("id")
                    cand_struct = _get_overview_structure(cand_id)
                    if cand_struct is None:
                        continue
                    try:
                        # Simple volume + lattice matching
                        if abs(opt_struct.volume - cand_struct.volume) / max(opt_struct.volume, 1e-10) < 0.05:
                            score = 10 - abs(opt_struct.volume - cand_struct.volume) / max(opt_struct.volume, 1e-10)
                            if score > best_score:
                                best_score = score
                                best = cand
                    except Exception:
                        continue
            if best is not None:
                bandgap = best.get("bg")
                matched_mc2d = best.get("id", "")
                matched_reason = "structure_match"
            else:
                # Fall back to listing all
                bandgaps_list = [str(c.get("bg")) for c in candidates if c.get("bg") is not None]
                mc2d_list = [str(c.get("id")) for c in candidates]
                bandgap = ";".join(bandgaps_list) if bandgaps_list else "AMBIGUOUS"
                matched_mc2d = ";".join(mc2d_list)
                matched_reason = "ambiguous"

        rows.append({
            "mc2d_id": mc2d_id,
            "formula": formula,
            "reduced_formula": rf or "",
            "nelements": item.get("nelements", ""),
            "bandgap": bandgap,
            "matched_mc2d_id": matched_mc2d,
            "match_reason": matched_reason,
        })
        if (i + 1) % 50 == 0:
            LOGGER.info(f"Processed {i + 1}/{len(all_structs)}")

    rows.sort(key=lambda x: (x["reduced_formula"] or x["formula"], x["mc2d_id"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mc2d_id", "formula", "reduced_formula", "nelements", "bandgap", "matched_mc2d_id", "match_reason"])
        w.writeheader()
        w.writerows(rows)

    found = sum(1 for r in rows if r["bandgap"] not in ("NOT_FOUND", "AMBIGUOUS") and ";" not in str(r["bandgap"]))
    LOGGER.info(f"Done. Saved {len(rows)} rows to {args.out} ({found}/{len(rows)} uniquely matched bandgaps found)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
