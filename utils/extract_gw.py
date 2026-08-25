#!/usr/bin/env python3
"""Extract GW calculation summaries (gaps, wall time, energies) from an AiiDA profile.

Tabulates every GwWorkChain in the database, similar in spirit to utils/extract.py,
but for GW band-structure results instead of MD training data.

Usage:
    python extract_gw.py                     # human-readable table of finished runs
    python extract_gw.py --running           # include still-running workchains
    python extract_gw.py --pks 358 421       # restrict to specific workchain PKs
    python extract_gw.py --group ht_runs     # only workchains in a given Group
    python extract_gw.py --csv gw.csv        # additionally write CSV
    python extract_gw.py --json gw.json      # additionally write JSON
"""

import argparse
import csv
import json
import sys

from aiida.manage.configuration import load_profile

load_profile()

from aiida.orm import CalcJobNode, Group, ProcessNode, QueryBuilder

WORKCHAIN_LABEL = "GwWorkChain"

GAP_KEYS = {
    "scf_gap_indirect": "scf_gap",
    "scf_soc_gap_indirect": "scf_soc_gap",
    "g0w0_gap_indirect": "g0w0_gap",
    "g0w0_soc_gap_indirect": "g0w0_soc_gap",
    "hf_gap_direct": "hf_gap",
}
EXTRA_PARAM_KEYS = ("g0w0_vbm", "g0w0_cbm", "energy", "nwarnings")


def _calcjob_params(wc):
    """Return the output_parameters dict of the workchain's first CalcJobNode."""
    try:
        cj = next(n for n in wc.called_descendants if isinstance(n, CalcJobNode))
        return cj.outputs.output_parameters.get_dict()
    except Exception:
        return {}


def _formula(wc):
    try:
        cj = next(n for n in wc.called_descendants if isinstance(n, CalcJobNode))
        return cj.outputs.output_structure.get_formula()
    except Exception:
        return ""


def collect_runs(pks=None, group_label=None, include_running=False):
    """Query the profile and return one summary dict per GwWorkChain."""
    builder = QueryBuilder().append(
        ProcessNode,
        filters={"attributes.process_label": WORKCHAIN_LABEL},
        tag="wc",
        project="*",
    )
    if pks:
        builder.add_filter("wc", {"id": {"in": list(pks)}})
    if group_label:
        builder.append(Group, filters={"label": group_label}, with_node="wc")
    builder.order_by({ProcessNode: {"ctime": "asc"}})

    rows = []
    for (wc,) in builder.all():
        running = not wc.is_finished
        if running and not include_running:
            continue
        params = _calcjob_params(wc)
        wall_seconds = None
        if not running:
            wall_seconds = (wc.mtime - wc.ctime).total_seconds()
        row = {
            "pk": wc.pk,
            "ctime": wc.ctime.isoformat(),
            "status": "running" if running else ("ok" if wc.is_finished_ok else f"exit_{wc.exit_status}"),
            "wall_s": wall_seconds,
            "label": wc.label,
            "formula": _formula(wc),
        }
        for key, col in GAP_KEYS.items():
            value = params.get(key)
            row[col] = float(value) if value is not None else None
        for key in EXTRA_PARAM_KEYS:
            value = params.get(key)
            row[key] = float(value) if isinstance(value, (int, float)) else value
        rows.append(row)
    return rows


def print_table(rows):
    cols = ["pk", "status", "wall_h"] + list(GAP_KEYS.values()) + ["energy", "formula"]
    header = f"{'PK':>7}  {'status':<9} {'wall_h':>7}  " + " ".join(f"{c:>10}" for c in list(GAP_KEYS.values()) + ["energy"]) + f"  {'formula':<14}"
    print(header)
    print("-" * len(header))
    for row in rows:
        wall = f"{row['wall_s'] / 3600:.2f}" if row["wall_s"] is not None else "-"
        values = []
        for key in list(GAP_KEYS.values()) + ["energy"]:
            v = row.get(key)
            values.append(f"{v:>10.3f}" if isinstance(v, float) else " " * 9 + "-")
        print(f"{row['pk']:>7}  {row['status']:<9} {wall:>7}  " + " ".join(values) + f"  {row['formula'][:14]:<14}")


def write_csv(rows, path):
    if not rows:
        print(f"No data, not writing {path}")
        return
    fields = ["pk", "label", "ctime", "status", "wall_s", "formula", "nwarnings"]
    fields += list(GAP_KEYS.values()) + ["g0w0_vbm", "g0w0_cbm", "energy"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def write_json(rows, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Wrote {path} ({len(rows)} rows)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pks", nargs="+", type=int, help="restrict to these workchain PKs")
    parser.add_argument("--group", help="only workchains belonging to this Group label")
    parser.add_argument("--running", action="store_true", help="include still-running workchains")
    parser.add_argument("--csv", metavar="PATH", help="also write results to CSV file")
    parser.add_argument("--json", dest="json_path", metavar="PATH", help="also write results to JSON file")
    args = parser.parse_args(argv)

    rows = collect_runs(pks=args.pks, group_label=args.group, include_running=args.running)
    if not rows:
        print("No matching GwWorkChain nodes found.")
        return 1
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    if args.json_path:
        write_json(rows, args.json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
