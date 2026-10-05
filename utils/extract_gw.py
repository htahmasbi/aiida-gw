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
PHYSICS_LEVELS = ("scf", "scf_soc", "g0w0", "g0w0_soc", "hf")


def _last_calcjob(wc):
    calcs = [n for n in wc.called_descendants if isinstance(n, CalcJobNode)]
    return max(calcs, key=lambda c: c.ctime) if calcs else None


def _calcjob_params(wc):
    """Return (calc_pk, params, note) from the workchain's last CalcJobNode.

    ``note`` records *why* no parameters could be read, so a run whose results
    exist but could not be deserialized is never reported as data-less:

      ``""``           - parameters read successfully
      ``"no_calcjob"`` - the workchain has no CalcJobNode descendant
      ``"no_link"``    - the calcjob never stored output_parameters
      ``"empty"``      - output_parameters exists but the Dict holds nothing
      ``"read_error"`` - resolving or deserializing the Dict raised; the
                         exception type and message are appended

    A bare ``except`` here previously turned every failure into an empty dict,
    which made healthy runs look like ``BAD:no_data``.
    """
    cj = _last_calcjob(wc)
    if cj is None:
        return None, {}, "no_calcjob"
    try:
        node = cj.outputs.output_parameters
    except Exception as exc:
        return cj.pk, {}, f"read_error:{type(exc).__name__}:{exc}"
    try:
        params = node.get_dict()
    except Exception as exc:
        return cj.pk, {}, f"read_error:{type(exc).__name__}:{exc}"
    if not params:
        return cj.pk, {}, "empty"
    return cj.pk, params, ""


def _physics_summary(params):
    """Return (ok, issues) for a run from its physics_flags or stored gaps.

    Runs re-parsed with the current code carry an explicit physics_flags dict;
    older runs are judged from the stored gap values (negative/zero gap ->
    physically suspect) so they are not silently reported as clean.
    A run with no stored physics data at all is reported as BAD:no_data,
    never as clean.
    """
    if not params:
        return False, "no_data"
    flags = params.get("physics_flags") or {}
    issues = []
    if "physics_ok" in flags:
        ok = bool(flags.get("physics_ok"))
        for level in PHYSICS_LEVELS:
            for issue in flags.get(level) or []:
                issues.append(f"{level}:{issue}")
        if not flags.get("scf_converged", True):
            issues.append("scf_not_converged")
        if flags.get("aborted"):
            issues.append("aborted")
    else:
        ok = True
        for level in PHYSICS_LEVELS:
            gap_direct = params.get(f"{level}_gap_direct")
            gap_indirect = params.get(f"{level}_gap_indirect")
            vbm = params.get(f"{level}_vbm")
            cbm = params.get(f"{level}_cbm")
            if gap_direct is not None:
                if gap_direct < 0.0:
                    issues.append(f"{level}:negative_direct_gap")
                elif gap_direct < 1e-3:
                    issues.append(f"{level}:zero_direct_gap")
            if gap_indirect is not None:
                if gap_indirect < 0.0:
                    issues.append(f"{level}:negative_indirect_gap")
                elif gap_indirect < 1e-3:
                    issues.append(f"{level}:zero_indirect_gap")
            if vbm is not None and cbm is not None and cbm <= vbm:
                issues.append(f"{level}:cbm_at_or_below_vbm")
        ok = not issues
    return ok, ",".join(issues)


def _formula(wc):
    cj = _last_calcjob(wc)
    if cj is None:
        return ""
    try:
        return cj.outputs.output_structure.get_formula()
    except Exception:
        return ""


def collect_runs(pks=None, group_label=None, include_running=False, ok_only=False):
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
        calc_pk, params, note = _calcjob_params(wc)
        if note:
            physics_ok, physics_issues = False, note
        else:
            physics_ok, physics_issues = _physics_summary(params)
        if ok_only and not physics_ok:
            continue
        wall_seconds = None
        if not running:
            wall_seconds = (wc.mtime - wc.ctime).total_seconds()
        row = {
            "pk": wc.pk,
            "calc": calc_pk,
            "ctime": wc.ctime.isoformat(),
            "status": "running" if running else ("ok" if wc.is_finished_ok else f"exit_{wc.exit_status}"),
            "wall_s": wall_seconds,
            "label": wc.label,
            "formula": _formula(wc),
            "physics_ok": physics_ok,
            "physics_issues": physics_issues,
            "params_note": note,
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
    figs = list(GAP_KEYS.values()) + ["energy"]
    header = (
        f"{'PK':>7}  {'calc':>7} {'status':<9} {'physics':<48} {'wall_h':>7}  "
        + " ".join(f"{c:>10}" for c in figs)
        + f"  {'formula':<14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        wall = f"{row['wall_s'] / 3600:.2f}" if row["wall_s"] is not None else "-"
        physics = "ok" if row["physics_ok"] else "BAD"
        if row["physics_issues"]:
            physics = "BAD:" + row["physics_issues"][:44]
        values = []
        for key in figs:
            v = row.get(key)
            values.append(f"{v:>10.3f}" if isinstance(v, float) else " " * 9 + "-")
        calc = row["calc"] if row["calc"] is not None else "-"
        print(
            f"{row['pk']:>7}  {calc:>7}  {row['status']:<9} {physics:<48} {wall:>7}  "
            + " ".join(values)
            + f"  {row['formula'][:14]:<14}"
        )


def write_csv(rows, path):
    if not rows:
        print(f"No data, not writing {path}")
        return
    fields = ["pk", "calc", "label", "ctime", "status", "physics_ok", "physics_issues", "params_note", "wall_s", "formula", "nwarnings"]
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
    parser.add_argument("--ok-only", action="store_true", help="only show runs whose physics flags are clean")
    parser.add_argument("--csv", metavar="PATH", help="also write results to CSV file")
    parser.add_argument("--json", dest="json_path", metavar="PATH", help="also write results to JSON file")
    args = parser.parse_args(argv)

    rows = collect_runs(pks=args.pks, group_label=args.group, include_running=args.running, ok_only=args.ok_only)
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
