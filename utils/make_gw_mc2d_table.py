#!/usr/bin/env python3
"""Compare GW results from GwWorkChain with MC2D bandgaps.

Similar to extract_gw.py - extracts gaps from last CalcJobNode's output_parameters.
"""

import argparse
import csv
import json
import logging
import sys

from aiida.manage.configuration import load_profile

load_profile()

from aiida.orm import CalcJobNode, Dict, Group, ProcessNode, QueryBuilder, StructureData, WorkflowNode, WorkChainNode

LOGGER = logging.getLogger(__name__)

WORKCHAIN_LABELS = {"GwWorkChain"}

GAP_KEYS = {
    "scf_gap_indirect": "scf_gap_indirect",
    "scf_soc_gap_indirect": "scf_soc_gap_indirect",
    "g0w0_gap_indirect": "g0w0_gap_indirect",
    "g0w0_soc_gap_indirect": "g0w0_soc_gap_indirect",
    "hf_gap_direct": "hf_gap_direct",
}

FAILURE_REASONS = {
    200: "ERROR_CALCJOB_INVALID_INPUT",
    210: "ERROR_NO_RETRIEVED_FOLDER",
    300: "ERROR_OUTPUT_READ/INCOMPLETE",
    301: "ERROR_OUTPUT_PARSING_FAILED",
    302: "ERROR_PARSING_MISSING_SOURCE",
    303: "ERROR_PARSING_MISSING_OBJECT",
    304: "ERROR_PARSING_EMPTY_SOURCE",
    350: "ERROR_UNSUPPORTED_PARSING",
    400: "ERROR_CALCJOB_SUBMISSION_FAILED",
    450: "ERROR_CALCJOB_INTERRUPT_EXCEPTION",
    500: "ERROR_UNKNOWN",
}


def _last_calcjob(wc):
    calcs = [n for n in wc.called_descendants if isinstance(n, CalcJobNode)]
    return max(calcs, key=lambda c: c.ctime) if calcs else None


def _output_nodes(cj, label):
    if cj is None:
        return []
    try:
        return [link.node for link in cj.base.links.get_outgoing() if link.link_label == label]
    except Exception:
        return []


def _calcjob_params(wc):
    cj = _last_calcjob(wc)
    if cj is None:
        return None, {}, "no_calcjob"

    candidates = _output_nodes(cj, "output_parameters")
    if not candidates:
        return cj.pk, {}, "no_link"

    errors = []
    for node in candidates:
        try:
            params = node.get_dict()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
            continue
        if params:
            return cj.pk, params, ""
    if errors:
        return cj.pk, {}, "read_error:" + ";".join(errors)
    return cj.pk, {}, "empty"


def _get_structure(wc):
    # Try to get structure from inputs
    try:
        inp = wc.inputs
        for name in ["structure", "input_structure", "initial_structure", "base_structure"]:
            try:
                if name in inp:
                    s = inp[name]
                    if isinstance(s, StructureData):
                        return s
            except Exception:
                continue
        # search recursively
        def search(d, depth=0):
            if depth > 4:
                return None
            try:
                for k in d:
                    try:
                        v = d[k]
                        if isinstance(v, StructureData):
                            return v
                        r = search(v, depth + 1)
                        if r is not None:
                            return r
                    except Exception:
                        continue
            except Exception:
                pass
            return None
        return search(inp)
    except Exception:
        pass
    return None


def _get_formula(struct):
    if struct is None:
        return ""
    try:
        return struct.get_formula()
    except Exception:
        pass
    try:
        return struct.get_formula(mode="hill")
    except Exception:
        pass
    return ""


def _get_mc2d_id(struct):
    if struct is None:
        return ""
    try:
        extras = struct.base.extras.all
        for key in ["optimade_id", "mc2d_id", "id", "material_id", "mc2d-id", "mcloud_id"]:
            if extras.get(key):
                return str(extras[key]).strip()
    except Exception:
        pass
    return ""


def _load_mc2d_lookup(csv_path):
    lookup_by_id = {}
    lookup_by_formula = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mc2d_id = (row.get("mc2d_id") or row.get("id") or row.get("material_id") or "").strip()
                formula = (row.get("formula") or row.get("Formula") or "").strip()
                bg = row.get("bandgap") or row.get("bg") or row.get("band_gap") or row.get("bandgap_mc2d_overview_eV") or row.get("E_g") or ""
                if mc2d_id and bg != "":
                    lookup_by_id[mc2d_id] = bg
                if formula and bg != "":
                    lookup_by_formula[formula] = bg
                    # try normalized
                    try:
                        from pymatgen.core import Composition
                        norm = Composition(formula).reduced_formula
                        if norm:
                            lookup_by_formula[norm] = bg
                    except Exception:
                        pass
    except Exception as exc:
        LOGGER.warning("Failed to load MC2D CSV %s: %s", csv_path, exc)
    return lookup_by_id, lookup_by_formula


def collect_rows(mc2d_csv, groups, all_workflows=False, debug=False):
    lookup_by_id, lookup_by_formula = _load_mc2d_lookup(mc2d_csv)
    rows = []
    
    for gname in groups:
        try:
            group = Group.collection.get(label=gname)
        except Exception as exc:
            LOGGER.warning("Group %s not found: %s", gname, exc)
            continue
        LOGGER.info("Scanning group %s (%d nodes)", gname, len(group.nodes))
        
        for wc in group.nodes:
            try:
                if not isinstance(wc, (WorkflowNode, WorkChainNode)):
                    if debug:
                        print(f"skip non-workflow {wc.pk} {type(wc)}")
                    continue
            except Exception:
                continue
            
            pl = getattr(wc, 'process_label', '') or ''
            if not all_workflows and pl not in WORKCHAIN_LABELS and 'Gw' not in str(pl) and 'GW' not in str(pl):
                if debug:
                    print(f"skip label {wc.pk} {pl}")
                continue
            
            calc_pk, params, note = _calcjob_params(wc)
            if note:
                physics_issues = note
            else:
                physics_issues = ""  # we just want gaps
            
            struct = _get_structure(wc)
            formula = _get_formula(struct)
            mc2d_id = _get_mc2d_id(struct)
            nelements = ""
            try:
                if struct:
                    nelements = len(struct.get_formula_dict())
            except Exception:
                pass
            
            bg_mc2d = lookup_by_id.get(mc2d_id) or lookup_by_formula.get(formula) or ""
            try:
                from pymatgen.core import Composition
                if not bg_mc2d and formula:
                    bg_mc2d = lookup_by_formula.get(Composition(formula).reduced_formula) or ""
            except Exception:
                pass
            
            wall_seconds = None
            try:
                if wc.is_finished:
                    wall_seconds = (wc.mtime - wc.ctime).total_seconds()
            except Exception:
                pass
            
            row = {
                "gw_pk": wc.pk,
                "calc_pk": calc_pk,
                "mc2d_id": mc2d_id,
                "formula": formula,
                "nelements": nelements,
                "group": gname,
                "bandgap_mc2d_overview_eV": bg_mc2d,
                "g0w0_gap_direct_eV": params.get("g0w0_gap_direct"),
                "g0w0_gap_indirect_eV": params.get("g0w0_gap_indirect"),
                "g0w0_soc_gap_direct_eV": params.get("g0w0_soc_gap_direct"),
                "g0w0_soc_gap_indirect_eV": params.get("g0w0_soc_gap_indirect"),
                "scf_gap_direct_eV": params.get("scf_gap_direct"),
                "scf_gap_indirect_eV": params.get("scf_gap_indirect"),
                "scf_soc_gap_direct_eV": params.get("scf_soc_gap_direct"),
                "scf_soc_gap_indirect_eV": params.get("scf_soc_gap_indirect"),
                "hf_gap_direct_eV": params.get("hf_gap_direct"),
                "hf_gap_indirect_eV": params.get("hf_gap_indirect"),
                "g0w0_vbm": params.get("g0w0_vbm"),
                "g0w0_cbm": params.get("g0w0_cbm"),
                "energy": params.get("energy"),
                "nwarnings": params.get("nwarnings"),
                "status": "running" if not wc.is_finished else ("ok" if wc.is_finished_ok else f"exit_{wc.exit_status}"),
                "exit_status": wc.exit_status,
                "failure_reason": FAILURE_REASONS.get(wc.exit_status, "") if wc.exit_status is not None and wc.exit_status != 0 else "",
                "physics_issues": physics_issues,
                "wall_h": f"{wall_seconds / 3600:.2f}" if wall_seconds is not None else "-",
                "label": getattr(wc, 'label', '') or "",
            }
            rows.append(row)
    
    return rows


def print_table(rows):
    if not rows:
        print("No data")
        return
    figs = ["scf_gap_indirect_eV", "scf_soc_gap_indirect_eV", "g0w0_gap_indirect_eV", "g0w0_soc_gap_indirect_eV", "hf_gap_direct_eV"]
    header = (
        f"{'PK':>7}  {'calc':>7} {'status':<9} {'physics':<48} {'wall_h':>7}  "
        + " ".join(f"{c.replace('_eV',''):>12}" for c in figs)
        + f" {'mc2d_bg':>10} {'energy':>12}  {'formula':<14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        physics = "ok" if not row["physics_issues"] or str(row["physics_issues"]) == "" else "BAD"
        if row["physics_issues"] and str(row["physics_issues"]) != "":
            physics = "BAD:" + str(row["physics_issues"])[:44]
        def fmt(v):
            if v is None or v == "":
                return " " * 11 + "-"
            try:
                return f"{float(v):>12.3f}"
            except Exception:
                return " " * 11 + "-"
        calc = row["calc_pk"] if row["calc_pk"] is not None else "-"
        print(
            f"{row['gw_pk']:>7}  {calc:>7}  {row['status']:<9} {physics:<48} {row['wall_h']:>7}  "
            + " ".join(fmt(row.get(c)) for c in figs)
            + f" {fmt(row.get('bandgap_mc2d_overview_eV'))} {fmt(row.get('energy'))}  {str(row['formula'])[:14]:<14}"
        )


def write_csv(rows, path):
    if not rows:
        print(f"No data, not writing {path}")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", help="Group label (can use multiple times)")
    parser.add_argument("--groups", nargs="+", help="Group labels")
    parser.add_argument("--mc2d-csv", help="MC2D CSV file")
    parser.add_argument("--all-workflows", action="store_true", help="Include all WorkflowNode")
    parser.add_argument("--csv", help="Write CSV")
    parser.add_argument("--debug", action="store_true", help="Debug")
    args = parser.parse_args(argv)
    
    groups = args.groups or []
    if args.group:
        groups.extend(args.group)
    if not groups:
        groups = ["gw_runs_2el8"]
    
    mc2d_csv = args.mc2d_csv or ""
    
    rows = collect_rows(mc2d_csv, groups, all_workflows=args.all_workflows, debug=args.debug)
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
