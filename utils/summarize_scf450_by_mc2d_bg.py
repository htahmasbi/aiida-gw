#!/usr/bin/env python3
"""Summarize SCF-not-converged (exit 450) GwWorkChains by MC2D bandgap."""

import argparse
import csv
import json
import logging
import sys

from aiida.manage.configuration import load_profile

load_profile()

from aiida.orm import CalcJobNode, Dict, Group, QueryBuilder, StructureData, WorkflowNode

LOGGER = logging.getLogger(__name__)


def _last_calcjob(node):
    calcs = [n for n in node.called_descendants if isinstance(n, orm.CalcJobNode)]
    return max(calcs, key=lambda c: c.ctime) if calcs else None


def _get_structure(node):
    try:
        inp = node.inputs
        for name in ["structure", "input_structure", "initial_structure", "base_structure"]:
            try:
                if name in inp:
                    s = inp[name]
                    if isinstance(s, StructureData):
                        return s
            except Exception:
                continue
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
    return ""


def _get_mc2d_id(struct):
    if struct is None:
        return ""
    try:
        extras = struct.base.extras.all
        for key in ["optimade_id", "mc2d_id", "id", "material_id"]:
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
                try:
                    bg_float = float(bg) if bg not in ["", None] else None
                except Exception:
                    bg_float = None
                if mc2d_id and bg_float is not None:
                    lookup_by_id[mc2d_id] = bg_float
                if formula and bg_float is not None:
                    lookup_by_formula[formula] = bg_float
                    try:
                        from pymatgen.core import Composition
                        norm = Composition(formula).reduced_formula
                        if norm:
                            lookup_by_formula[norm] = bg_float
                    except Exception:
                        pass
    except Exception as exc:
        LOGGER.warning("Failed to load MC2D CSV %s: %s", csv_path, exc)
    return lookup_by_id, lookup_by_formula


def collect_scf450(mc2d_csv, groups=None, all_groups=False):
    lookup_by_id, lookup_by_formula = _load_mc2d_lookup(mc2d_csv)
    
    if all_groups:
        try:
            groups_to_scan = [g.label for g in Group.collection.all()]
        except Exception:
            groups_to_scan = groups or []
    else:
        groups_to_scan = groups or ["gw_runs_2el8"]
    
    results = []
    
    for gname in groups_to_scan:
        try:
            group = Group.collection.get(label=gname)
        except Exception:
            continue
        for node in group.nodes:
            try:
                if not isinstance(node, WorkflowNode):
                    continue
                if node.process_label != "GwWorkChain":
                    continue
                if not node.is_finished:
                    continue
            except Exception:
                continue
            calc = _last_calcjob(node)
            if not calc or calc.exit_status != 450:
                continue
            
            struct = _get_structure(node)
            formula = _get_formula(struct)
            mc2d_id = _get_mc2d_id(struct)
            bg_mc2d = lookup_by_id.get(mc2d_id)
            if bg_mc2d is None:
                bg_mc2d = lookup_by_formula.get(formula)
            try:
                from pymatgen.core import Composition
                if bg_mc2d is None and formula:
                    bg_mc2d = lookup_by_formula.get(Composition(formula).reduced_formula)
            except Exception:
                pass
            
            results.append({
                "gw_pk": node.pk,
                "calc_pk": calc.pk,
                "group": gname,
                "formula": formula,
                "mc2d_id": mc2d_id,
                "mc2d_bg": bg_mc2d,
            })
    
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", help="Group label")
    parser.add_argument("--groups", nargs="+", help="Group labels")
    parser.add_argument("--all-groups", action="store_true", help="Scan all groups")
    parser.add_argument("--mc2d-csv", required=True, help="MC2D CSV file")
    parser.add_argument("--csv", help="Write detailed CSV")
    parser.add_argument("--json", help="Write JSON")
    args = parser.parse_args(argv)
    
    groups = args.groups or []
    if args.group:
        groups.extend(args.group)
    
    results = collect_scf450(mc2d_csv=args.mc2d_csv, groups=groups, all_groups=args.all_groups)
    
    total = len(results)
    print(f"Total SCF-not-converged (exit 450): {total}")
    
    # Categorize by MC2D bg
    bg_zero = []
    bg_nonzero = []
    bg_missing = []
    
    for r in results:
        bg = r["mc2d_bg"]
        if bg is None:
            bg_missing.append(r)
        elif abs(float(bg)) < 1e-6:
            bg_zero.append(r)
        else:
            bg_nonzero.append(r)
    
    print(f"\nMC2D bandgap == 0.0: {len(bg_zero)}")
    print(f"MC2D bandgap > 0.0: {len(bg_nonzero)}")
    print(f"MC2D bandgap missing: {len(bg_missing)}")
    
    print("\n=== MC2D bg == 0.0 ===")
    for r in bg_zero:
        print(f"  PK {r['gw_pk']} ({r['calc_pk']}) {r['formula']} mc2d_bg={r['mc2d_bg']} [{r['group']}]")
    
    print("\n=== MC2D bg > 0.0 ===")
    for r in bg_nonzero:
        print(f"  PK {r['gw_pk']} ({r['calc_pk']}) {r['formula']} mc2d_bg={r['mc2d_bg']} [{r['group']}]")
    
    print("\n=== MC2D bg missing ===")
    for r in bg_missing:
        print(f"  PK {r['gw_pk']} ({r['calc_pk']}) {r['formula']} mc2d_id={r['mc2d_id']} [{r['group']}]")
    
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["gw_pk", "calc_pk", "group", "formula", "mc2d_id", "mc2d_bg"])
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote {args.csv}")
    
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
