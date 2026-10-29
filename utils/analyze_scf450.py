#!/usr/bin/env python3
"""Analyze SCF-not-converged runs - look for patterns by formula/MC2D bg."""

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict

from aiida.manage.configuration import load_profile
load_profile()

from aiida import orm
from aiida.orm import CalcJobNode, Dict, Group, StructureData, WorkflowNode

LOGGER = logging.getLogger(__name__)


def _last_calcjob(node):
    calcs = [n for n in node.called_descendants if isinstance(n, CalcJobNode)]
    return max(calcs, key=lambda c: c.ctime) if calcs else None


def _get_retrieved(calc):
    try:
        return [out.node for out in calc.base.links.get_outgoing() if out.link_label == "retrieved"]
    except Exception:
        return []


def _output_nodes(cj, label):
    if cj is None:
        return []
    try:
        return [link.node for link in cj.base.links.get_outgoing() if link.link_label == label]
    except Exception:
        return []


def _calcjob_params(cj):
    if cj is None:
        return {}
    candidates = _output_nodes(cj, "output_parameters")
    for node in candidates:
        try:
            if isinstance(node, Dict):
                d = node.get_dict()
                if d:
                    return d
            elif hasattr(node, "get_dict"):
                d = node.get_dict()
                if d:
                    return d
        except Exception:
            continue
    return {}


def _is_scf_not_converged(wc):
    calc = _last_calcjob(wc)
    if calc and calc.exit_status == 450:
        return True
    params = _calcjob_params(calc)
    if params:
        flags = params.get("physics_flags") or {}
        if "scf_converged" in flags and flags.get("scf_converged") is False:
            return True
        if flags.get("scf") and isinstance(flags.get("scf"), list):
            for issue in flags.get("scf"):
                if "converge" in str(issue).lower():
                    return True
    if calc:
        for retrieved in _get_retrieved(calc):
            try:
                content = retrieved.base.repository.get_object_content("aiida.out")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                if content and "SCF run NOT converged" in content:
                    return True
            except Exception:
                continue
    return False


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", help="Group label")
    parser.add_argument("--groups", nargs="+", help="Group labels")
    parser.add_argument("--all-groups", action="store_true", help="Scan all groups")
    parser.add_argument("--mc2d-csv", required=True, help="MC2D CSV file")
    parser.add_argument("--csv", help="Write detailed CSV")
    args = parser.parse_args(argv)

    groups = args.groups or []
    if args.group:
        groups.extend(args.group)

    lookup_by_id, lookup_by_formula = _load_mc2d_lookup(args.mc2d_csv)

    if args.all_groups:
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
            if not _is_scf_not_converged(node):
                continue
            calc = _last_calcjob(node)
            struct = _get_structure(node)
            formula = _get_formula(struct)
            mc2d_id = _get_mc2d_id(struct)
            bg_mc2d = lookup_by_id.get(mc2d_id)
            if bg_mc2d is None:
                bg_mc2d = lookup_by_formula.get(formula)
            try:
                from pymatgen.core import Composition
                if bg_mc2d is None and formula:
                    norm = Composition(formula).reduced_formula
                    bg_mc2d = lookup_by_formula.get(norm)
            except Exception:
                pass
            results.append({
                "gw_pk": node.pk,
                "calc_pk": calc.pk if calc else None,
                "group": gname,
                "formula": formula,
                "mc2d_id": mc2d_id,
                "mc2d_bg": bg_mc2d,
                "wc_exit": node.exit_status,
            })

    print(f"Total SCF-not-converged: {len(results)}")

    # by mc2d bg
    zero = [r for r in results if r["mc2d_bg"] is not None and abs(float(r["mc2d_bg"])) < 1e-6]
    nonzero = [r for r in results if r["mc2d_bg"] is not None and abs(float(r["mc2d_bg"])) >= 1e-6]
    missing = [r for r in results if r["mc2d_bg"] is None]

    print(f"mc2d_bg == 0: {len(zero)}, mc2d_bg > 0: {len(nonzero)}, missing: {len(missing)}")

    print("\n=== Patterns by formula (top 30) ===")
    # Get elements
    from pymatgen.core import Composition
    elem_counter = Counter()
    elem_counter_zero = Counter()
    elem_counter_nonzero = Counter()
    for r in results:
        try:
            comp = Composition(r["formula"])
            elems = tuple(sorted([e.symbol for e in comp.elements]))
            elem_counter[elems] += 1
            if r["mc2d_bg"] is not None and abs(float(r["mc2d_bg"])) < 1e-6:
                elem_counter_zero[elems] += 1
            elif r["mc2d_bg"] is not None and abs(float(r["mc2d_bg"])) >= 1e-6:
                elem_counter_nonzero[elems] += 1
        except Exception:
            continue

    print("All SCF-not-converged:")
    for elems, cnt in elem_counter.most_common(30):
        print(f"  {''.join(elems)}: {cnt} (zero:{elem_counter_zero[elems]}, nonzero:{elem_counter_nonzero[elems]})")

    print("\n=== Elements appearing ===")
    all_elems = Counter()
    zero_elems = Counter()
    nonzero_elems = Counter()
    for r in results:
        try:
            comp = Composition(r["formula"])
            for e in comp.elements:
                all_elems[e.symbol] += 1
                if r["mc2d_bg"] is not None and abs(float(r["mc2d_bg"])) < 1e-6:
                    zero_elems[e.symbol] += 1
                elif r["mc2d_bg"] is not None and abs(float(r["mc2d_bg"])) >= 1e-6:
                    nonzero_elems[e.symbol] += 1
        except Exception:
            continue
    print("All:")
    for e, c in all_elems.most_common(20):
        print(f"  {e}: {c} (zero:{zero_elems[e]}, nonzero:{nonzero_elems[e]})")

    print("\n=== Binary systems with mc2d_bg > 0 (need fixing) ===")
    for r in nonzero:
        print(f"  PK {r['gw_pk']} {r['formula']} mc2d_bg={r['mc2d_bg']}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["gw_pk", "calc_pk", "group", "formula", "mc2d_id", "mc2d_bg", "wc_exit"])
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote {args.csv}")

    return 0


if __name__ == "__main__":
    main()
