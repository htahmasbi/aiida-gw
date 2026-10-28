#!/usr/bin/env python3
"""Build a comparison table of MC2D overview bandgaps vs AiiDA GW results."""
import argparse
import csv
import logging
import sys

from aiida.manage.configuration import load_profile
from aiida import orm

load_profile()

LOGGER = logging.getLogger("make_2el_bandgap_table")
WORKCHAIN_LABELS = {"GwWorkChain", "GWWorkChain", "gw_workchain"}

FAILURE_REASONS = {
    200: "ERROR_NO_RETRIEVED_FOLDER",
    210: "ERROR_OUTPUT_MISSING",
    300: "ERROR_OUTPUT_READ/INCOMPLETE",
    301: "ERROR_OUTPUT_READ",
    302: "ERROR_OUTPUT_PARSE",
    303: "ERROR_OUTPUT_INCOMPLETE",
    304: "ERROR_SOME_FILES_MISSING",
    350: "ERROR_UNEXPECTED_PARSER_EXCEPTION",
    400: "ERROR_OUT_OF_WALLTIME",
    450: "SCF_NOT_CONVERGED",
    500: "ERROR_GEOMETRY_CONVERGENCE_NOT_REACHED",
}


def _load_mc2d_lookup(csv_path):
    lookup_by_id = {}
    lookup_by_formula = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mc2d_id = (row.get("mc2d_id") or "").strip()
            formula = (row.get("formula") or "").strip()
            bg = (row.get("bandgap") or "").strip()
            if mc2d_id and bg and bg != "NOT_FOUND" and ";" not in bg:
                lookup_by_id[mc2d_id] = bg
            if formula and bg and bg != "NOT_FOUND" and ";" not in bg and formula not in lookup_by_formula:
                lookup_by_formula[formula] = bg
    LOGGER.info("Loaded MC2D overview: %d by mc2d_id, %d by formula", len(lookup_by_id), len(lookup_by_formula))
    return lookup_by_id, lookup_by_formula


def _get_structure(node):
    try:
        if "structure" in node.inputs:
            return node.inputs.structure
    except Exception:
        pass
    return None


def _get_formula(struct):
    if struct is None:
        return ""
    try:
        extras = struct.base.extras.all
        if extras.get("formula"):
            return extras["formula"]
        if extras.get("chemical_formula_reduced"):
            return extras["chemical_formula_reduced"]
    except Exception:
        pass
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


def _get_last_calcjob(node):
    try:
        calcs = [n for n in node.called_descendants if isinstance(n, orm.CalcJobNode)]
        if not calcs:
            return None
        return max(calcs, key=lambda c: c.ctime)
    except Exception:
        return None


def _get_gw_gaps(node):
    gaps = {}
    try:
        if "output_parameters" in node.outputs:
            out = node.outputs.output_parameters
            if isinstance(out, orm.Dict):
                d = out.get_dict()
            elif hasattr(out, "get_dict"):
                d = out.get_dict()
            else:
                return gaps
            for k in [
                "g0w0_gap_direct",
                "g0w0_gap_indirect",
                "scf_gap_direct",
                "scf_gap_indirect",
                "hf_gap_direct",
                "hf_gap_indirect",
                "g0w0_soc_gap_direct",
                "g0w0_soc_gap_indirect",
            ]:
                if k in d and d[k] is not None:
                    gaps[k] = d[k]
    except Exception:
        pass
    return gaps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mc2d-csv", required=True, help="CSV from fetch_mc2d_bandgaps_from_overview.py")
    parser.add_argument("--groups", nargs="+", default=["gw_runs"], help="AiiDA groups containing GwWorkChain nodes")
    parser.add_argument("--out", default="2elements_bandgap_comparison.csv", help="Output CSV")
    parser.add_argument("--only-2elements", action="store_true", help="Try to filter to 2-element formulas")
    parser.add_argument("--all-workflows", action="store_true", help="Include all WorkflowNode regardless of label")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    lookup_by_id, lookup_by_formula = _load_mc2d_lookup(args.mc2d_csv)

    rows = []
    for gname in args.groups:
        try:
            group = orm.Group.collection.get(label=gname)
        except Exception as exc:
            LOGGER.warning("Group %s not found: %s", gname, exc)
            continue
        LOGGER.info("Scanning group %s (%d nodes)", gname, len(group.nodes))
        for node in group.nodes:
            if not isinstance(node, orm.WorkflowNode):
                continue
            if not args.all_workflows and node.process_label not in WORKCHAIN_LABELS:
                continue

            struct = _get_structure(node)
            formula = _get_formula(struct)
            mc2d_id = _get_mc2d_id(struct)
            nelements = ""
            try:
                if struct:
                    nelements = len(struct.get_formula_dict())
            except Exception:
                pass

            if args.only_2elements and nelements != "2":
                continue

            bg_mc2d = lookup_by_id.get(mc2d_id) or lookup_by_formula.get(formula) or ""

            gw_gaps = _get_gw_gaps(node)
            last_calc = _get_last_calcjob(node)
            exit_status = node.exit_status if node.exit_status is not None else (last_calc.exit_status if last_calc else "")
            state = node.process_state.value if node.process_state else ""

            failure_reason = ""
            if exit_status and int(exit_status) != 0:
                failure_reason = FAILURE_REASONS.get(int(exit_status), f"EXIT_{exit_status}")

            rows.append(
                {
                    "gw_pk": node.pk,
                    "mc2d_id": mc2d_id,
                    "formula": formula,
                    "nelements": nelements,
                    "group": gname,
                    "bandgap_mc2d_overview_eV": bg_mc2d,
                    "g0w0_gap_direct_eV": gw_gaps.get("g0w0_gap_direct", ""),
                    "g0w0_gap_indirect_eV": gw_gaps.get("g0w0_gap_indirect", ""),
                    "scf_gap_direct_eV": gw_gaps.get("scf_gap_direct", ""),
                    "scf_gap_indirect_eV": gw_gaps.get("scf_gap_indirect", ""),
                    "hf_gap_direct_eV": gw_gaps.get("hf_gap_direct", ""),
                    "hf_gap_indirect_eV": gw_gaps.get("hf_gap_indirect", ""),
                    "state": state,
                    "exit_status": exit_status,
                    "failure_reason": failure_reason,
                    "last_calcjob_pk": last_calc.pk if last_calc else "",
                    "label": node.label or "",
                }
            )

    fieldnames = [
        "gw_pk",
        "mc2d_id",
        "formula",
        "nelements",
        "group",
        "bandgap_mc2d_overview_eV",
        "g0w0_gap_direct_eV",
        "g0w0_gap_indirect_eV",
        "scf_gap_direct_eV",
        "scf_gap_indirect_eV",
        "hf_gap_direct_eV",
        "hf_gap_indirect_eV",
        "state",
        "exit_status",
        "failure_reason",
        "last_calcjob_pk",
        "label",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    LOGGER.info("Wrote %d rows to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
