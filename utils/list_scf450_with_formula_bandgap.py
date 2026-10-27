#!/usr/bin/env python3
"""List GwWorkChains that failed with SCF not converged (exit 450) with structure info.

Shows: GwWorkChain PK, last Cp2kCalculation PK, formula, MC2D ID, band gap (if available)
and group membership.
"""

import argparse
import csv
import logging
import sys

from aiida.manage.configuration import load_profile
from aiida import orm

load_profile()

LOGGER = logging.getLogger("list_scf450")
WORKCHAIN_LABELS = {"GwWorkChain"}


def _last_calcjob(node):
    calcs = [n for n in node.called_descendants if isinstance(n, orm.CalcJobNode)]
    if not calcs:
        return None
    return max(calcs, key=lambda c: c.ctime)


def _get_structure(node):
    try:
        if "structure" in node.inputs:
            return node.inputs.structure
    except Exception:
        pass
    return None


def _get_bandgap(struct):
    if struct is None:
        return None
    try:
        extras = struct.base.extras.all
    except Exception:
        extras = {}
    for key in [
        "band_gap",
        "bandgap",
        "gap",
        "gap_indirect",
        "indirect_band_gap",
        "direct_band_gap",
        "electronic_band_gap",
        "hse_gap",
        "pbe_gap",
    ]:
        if key in extras and extras[key] is not None:
            return extras[key]
    for key in extras:
        if "gap" in key.lower():
            val = extras[key]
            if val is not None and isinstance(val, (int, float, str)):
                return val
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


def _get_optimade_id(struct):
    if struct is None:
        return ""
    try:
        extras = struct.base.extras.all
        for key in ["optimade_id", "mc2d_id", "id", "optimade-id"]:
            if extras.get(key):
                return extras[key]
    except Exception:
        pass
    return ""


def find_scf450(group=None, pks=None):
    results = []

    if pks:
        for pk in pks:
            try:
                node = orm.load_node(pk)
            except Exception as exc:
                LOGGER.warning("Could not load %s: %s", pk, exc)
                continue
            if node.process_label not in WORKCHAIN_LABELS:
                continue
            if not node.is_finished:
                continue
            calc = _last_calcjob(node)
            if calc and calc.exit_status == 450:
                results.append((node, calc))
        return results

    qb = orm.QueryBuilder()
    qb.append(
        orm.WorkflowNode,
        tag="w",
        filters={
            "attributes.process_label": {"in": sorted(WORKCHAIN_LABELS)},
            "attributes.process_state": "finished",
            "attributes.exit_status": {">": 0},
        },
        project=["id"],
    )
    if group is not None:
        qb.append(orm.Group, filters={"id": group.pk}, with_node="w")

    for (pk,) in qb.all():
        node = orm.load_node(pk)
        calc = _last_calcjob(node)
        if not calc:
            continue
        if calc.exit_status == 450:
            results.append((node, calc))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", help="AiiDA group label to search in")
    parser.add_argument("--pks", nargs="+", type=int, help="Specific GwWorkChain PKs to check")
    parser.add_argument("--csv", help="Write results to CSV file")
    parser.add_argument("--all", action="store_true", help="Search across ALL groups")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.group and not args.pks and not args.all:
        parser.error("Provide --group, --pks, or --all")

    group = None
    if args.group:
        try:
            group = orm.Group.collection.get(label=args.group)
        except Exception as exc:
            LOGGER.error("Group '%s' not found: %s", args.group, exc)
            return 1

    pairs = find_scf450(group=group, pks=args.pks)
    if not pairs:
        LOGGER.info("No GwWorkChains found with exit_status 450.")
        return 0

    rows = []
    for node, calc in pairs:
        struct = _get_structure(node)
        formula = _get_formula(struct)
        mc2d_id = _get_optimade_id(struct)
        bandgap = _get_bandgap(struct)

        groups = []
        try:
            for g in node.base.groups:
                if g.label != node.label and not g.label.startswith("autogroup"):
                    groups.append(g.label)
        except Exception:
            pass

        rows.append(
            {
                "gw_pk": node.pk,
                "calcjob_pk": calc.pk,
                "formula": formula,
                "mc2d_id": mc2d_id,
                "bandgap": bandgap,
                "groups": ",".join(groups),
                "ctime": node.ctime.strftime("%Y-%m-%d %H:%M:%S") if node.ctime else "",
            }
        )

    rows.sort(key=lambda x: (x["formula"], x["gw_pk"]))

    print(f"{'GW PK':<8} {'CalcJob':<10} {'Formula':<12} {'MC2D ID':<30} {'Bandgap':<12} {'Group':<20}")
    print("-" * 90)
    for r in rows:
        bg = r["bandgap"] if r["bandgap"] is not None else "N/A"
        grp = r["groups"][:20]
        print(f"{r['gw_pk']:<8} {r['calcjob_pk']:<10} {r['formula']:<12} {r['mc2d_id']:<30} {str(bg):<12} {grp:<20}")

    print(f"\nTotal: {len(rows)}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        LOGGER.info("Wrote %s", args.csv)


if __name__ == "__main__":
    sys.exit(main())
