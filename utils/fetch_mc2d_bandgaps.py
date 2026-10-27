#!/usr/bin/env python3
"""Fetch band gaps from MC2D REST API for structures that failed with SCF not converged (exit 450)."""

import argparse
import csv
import json
import logging
import sys
import time

import requests
from aiida.manage.configuration import load_profile
from aiida import orm

load_profile()

LOGGER = logging.getLogger("fetch_mc2d_bandgaps")
WORKCHAIN_LABELS = {"GwWorkChain"}

MC2D_API = "https://mc2d.materialscloud.org/api"
HEADERS = {"User-Agent": "aiida-gw-bandgap-fetcher/1.0", "Accept": "application/json"}


def _last_calcjob(node):
    calcs = [n for n in node.called_descendants if isinstance(n, orm.CalcJobNode)]
    if not calcs:
        return None
    return max(calcs, key=lambda c: c.ctime)


def _find_gw_ancestor(node, depth=0):
    if depth > 10:
        return None
    try:
        for link in node.base.links.get_incoming():
            parent = link.node
            if not isinstance(parent, orm.WorkflowNode):
                continue
            if parent.process_label == "GwWorkChain":
                return parent
            found = _find_gw_ancestor(parent, depth + 1)
            if found:
                return found
    except Exception:
        pass
    return None


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


def _get_optimade_id(struct):
    if struct is None:
        return ""
    try:
        extras = struct.base.extras.all
        for key in ["optimade_id", "mc2d_id", "id", "material_id"]:
            if extras.get(key):
                return str(extras[key])
    except Exception:
        pass
    return ""


def _fetch_from_mc2d_by_formula(formula):
    if not formula:
        return None
    endpoints = [
        f"{MC2D_API}/v2/structures/?formula={formula}&limit=10",
        f"{MC2D_API}/structures?formula={formula}&limit=10",
    ]
    for url in endpoints:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                continue
            data = resp.json()
            items = []
            if isinstance(data, dict):
                for k in ["data", "results", "structures", "items"]:
                    if k in data and isinstance(data[k], list):
                        items = data[k]
                        break
            elif isinstance(data, list):
                items = data
            for item in items:
                for key in ["band_gap", "bandgap", "gap", "Eg", "energy_gap"]:
                    if key in item and item[key] is not None:
                        return {"gap": item[key], "source": url}
                if "properties" in item and isinstance(item["properties"], dict):
                    for key in ["band_gap", "bandgap", "gap"]:
                        if key in item["properties"] and item["properties"][key] is not None:
                            return {"gap": item["properties"][key], "source": url}
        except Exception as e:
            LOGGER.debug("Failed %s: %s", url, e)
            continue
    return None


def find_scf450_gws():
    qb = orm.QueryBuilder()
    qb.append(orm.CalcJobNode, filters={'attributes.exit_status': 450}, project=['id'])
    seen = set()
    results = []
    for (cid,) in qb.all():
        try:
            calc = orm.load_node(cid)
            gw = _find_gw_ancestor(calc)
            if not gw or gw.process_label != 'GwWorkChain':
                continue
            if not gw.is_finished or gw.pk in seen:
                continue
            last = _last_calcjob(gw)
            if not last:
                continue
            results.append((gw, last))
            seen.add(gw.pk)
        except Exception:
            continue
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="mc2d_bandgaps_scf450.csv", help="Output CSV file")
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between requests")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pairs = find_scf450_gws()
    LOGGER.info("Found %d GwWorkChains with exit 450", len(pairs))

    rows = []
    for gw, calc in pairs:
        struct = _get_structure(gw)
        formula = _get_formula(struct)
        mc2d_id = _get_optimade_id(struct)
        
        LOGGER.info("Fetching for GW %s | %s | %s", gw.pk, formula, mc2d_id or "no-id")
        res = _fetch_from_mc2d_by_formula(formula)
        gap = None
        source = None
        if res:
            gap = res.get("gap")
            source = res.get("source")
        else:
            gap = "NOT_FOUND"
            source = "N/A"
        
        rows.append({
            "gw_pk": gw.pk,
            "calcjob_pk": calc.pk,
            "formula": formula,
            "mc2d_id": mc2d_id,
            "bandgap": gap,
            "source_api": source,
        })
        time.sleep(args.sleep)

    rows.sort(key=lambda x: (str(x["bandgap"]), x["formula"]))
    
    print(f"\n{'GW PK':<8} {'Formula':<12} {'MC2D ID':<25} {'Bandgap':<15}")
    print("-" * 65)
    for r in rows:
        print(f"{r['gw_pk']:<8} {r['formula']:<12} {r['mc2d_id']:<25} {str(r['bandgap']):<15}")
    
    print(f"\nTotal: {len(rows)}")
    
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    LOGGER.info("Saved to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
