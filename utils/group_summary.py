#!/usr/bin/env python3
"""Summarize the workflow outcome of every AiiDA Group.

One row per group: how many nodes it holds and how many of its top-level
workchains ended Ok / failed / Excepted / Killed / are still running. This is
the cross-group view that ``verdi process list --group`` only provides one
group at a time.

Usage:
    python group_summary.py                        # every group of the default user
    python group_summary.py --contains gw_         # only groups whose label matches
    python group_summary.py --label gw_runs_2el6   # specific groups only
    python group_summary.py --nonempty             # skip groups with no workchains
    python group_summary.py --csv summary.csv      # additionally write CSV
"""

import argparse
import csv
import sys
from collections import Counter

from aiida.manage.configuration import load_profile

load_profile()

from aiida import orm

FINISHED = "finished"
STATE_ORDER = ("created", "waiting", "running", "ok", "failed", "excepted", "killed", "crashed", "unknown")


def _group_filters(args):
    """Label filter shared by both queries."""
    from aiida.common.escaping import escape_for_sql_like

    if args.label:
        return {"label": {"in": args.label}}
    if args.startswith:
        return {"label": {"like": f"{escape_for_sql_like(args.startswith)}%"}}
    if args.contains:
        return {"label": {"like": f"%{escape_for_sql_like(args.contains)}%"}}
    return {}


def _classify(state, exit_status):
    """Map (process_state, exit_status) onto a single outcome bucket."""
    if state == FINISHED:
        return "ok" if not exit_status else "failed"
    if state is None:
        return "unknown"
    return state


def collect(args):
    """Return one dict per group with per-outcome counts."""
    filters = _group_filters(args)

    groups = {
        pk: label
        for pk, label in orm.QueryBuilder()
        .append(orm.Group, filters=filters, project=["id", "label"])
        .all()
    }
    if not groups:
        return []

    node_filters = {}
    if args.process_label:
        node_filters["attributes.process_label"] = {"in": args.process_label}

    builder = orm.QueryBuilder()
    builder.append(orm.Group, filters={"id": {"in": sorted(groups)}}, tag="g", project="id")
    builder.append(
        orm.WorkflowNode,
        with_group="g",
        filters=node_filters,
        project=["attributes.process_state", "attributes.exit_status"],
    )

    counts = {pk: Counter() for pk in groups}
    for group_pk, state, exit_status in builder.all():
        counts[group_pk][_classify(state, exit_status)] += 1

    rows = []
    for pk, label in sorted(groups.items(), key=lambda item: item[1]):
        counter = counts[pk]
        row = {
            "pk": pk,
            "label": label,
            "nodes": orm.load_group(pk).count(),
            "wf": sum(counter.values()),
        }
        row.update({state: counter.get(state, 0) for state in STATE_ORDER})
        rows.append(row)

    if args.nonempty:
        rows = [row for row in rows if row["wf"]]
    return rows


def print_table(rows):
    columns = [
        ("wf", 5),
        ("nodes", 6),
        ("created", 7),
        ("waiting", 7),
        ("running", 7),
        ("ok", 5),
        ("failed", 6),
        ("excepted", 8),
        ("killed", 6),
        ("crashed", 7),
    ]
    header = f"{'PK':>4}  {'label':<26} " + " ".join(f"{name:>{width}}" for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = " ".join(f"{row[name]:>{width}}" for name, width in columns)
        print(f"{row['pk']:>4}  {row['label'][:26]:<26} {cells}")

    if len(rows) > 1:
        totals = {name: sum(row[name] for row in rows) for name, _ in columns}
        cells = " ".join(f"{totals[name]:>{width}}" for name, width in columns)
        print(f"{'':>4}  {'TOTAL':<26} {cells}")


def write_csv(rows, path):
    if not rows:
        print(f"No data, not writing {path}")
        return
    fields = ["pk", "label", "nodes", "wf"] + list(STATE_ORDER)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", nargs="+", help="only these group labels")
    parser.add_argument("--startswith", help="only groups whose label starts with STRING")
    parser.add_argument("--contains", help="only groups whose label contains STRING")
    parser.add_argument("--process-label", nargs="+", help="only workchains with these process labels")
    parser.add_argument("--nonempty", action="store_true", help="skip groups that hold no workflow node")
    parser.add_argument("--csv", metavar="PATH", help="also write results to CSV file")
    args = parser.parse_args(argv)

    rows = collect(args)
    if not rows:
        print("No matching groups found.")
        return 1
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
