#!/usr/bin/env python3
"""List only original GwWorkChain runs (exclude resubmits/timeouts/retries)."""

import argparse
import logging
import sys

from aiida.manage.configuration import load_profile

load_profile()

from aiida import orm
from aiida.orm import Group, WorkflowNode

LOGGER = logging.getLogger(__name__)

WORKCHAIN_LABELS = {"GwWorkChain"}

EXCLUDE_PATTERNS = [
    "_r2", "_r3", "_resub", "resub", "timeout", "_retry", "retry", "scf450",
    "1test", "2test", "wx2", "test",
]


def is_excluded_group(name):
    n = name.lower()
    for p in EXCLUDE_PATTERNS:
        if p in n:
            return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mc2d-csv", help="MC2D CSV file (optional, just for info)")
    parser.add_argument("--groups", nargs="+", help="Explicit groups to include (overrides filter)")
    parser.add_argument("--csv", help="Write CSV")
    parser.add_argument("--all-groups", action="store_true", help="Consider all groups and filter")
    args = parser.parse_args(argv)

    if args.groups:
        groups_to_scan = args.groups
    elif args.all_groups:
        groups_to_scan = [g.label for g in Group.collection.all()]
    else:
        # Try to find gw_runs_2el* groups
        groups_to_scan = []
        for g in Group.collection.all():
            name = g.label
            if name.lower().startswith("gw_runs") and not is_excluded_group(name):
                groups_to_scan.append(name)
        groups_to_scan = sorted(groups_to_scan)

    LOGGER.info("Scanning groups: %s", groups_to_scan)

    rows = []
    for gname in groups_to_scan:
        try:
            group = Group.collection.get(label=gname)
        except Exception as exc:
            LOGGER.warning("Group %s not found: %s", gname, exc)
            continue
        for node in group.nodes:
            try:
                if not isinstance(node, WorkflowNode):
                    continue
                if node.process_label not in WORKCHAIN_LABELS and 'Gw' not in str(getattr(node, 'process_label', '')):
                    continue
            except Exception:
                continue
            # Check if this node came from resubmit - resubmits usually have different labels or are in resub groups
            # But we want original ones in original groups
            label = getattr(node, 'label', '') or ''
            rows.append({
                "gw_pk": node.pk,
                "group": gname,
                "label": label,
                "process_label": getattr(node, 'process_label', ''),
                "finished": node.is_finished,
                "exit_status": node.exit_status,
            })

    print(f"Found {len(rows)} original GwWorkChain nodes")
    # Group by group
    from collections import Counter
    c = Counter(r["group"] for r in rows)
    for g, cnt in sorted(c.items()):
        print(f"  {g}: {cnt}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["gw_pk", "group", "label", "process_label", "finished", "exit_status"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
