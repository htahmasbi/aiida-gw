#!/usr/bin/env python3
"""Resubmit GwWorkChains whose last Cp2kCalculation failed with SCF not converged (exit 450).

Uses the current ``config.toml`` (fresh snapshot) rather than the old stored
``config`` input so that SCF/mixing/max_scf changes take effect on resubmits.
"""

import argparse
import logging
import sys

from aiida.manage.configuration import load_profile
from aiida import orm
from aiida.engine import submit

load_profile()

LOGGER = logging.getLogger("resubmit_scf450")
WORKCHAIN_LABELS = {"GwWorkChain"}


def _last_calcjob(node):
    """Return the most recent CalcJobNode descendant of node."""
    calcs = [n for n in node.called_descendants if isinstance(n, orm.CalcJobNode)]
    if not calcs:
        return None
    return max(calcs, key=lambda c: c.ctime)


def _get_candidates(group=None, pks=None):
    """Find finished GwWorkChains with a child Cp2kCalculation that ended with exit_status 450."""
    candidates = []

    if pks:
        for pk in pks:
            try:
                node = orm.load_node(pk)
            except Exception as exc:
                LOGGER.warning("Could not load %s: %s", pk, exc)
                continue
            if node.process_label not in WORKCHAIN_LABELS:
                LOGGER.warning("%s: %s is not a GwWorkChain, skipping", pk, node.process_label)
                continue
            if not node.is_finished:
                LOGGER.warning("%s: not finished yet, skipping", pk)
                continue
            if node.is_finished_ok:
                LOGGER.warning("%s: finished ok, skipping", pk)
                continue
            calc = _last_calcjob(node)
            if calc and calc.exit_status == 450:
                candidates.append(node)
            else:
                LOGGER.warning("%s: last calcjob exit_status != 450, skipping", pk)
        return candidates

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
            candidates.append(node)
    return candidates


def _resubmit(node, run_group=None, dry_run=False):
    from aiida.orm import Dict
    from aiida_gw.core.config import load_config

    builder = node.process_class.get_builder()
    for port in node.inputs:
        if port == "config":
            continue  # drop old snapshot so we pick up current config.toml
        builder[port] = node.inputs[port]

    cfg = load_config()
    builder.config = Dict(dict=cfg.model_dump())

    label = node.label or ""
    LOGGER.info("%s: resubmit GwWorkChain (last calcjob exit 450)", node.pk)
    if dry_run:
        return True

    new_node = submit(builder)
    if run_group is not None:
        run_group.add_nodes([new_node])
    LOGGER.info("  -> new pk %s", new_node.pk)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pks", nargs="+", type=int, help="GwWorkChain PKs to check/resubmit")
    parser.add_argument("--group", help="AiiDA group label to search in")
    parser.add_argument("--run-group", help="Group to add newly submitted nodes to")
    parser.add_argument("--dry-run", action="store_true", help="List candidates without submitting")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.pks and not args.group:
        parser.error("Provide either --pks or --group")

    group = None
    if args.group:
        try:
            group = orm.Group.collection.get(label=args.group)
        except Exception as exc:
            LOGGER.error("Group '%s' not found: %s", args.group, exc)
            return 1

    run_group = None
    if args.run_group and not args.dry_run:
        run_group, _ = orm.Group.collection.get_or_create(label=args.run_group)

    candidates = _get_candidates(group=group, pks=args.pks)
    if not candidates:
        LOGGER.info("No GwWorkChains found with last Cp2kCalculation exit_status 450.")
        return 0

    LOGGER.info("Found %d candidate(s)", len(candidates))
    submitted = 0
    for node in candidates:
        if not node.is_sealed:
            LOGGER.warning("%s: still active, skipping", node.pk)
            continue
        if _resubmit(node, run_group=run_group, dry_run=args.dry_run):
            submitted += 1

    verb = "would resubmit" if args.dry_run else "resubmitted"
    LOGGER.info("%s %d GwWorkChain(s)", verb, submitted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
