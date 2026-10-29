#!/usr/bin/env python3
"""Resubmit GwWorkChains killed by walltime/time limit."""

import argparse
import logging
import sys

from aiida.manage.configuration import load_profile
from aiida import orm
from aiida.engine import submit

load_profile()

LOGGER = logging.getLogger("resubmit_timeout")
WORKCHAIN_LABELS = {"GwWorkChain"}
TIMEOUT_KEYWORDS = ("TIMEOUT", "TIMED_OUT", "CANCELLED", "CANCELED", "NODE_FAIL", "PREEMPTED", "DUE TO TIME", "WALLTIME")


def _last_calcjob(node):
    calcs = [n for n in node.called_descendants if isinstance(n, orm.CalcJobNode)]
    if not calcs:
        return None
    return max(calcs, key=lambda c: c.ctime)


def _get_retrieved(calc):
    try:
        return [out.node for out in calc.base.links.get_outgoing() if out.link_label == "retrieved"]
    except Exception:
        return []


def _is_timeout(calc):
    if calc is None:
        return False
    for retrieved in _get_retrieved(calc):
        for name in ("_scheduler-stdout.txt", "_scheduler-stderr.txt"):
            try:
                content = retrieved.base.repository.get_object_content(name)
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                if content and any(k in content.upper() for k in TIMEOUT_KEYWORDS):
                    return True
            except Exception:
                continue
        try:
            content = retrieved.base.repository.get_object_content("aiida.out")
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if content:
                c = content.upper()
                if "TIME" in c and "LIMIT" in c:
                    return True
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                if lines and "TIME" in lines[-1].upper() and "LIMIT" in lines[-1].upper():
                    return True
        except Exception:
            continue
    return False


def _get_candidates(group=None, groups=None, pks=None):
    candidates = []
    all_groups = groups or []

    if pks:
        for pk in pks:
            try:
                node = orm.load_node(pk)
            except Exception as exc:
                LOGGER.warning("Could not load %s: %s", pk, exc)
                continue
            if getattr(node, "process_label") not in WORKCHAIN_LABELS:
                LOGGER.warning("%s: %s is not a GwWorkChain, skipping", pk, getattr(node, "process_label"))
                continue
            if not node.is_finished:
                LOGGER.warning("%s: not finished yet, skipping", pk)
                continue
            calc = _last_calcjob(node)
            if _is_timeout(calc):
                candidates.append((node, None))
            else:
                LOGGER.warning("%s: no timeout detected in scheduler logs, skipping", pk)
        return candidates

    group_objs = []
    if all_groups:
        for g in all_groups:
            try:
                group_objs.append(orm.Group.collection.get(label=g))
            except Exception as exc:
                LOGGER.warning("Group '%s' not found: %s", g, exc)
    elif group is not None:
        group_objs = [orm.Group.collection.get(label=group)]
    else:
        parser = argparse.ArgumentParser()
        parser.error("Provide either --pks or --group/--groups")

    for gobj in group_objs:
        qb = orm.QueryBuilder()
        qb.append(
            orm.WorkflowNode,
            tag="w",
            filters={
                "attributes.process_label": {"in": sorted(WORKCHAIN_LABELS)},
                "attributes.process_state": "finished",
            },
            project=["id"],
        )
        qb.append(orm.Group, filters={"id": gobj.pk}, with_node="w")

        for (pk,) in qb.all():
            node = orm.load_node(pk)
            calc = _last_calcjob(node)
            if _is_timeout(calc):
                candidates.append((node, gobj.label))


def _resubmit(node, run_group=None, dry_run=False):
    from aiida.orm import Dict
    from aiida_gw.core.config import load_config

    builder = node.process_class.get_builder()
    for port in node.inputs:
        if port == "config":
            continue
        builder[port] = node.inputs[port]

    cfg = load_config()
    builder.config = Dict(dict=cfg.model_dump())

    LOGGER.info("%s: resubmit GwWorkChain (time limit)", node.pk)
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
    parser.add_argument("--groups", nargs="+", help="Multiple group labels")
    parser.add_argument("--run-group", help="Group to add newly submitted nodes to")
    parser.add_argument("--dry-run", action="store_true", help="List candidates without submitting")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.pks and not args.group and not args.groups:
        parser.error("Provide either --pks or --group/--groups")

    run_group = None
    if args.run_group and not args.dry_run:
        run_group, _ = orm.Group.collection.get_or_create(label=args.run_group)

    candidates = _get_candidates(group=args.group, groups=args.groups, pks=args.pks)
    if not candidates:
        LOGGER.info("No GwWorkChains found with time limit.")
        return 0

    LOGGER.info("Found %d candidate(s)", len(candidates))
    submitted = 0
    for node, src_group in candidates:
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
