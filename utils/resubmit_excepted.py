#!/usr/bin/env python3
"""Resubmit workchains that failed with the same inputs.

Covers workflows that ended as Excepted/Crashed/Killed, and also Finished
workflows with a non-zero exit status (e.g. GwWorkChain exit_300 because the
child CalcJob never produced outputs). Each listed node (or every failed
workflow in ``--group``) is re-submitted as a fresh process with identical
inputs; provenance of the old attempt is preserved untouched.

Use ``--dry-run`` to list what would be resubmitted first. Note this only
applies to the current workchain code; for old failed runs where the child
CalcJob actually succeeded on the cluster, prefer ``reparse_excepted.py``.

When a batch mixed several failure modes, ``--only-no-output`` is the safe
way to resubmit a subset: it only resubmits workchains whose CalcJob left no
``aiida.out`` (killed before CP2K printed anything, e.g. a cluster outage).
Failures that are the job's own fault (out-of-memory, SCF not converging,
CP2K abort) always leave a log, so they are skipped rather than re-run into
the same wall.

Run inside the aiida environment:

    python3 utils/resubmit_excepted.py --group gw_runs_2el5_2test --dry-run
    python3 utils/resubmit_excepted.py --group gw_runs_2el6 --only-no-output --dry-run
    python3 utils/resubmit_excepted.py --pks 5001 5002 --run-group gw_runs
"""
import argparse
import logging
import sys

LOGGER = logging.getLogger("resubmit_excepted")

FAILED_STATES = {"excepted", "crashed", "killed"}
WORKCHAIN_LABELS = {"GwWorkChain", "SinglePointWorkChain", "RelaxWorkChain"}


def _last_calcjob(node):
    """Return the most recent CalcJobNode descendant of ``node``, or None."""
    from aiida import orm

    calcs = [n for n in node.called_descendants if isinstance(n, orm.CalcJobNode)]
    return max(calcs, key=lambda c: c.ctime) if calcs else None


def _has_aiida_out(calc):
    """True if any retrieved folder of ``calc`` holds an ``aiida.out``.

    A CalcJob that died of its own fault (out-of-memory, SCF not converging,
    a CP2K abort) always leaves a log. A job killed before it produced output
    (cluster outage, mass scancel) has none. This is what separates the two.
    """
    if calc is None:
        return False
    retrieved = [out.node for out in calc.base.links.get_outgoing() if out.link_label == "retrieved"]
    for folder in retrieved:
        try:
            folder.base.repository.get_object_size("aiida.out")
            return True
        except Exception:  # noqa: BLE001 - file absent/unreadable for this folder
            continue
    return False


def _failed_workflows(group=None, pks=None):
    """Yield failed workflow nodes from explicit PKs or from a group."""
    from aiida import orm

    if pks:
        for pk in pks:
            node = orm.load_node(pk)
            if node.process_label not in WORKCHAIN_LABELS:
                LOGGER.warning("%s: %s is not a workflow, skipping", pk, node.process_label)
                continue
            if node.is_finished_ok:
                LOGGER.warning("%s: finished ok, skipping (would create a duplicate run)", pk)
                continue
            yield node
        return

    builder = orm.QueryBuilder().append(
        orm.WorkflowNode,
        tag="node",
        filters={
            "attributes.process_label": {"in": sorted(WORKCHAIN_LABELS)},
            "or": [
                {"attributes.process_state": {"in": sorted(FAILED_STATES)}},
                {
                    "and": [
                        {"attributes.process_state": "finished"},
                        {"attributes.exit_status": {">": 0}},
                    ]
                },
            ],
        },
        project="id",
    )
    if group is not None:
        builder.append(orm.Group, filters={"id": group.pk}, with_node="node")
    for (pk,) in builder.all():
        yield orm.load_node(pk)


def _resubmit(node, run_group=None, dry_run=False):
    """Submit a fresh copy of ``node`` with identical inputs."""
    from aiida import orm
    from aiida.engine import submit

    builder = node.process_class.get_builder()
    for port in node.inputs:
        builder[port] = node.inputs[port]

    label = node.label or ""
    LOGGER.info(
        "%s: resubmit %s%s", node.pk, node.process_type, f" (label '{label}')" if label else ""
    )
    if dry_run:
        return True

    new_node = submit(builder)
    if run_group is not None:
        run_group.nodes = list(run_group.nodes) + [new_node]
    LOGGER.info("  -> new pk %s", new_node.pk)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pks", nargs="+", type=int, help="workflow PKs to resubmit")
    parser.add_argument("--group", help="label of an AiiDA group; resubmit all failed workflows in it")
    parser.add_argument("--run-group", help="store the newly submitted nodes in this group")
    parser.add_argument(
        "--only-no-output",
        action="store_true",
        help="only resubmit workchains whose CalcJob left no aiida.out "
        "(killed before CP2K printed anything); skips memory/SCF/abort failures",
    )
    parser.add_argument("--dry-run", action="store_true", help="list candidates without submitting")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from aiida import load_profile, orm

    load_profile()

    if not args.pks and not args.group:
        parser.error("provide --pks or --group")

    group = None
    if args.group:
        try:
            group = orm.Group.collection.get(label=args.group)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("group '%s' not found: %s", args.group, exc)
            return 1

    run_group = None
    if args.run_group and not args.dry_run:
        run_group, _ = orm.Group.collection.get_or_create(label=args.run_group)

    submitted = 0
    for node in _failed_workflows(group=group, pks=args.pks):
        if node.pk and node.is_sealed is False:
            LOGGER.warning("%s: still active, use 'verdi process kill' first", node.pk)
            continue
        if args.only_no_output and _has_aiida_out(_last_calcjob(node)):
            LOGGER.info("%s: has aiida.out (ran and failed on its own), skipping", node.pk)
            continue
        new_node = _resubmit(node, run_group=run_group, dry_run=args.dry_run)
        submitted += new_node is not None

    verb = "would resubmit" if args.dry_run else "resubmitted"
    LOGGER.info("%s %d workflow(s)", verb, submitted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
