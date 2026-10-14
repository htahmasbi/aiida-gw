#!/usr/bin/env python3
"""Rebuild and store outputs for CalcJobs that AiiDA ``Excepted`` while CP2K succeeded.

When the daemon executes the same CalcJob twice (the RabbitMQ ``consumer_timeout``
redelivery bug), the ``retrieved`` output gets attached more than once. AiiDA's
link uniqueness then raises during ``parse`` / ``update_outputs`` and the process
ends in ``Excepted`` even though the scheduler job returned exit 0. The physics is
already on disk in the retrieved folder, but ``output_parameters`` was never
stored, so ``extract_gw.py`` reports the run as data-less.

This script re-runs the *existing* parser logic (``Cp2kEFSParser._parse_efs``) on
the surviving retrieved folder and attaches the resulting ``output_parameters``
(and ``output_structure``) to the CalcJobNode with a single ``CREATE`` link. The
normal links API already tolerates the duplicate ``retrieved`` link, so existing
tooling picks the results up afterwards without any further repair.

The parser class is not instantiated directly because ``Parser.__init__`` /
``parse`` resolve ``self.node.outputs.retrieved``, which raises on the duplicate
label. Instead a small shim supplies the attributes ``_parse_efs`` needs
(``retrieved``, ``logger``, ``out``) so the real parsing code is reused verbatim
and cannot drift from the plugin.

Only successful runs are worth salvaging: an ``Excepted`` calcjob whose job exit
code is non-zero (256, 253, ...) is a genuine CP2K failure and is left untouched.

Run inside the aiida environment:

    python3 utils/reparse_excepted.py --pks 4337 4090
    python3 utils/reparse_excepted.py --pks 4302 --dry-run   # workchain PK works too
"""
import argparse
import logging
import sys

LOGGER = logging.getLogger("reparse_excepted")


def _resolve_calcjob(node):
    """Return the CalcJobNode to repair, accepting either a calcjob or a workchain PK."""
    from aiida.orm import CalcJobNode

    if isinstance(node, CalcJobNode):
        return node

    descendants = list(getattr(node, "called_descendants", []))
    calcs = [n for n in descendants if n.process_label == "Cp2kCalculation"]
    if not calcs:
        calcs = [n for n in descendants if isinstance(n, CalcJobNode)]
    return max(calcs, key=lambda c: c.ctime) if calcs else None


def _retrieved_nodes(calc):
    """Every node linked as ``retrieved`` from ``calc``.

    Resolved through the links API rather than ``calc.outputs.retrieved`` because
    the duplicate label makes that accessor raise.
    """
    return [out.node for out in calc.base.links.get_outgoing() if out.link_label == "retrieved"]


def _aiida_out_size(retrieved):
    for obj in retrieved.base.list_objects():
        if obj.name == "aiida.out":
            return obj.size
    return None


def _pick_retrieved(calc):
    """Pick the retrieved folder that actually holds ``aiida.out``.

    A double execution leaves two retrieved folders; the largest ``aiida.out`` is
    the most complete one, with the newest PK as a tie-breaker.
    """
    usable = [node for node in _retrieved_nodes(calc) if _aiida_out_size(node) is not None]
    if not usable:
        return None
    return max(usable, key=lambda node: (_aiida_out_size(node), node.pk))


class _ParserShim:
    """Minimal stand-in for a Parser so ``Cp2kEFSParser._parse_efs`` can be reused.

    ``_parse_efs`` only touches ``self.retrieved``, ``self.logger``, ``self.out``
    and ``self._parse_gw_outputs``; the shim provides those without ever resolving
    the duplicate ``retrieved`` output link.
    """

    def __init__(self, retrieved, logger):
        self.retrieved = retrieved
        self.logger = logger
        self.outputs = {}

    def out(self, label, node):
        self.outputs[label] = node

    def _parse_gw_outputs(self, result_dict):
        from aiida_gw.codes.cp2k.cp2k_parsers import Cp2kEFSParser

        return Cp2kEFSParser._parse_gw_outputs(self, result_dict)


def reparse_calcjob(calc, retrieved):
    """Return ``{link_label: Data}`` by parsing ``retrieved/aiida.out`` with the plugin parser."""
    from aiida_gw.codes.cp2k.cp2k_parsers import Cp2kEFSParser
    from aiida_gw.codes.cp2k.parsers import parse_cp2k_output_simple

    text = retrieved.base.repository.get_object_content("aiida.out")
    result_dict = parse_cp2k_output_simple(text)

    shim = _ParserShim(retrieved, LOGGER)
    Cp2kEFSParser._parse_efs(shim, result_dict)
    return shim.outputs


def attach_outputs(calc, outputs, dry_run=False):
    """Store and link the parsed outputs onto ``calc``; return the labels handled."""
    from aiida.common.links import LinkType

    existing = {link.link_label for link in calc.base.links.get_outgoing()}
    attached = []
    for label, node in outputs.items():
        if label in existing:
            LOGGER.info("  skip %s: label already linked", label)
            continue
        if dry_run:
            attached.append(label)
            continue
        node.store()
        node.base.links.add_incoming(calc, LinkType.CREATE, label)
        attached.append(label)
    return attached


def _summary(outputs):
    from aiida.orm import Dict

    params = outputs.get("output_parameters")
    if not isinstance(params, Dict):
        return "no output_parameters parsed"
    data = params.get_dict()
    parts = []
    for key in ("scf_gap_indirect", "g0w0_gap_indirect", "g0w0_soc_gap_indirect", "scf_soc_gap_indirect"):
        value = data.get(key)
        if value is not None:
            parts.append(f"{key}={value:.3f}")
    return " ".join(parts) or "parsed, but no gaps present"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pks", nargs="+", type=int, required=True, help="CalcJob or workchain PKs to repair")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and report without storing or linking anything on the node",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from aiida import load_profile, orm

    load_profile()

    failures = 0
    for pk in args.pks:
        node = orm.load_node(pk)
        calc = _resolve_calcjob(node)
        if calc is None:
            LOGGER.error("%s: no CalcJobNode descendant", pk)
            failures += 1
            continue

        retrieved = _pick_retrieved(calc)
        if retrieved is None:
            LOGGER.error("%s (calc %s): no retrieved folder contains aiida.out", pk, calc.pk)
            failures += 1
            continue

        try:
            outputs = reparse_calcjob(calc, retrieved)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next PK
            LOGGER.error("%s (calc %s): reparse failed: %s", pk, calc.pk, exc)
            failures += 1
            continue

        try:
            attached = attach_outputs(calc, outputs, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 - a sealed node may refuse the link
            LOGGER.error("%s (calc %s): attaching outputs failed: %s", pk, calc.pk, exc)
            failures += 1
            continue

        verb = "would attach" if args.dry_run else "attached"
        LOGGER.info(
            "%s -> calc %s (retrieved %s): %s %s | %s",
            pk, calc.pk, retrieved.pk, verb, ", ".join(attached) or "nothing", _summary(outputs),
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
