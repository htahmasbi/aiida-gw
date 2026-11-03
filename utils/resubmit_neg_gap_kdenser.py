#!/usr/bin/env python3
"""Submit fresh GwWorkChains for structures with negative GW gaps using a denser-k config.

Does NOT copy old stored configs/inputs from failed workchains. Extracts the
input StructureData from old workchains and submits new ones with the new config.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from aiida.manage.configuration import load_profile

load_profile()

from aiida import orm
from aiida.engine import submit

from aiida_gw.workflows.gw import GwWorkChain
from aiida_gw.utils.config import load_config

LOGGER = logging.getLogger("resubmit_neg_gap_kdenser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _get_input_structure(wc):
    """Extract input StructureData from GwWorkChain."""
    try:
        # Try common input links
        for inp in wc.inputs:
            try
                node = wc.inputs[inp]
                if isinstance(node, orm.StructureData):
                    return node
            except Exception:
                continue
    except Exception:
        pass
    try:
        # Try to find from predecessors/inputs
        for link in wc.base.links.get_incoming():
            node = link.node
            if isinstance(node, orm.StructureData):
                return node
    except Exception:
        pass
    return None


def _get_old_config_dict(wc):
    """Try to extract old config dict for reference."""
    try:
        for inp in wc.inputs:
            try
                node = wc.inputs[inp]
                if isinstance(node, orm.Dict):
                    d = node.get_dict()
                    if d.get('cp2k') or d.get('kpoints') or d.get('gw'):
                        return d
            except Exception:
                continue
    except Exception:
        pass
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pks", nargs="+", type=int, required=True, help="Old GwWorkChain PKs to resubmit")
    parser.add_argument("--config", default="config.toml", help="New config with denser k-points (default: config.toml)")
    parser.add_argument("--run-group", required=True, help="AiiDA group to put new workchains in")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be done")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        LOGGER.error("Config file not found: %s", config_path)
        return 1

    LOGGER.info("Loading config from %s", config_path)
    config = load_config(str(config_path))

    run_group = None
    try:
        run_group = orm.Group.collection.get_or_create(label=args.run_group)[0]
        LOGGER.info("Using group: %s", args.run_group)
    except Exception as exc:
        LOGGER.error("Failed to get/create group: %s", exc)
        return 1

    submitted = []
    skipped = []

    for pk in args.pks:
        try            node = orm.load_node(pk)ná
        except Exception as exc:
            LOGGER.warning("Could not load PK %s: %s", pk, exc)
            skipped.append((pk, str(exc)))
            continue

        if node.process_label not in ["GwWorkChain"]:
            LOGGER.warning("%s: process_label %s not GwWorkChain, skipping", pk, node.process_label)
            skipped.append((pk, f"not GwWorkChain ({node.process_label})"))
            continue

        struct = _get_input_structure(node)
        if not struct:
            LOGGER.warning("%s: could not find input StructureData, skipping", pk)
            skipped.append((pk, "no input StructureData"))
            continue

        old_cfg = _get_old_config_dict(node)
        if old_cfg and not args.dry_run:
            LOGGER.info("Old config reference for %s found (kpoints/gw/cp2k present)", pk)

        if args.dry_run:
            LOGGER.info("[DRY RUN] Would submit fresh GwWorkChain from PK %s (struct %s, formula %s) with config %s into group %s",
                        pk, struct.uuid, getattr(struct, "get_formula", lambda: "N/A")(), config_path, args.run_group)
            continue

        try:
            # Build inputs - pass config dict/object as expected by GwWorkChain
            inputs = {"structure": struct}
            # Try to pass config
            try:
                inputs["config"] = orm.Dict(dict=config)
            except Exception:
                try:
                    inputs["config"] = config
                except Exception:
                    pass

            # Also try to preserve code/computer if possible from config
            try:
                node_new = submit(GwWorkChain, **inputs)
                LOGGER.info("Submitted %s from %s -> new PK %s", struct.get_formula(), pk, node_new.pk)
                run_group.add_nodes([node_new])
                submitted.append((pk, node_new.pk))
            except Exception as exc:
                LOGGER.error("Failed to submit from %s: %s", pk, exc)
                skipped.append((pk, str(exc)))
                continue
        except Exception as exc:
            LOGGER.error("Error processing %s: %s", pk, exc)
            skipped.append((pk, str(exc)))
            continue

    LOGGER.info("=" * 60)
    LOGGER.info("Submitted: %d", len(submitted))
    for old_pk, new_pk in submitted:
        LOGGER.info("  %s -> %s", old_pk, new_pk)
    LOGGER.info("Skipped: %d", len(skipped))
    for old_pk, reason in skipped:
        LOGGER.info("  %s: %s", old_pk, reason)
    LOGGER.info("=" * 60)

    return 0


if __name__ == "__main":
    sys.exit(main())
