#!/usr/bin/env python3
"""Submit fresh GwWorkChains for structures with negative GW gaps using a denser-k config.

Does NOT copy old stored configs/inputs from failed workchains. Extracts the
input StructureData from old workchains and submits new ones with the new config.
"""

import argparse
import logging
import sys
from pathlib import Path

from aiida.manage.configuration import load_profile

load_profile()

from aiida import orm
from aiida.engine import submit

from aiida_gw.workflows.gw import GwWorkChain
try:
    from aiida_gw.core.config import load_config
except Exception:
    try:
        from aiida_gw.utils.config import load_config
    except Exception:
        load_config = None


LOGGER = logging.getLogger("resubmit_neg_gap_kdenser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _get_input_structure(wc):
    """Extract input StructureData from GwWorkChain."""
    try:
        for inp in wc.inputs:
            try:
                node = wc.inputs[inp]
                if isinstance(node, orm.StructureData):
                    return node
            except Exception:
                continue
    except Exception:
        pass
    try:
        for link in wc.base.links.get_incoming():
            node = link.node
            if isinstance(node, orm.StructureData):
                return node
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
    if load_config is not None:
        try:
            config = load_config(str(config_path))
        except Exception as exc:
            LOGGER.error("Failed to load config via load_config: %s", exc)
            return 1
    else:
        LOGGER.error("Could not import load_config from aiida_gw.core.config or aiida_gw.utils.config")
        return 1

    run_group = None
    if not args.dry_run:
        try:
            run_group = orm.Group.collection.get_or_create(label=args.run_group)[0]
            LOGGER.info("Using group: %s", args.run_group)
        except Exception as exc:
            LOGGER.error("Failed to get/create group: %s", exc)
            return 1
    else:
        LOGGER.info("[DRY RUN] Would use/create group: %s (not created now)", args.run_group)

    submitted = []
    skipped = []

    for pk in args.pks:
        try:
            node = orm.load_node(pk)
        except Exception as exc:
            LOGGER.warning("Could not load PK %s: %s", pk, exc)
            skipped.append((pk, str(exc)))
            continue

        if getattr(node, "process_label", "") not in ["GwWorkChain"]:
            LOGGER.warning("%s: process_label %s not GwWorkChain, skipping", pk, getattr(node, "process_label", ""))
            skipped.append((pk, f"not GwWorkChain ({getattr(node, 'process_label', '')})"))
            continue

        struct = _get_input_structure(node)
        if not struct:
            LOGGER.warning("%s: could not find input StructureData, skipping", pk)
            skipped.append((pk, "no input StructureData"))
            continue

        if args.dry_run:
            try:
                formula = struct.get_formula()
            except Exception:
                formula = "N/A"
            LOGGER.info("[DRY RUN] Would submit fresh GwWorkChain from PK %s (struct %s, formula %s) with config %s into group %s",
                        pk, getattr(struct, "uuid", "N/A"), formula, config_path, args.run_group)
            continue

        try:
            inputs = {"structure": struct}
            try:
                if hasattr(config, "model_dump"):
                    cfg_dict = config.model_dump()
                elif hasattr(config, "__dict__"):
                    cfg_dict = dict(config)
                else:
                    cfg_dict = config
                inputs["config"] = orm.Dict(cfg_dict)
            except Exception:
                try:
                    inputs["config"] = orm.Dict(config)
                except Exception:
                    pass

            try:
                node_new = submit(GwWorkChain, **inputs)
                try:
                    formula = struct.get_formula()
                except Exception:
                    formula = "N/A"
                LOGGER.info("Submitted %s from %s -> new PK %s", formula, pk, node_new.pk)
                if run_group is not None:
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


if __name__ == "__main__":
    sys.exit(main())
