from __future__ import annotations

import logging
from typing import Any

from aiida.orm import Group, StructureData
from aiida.plugins import DataFactory
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

StructureData = DataFactory("core.structure")

OPTIMADE_MC2D_URL = "https://aiida-optimade.materialsproject.org"


def fetch_structures_from_optimade(
    group_label: str = "mc2d_structures",
    max_structures: int = 10,
    elements: list[str] | None = None,
) -> Group:
    """Fetch 2D structures from the MC2D database via OPTIMADE."""
    try:
        from optimade_client import QueryBuilder as OptQueryBuilder
    except ImportError:
        logger.error(
            "optimade-client not installed. Run: pip install optimade-client"
        )
        raise

    group, _ = Group.collection.get_or_create(group_label)
    group.clear()

    builder = OptQueryBuilder(
        base_url=OPTIMADE_MC2D_URL,
    )

    filters = ["nelements >= 2"]
    if elements:
        element_filters = [f'has "{el}"' for el in elements]
        filters.append(" AND ".join(element_filters))

    builder.where(" AND ".join(filters)).limit(max_structures)

    try:
        results = builder.all()
    except Exception as e:
        logger.error(f"OPTIMADE query failed: {e}")
        raise

    count = 0
    for entry in results:
        try:
            attributes = getattr(entry, "attributes", entry)
            pymatgen_structure = Structure.from_dict(attributes["structure"])
            node = StructureData(pymatgen=pymatgen_structure)
            node.store()
            node.base.extras.set("source", "mc2d_optimade")
            group.add_nodes(node)
            count += 1
        except Exception as e:
            logger.warning(f"Skipping entry: {e}")
            continue

    logger.info(f"Fetched {count} structures from MC2D/OPTIMADE into group '{group_label}'")
    return group
