from __future__ import annotations

import logging
from typing import Any, Callable

import requests
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element

logger = logging.getLogger(__name__)

MC2D_STRUCTURES_URL = "https://optimade.materialscloud.org/main/mc2d/v1/structures"

OPTIMADE_RESPONSE_FIELDS = ",".join([
    "id",
    "chemical_formula_reduced",
    "chemical_formula_descriptive",
    "elements",
    "nelements",
    "nsites",
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "species",
    "space_group_symbol_hermann_mauguin",
    "space_group_it_number",
])


def optimade_entry_to_pymatgen(entry: dict) -> Structure:
    """Convert an OPTIMADE entry dict to a pymatgen Structure."""
    attributes = entry["attributes"]
    lattice = Lattice(attributes["lattice_vectors"])
    species = [Element(s) for s in attributes["species_at_sites"]]
    coords = attributes["cartesian_site_positions"]
    return Structure(lattice, species, coords, coords_are_cartesian=True)


def fetch_mc2d_structures(
    optimade_filter: str | None = None,
    page_limit: int = 100,
    max_structures: int | None = None,
    modifier: Callable[[Structure], Structure] | None = None,
    max_atoms: int | None = None,
    min_atoms: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch structures from the MC2D OPTIMADE endpoint.

    Args:
        optimade_filter: OPTIMADE filter string.
        page_limit: Structures per API page.
        max_structures: Stop after this many structures.
        modifier: Optional function applied to each pymatgen Structure.
        max_atoms: Max atoms (nsites) filter.
        min_atoms: Min atoms (nsites) filter.

    Returns:
        List of dicts with keys: id, formula, entry, structure, original_structure.
    """
    params: dict[str, Any] = {
        "page_limit": page_limit,
        "response_fields": OPTIMADE_RESPONSE_FIELDS,
    }
    if optimade_filter:
        params["filter"] = optimade_filter

    results: list[dict[str, Any]] = []
    url: str | None = MC2D_STRUCTURES_URL

    while url:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        for entry in data["data"]:
            attributes = entry["attributes"]
            nsites = attributes.get("nsites")

            if max_atoms is not None and (nsites is None or nsites > max_atoms):
                continue
            if min_atoms is not None and (nsites is None or nsites < min_atoms):
                continue

            structure = optimade_entry_to_pymatgen(entry)
            original_structure = structure.copy()

            if modifier is not None:
                structure = modifier(structure)

            results.append({
                "id": entry["id"],
                "formula": attributes.get("chemical_formula_reduced"),
                "entry": entry,
                "structure": structure,
                "original_structure": original_structure,
                "nsites": nsites,
            })

            if max_structures is not None and len(results) >= max_structures:
                return results

        url = data.get("links", {}).get("next")
        params = None

    return results


def fetch_and_store_mc2d(
    group_label: str = "mc2d_structures",
    max_structures: int = 10,
    elements: list[str] | None = None,
    modifier: Callable[[Structure], Structure] | None = None,
) -> list[dict[str, Any]]:
    """Fetch MC2D structures, store in an AiiDA group, return data list."""
    from aiida.orm import Group, StructureData

    group, _ = Group.collection.get_or_create(group_label)

    if elements:
        quoted = '","'.join(elements)
        optimade_filter = f'elements HAS ALL "{quoted}" AND nelements={len(elements)}'
    else:
        optimade_filter = None

    data = fetch_mc2d_structures(
        optimade_filter=optimade_filter,
        max_structures=max_structures,
        modifier=modifier,
    )

    existing_ids = set()
    for node in group.nodes:
        oid = node.base.extras.get("optimade_id", None)
        if oid is not None:
            existing_ids.add(oid)

    count = 0
    for item in data:
        if item["id"] in existing_ids:
            continue
        pymatgen_structure = item["structure"]
        try:
            node = StructureData(pymatgen=pymatgen_structure)
            node.store()
            node.base.extras.set("source", "mc2d_optimade")
            node.base.extras.set("optimade_id", item["id"])
            node.base.extras.set("formula", item["formula"])
            group.add_nodes(node)
            count += 1
        except Exception as e:
            logger.warning(f"Skipping {item['id']}: {e}")
            continue

    logger.info(f"Fetched {count} MC2D structures into group '{group_label}'")
    return data
