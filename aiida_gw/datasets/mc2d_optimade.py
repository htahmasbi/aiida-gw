from __future__ import annotations

import json
import logging
import time
from pathlib import Path
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

    max_retries = 3
    while url:
        retries = 0
        while retries < max_retries:
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                retries += 1
                if retries == max_retries:
                    raise
                wait = 2 ** retries
                logger.warning("OPTIMADE request failed (attempt %d/%d): %s. Retrying in %ds ...",
                               retries, max_retries, exc, wait)
                time.sleep(wait)

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


def _get_structure_elements(item: dict[str, Any]) -> set[str]:
    """Extract element set from a structure item."""
    attrs = item.get("entry", {}).get("attributes", {})
    if "elements" in attrs:
        return set(attrs["elements"])
    struct = item.get("structure")
    if hasattr(struct, "composition"):
        return {str(e) for e in struct.composition.elements}
    if hasattr(struct, "species"):
        return set(struct.species)
    return set()


def _structure_has_unsupported_elements(
    item: dict[str, Any],
    exclude_elements: set[str] | None,
    supported_elements: set[str] | None,
) -> bool:
    """Check if structure contains excluded or unsupported elements."""
    elements = _get_structure_elements(item)
    if not elements:
        return False
    if exclude_elements and (elements & exclude_elements):
        return True
    if supported_elements and not (elements <= supported_elements):
        return True
    return False


def fetch_and_store_mc2d(
    group_label: str = "mc2d_structures",
    max_structures: int = 10,
    elements: list[str] | None = None,
    modifier: Callable[[Structure], Structure] | None = None,
    optimade_filter: str | None = None,
    exclude_elements: set[str] | None = None,
    supported_elements: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch MC2D structures, store in an AiiDA group, return data list.

    Args:
        group_label: AiiDA group label.
        max_structures: Maximum number of structures to fetch.
        elements: Filter by elements (e.g. ["B","N"]). Ignored if *optimade_filter* is set.
        modifier: Optional function applied to each pymatgen Structure.
        optimade_filter: Raw OPTIMADE filter string. Overrides *elements* when provided.
        exclude_elements: Set of elements to exclude; structures containing any are skipped.
        supported_elements: Whitelist; structures with elements outside this set are skipped.
            When given, auto-detects unsupported elements.
    """
    from aiida.orm import Group, StructureData

    group, _ = Group.collection.get_or_create(group_label)

    if optimade_filter:
        pass  # use as-is
    elif elements:
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
        if _structure_has_unsupported_elements(item, exclude_elements, supported_elements):
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


def fetch_all_mc2d(
    max_structures: int | None = None,
    exclude_elements: set[str] | None = None,
    supported_elements: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch all 2D structures from MC2D OPTIMADE.

    Args:
        max_structures: Optional cap on total structures.
        exclude_elements: Set of elements to exclude; structures containing any are skipped.
        supported_elements: Whitelist; structures with elements outside this set are skipped.

    Returns:
        List of dicts with keys: id, formula, nelements, elements, structure (dict), entry.
    """
    raw = fetch_mc2d_structures(max_structures=max_structures)
    results = []
    for item in raw:
        if _structure_has_unsupported_elements(item, exclude_elements, supported_elements):
            continue
        attrs = item["entry"]["attributes"]
        results.append({
            "id": item["id"],
            "formula": item["formula"],
            "nelements": attrs.get("nelements"),
            "elements": attrs.get("elements"),
            "nsites": item.get("nsites"),
            "structure": item["structure"].as_dict(),
        })
    return results


def save_mc2d_by_nelements(
    output_dir: str | Path = ".",
    max_structures: int | None = None,
    exclude_elements: set[str] | None = None,
    supported_elements: set[str] | None = None,
) -> dict[int, str]:
    """Fetch all MC2D structures and save them into ``mc2d_{N}elements.json`` per element count.

    Args:
        output_dir: Directory to write JSON files into.
        max_structures: Optional cap on total structures.
        exclude_elements: Set of elements to exclude; structures containing any are skipped.
        supported_elements: Whitelist; structures with elements outside this set are skipped.

    Returns:
        Mapping of ``nelements → file path``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = fetch_all_mc2d(
        max_structures=max_structures,
        exclude_elements=exclude_elements,
        supported_elements=supported_elements,
    )

    by_nelements: dict[int, list[dict]] = {}
    for item in data:
        n = item["nelements"]
        if n is None:
            logger.warning("Skipping structure %s with nelements=None", item.get("id", "unknown"))
            continue
        by_nelements.setdefault(n, []).append(item)

    paths: dict[int, str] = {}
    for n, entries in sorted(by_nelements.items()):
        filename = f"mc2d_{n}elements.json"
        filepath = out / filename
        with open(filepath, "w") as f:
            json.dump(entries, f, indent=2)
        logger.info(f"Saved {len(entries)} structures to {filepath}")
        paths[n] = str(filepath)

    return paths
