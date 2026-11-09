"""Fetch the current MC2D ``pbe-v1`` data from the Materials Cloud REST APIs.

The MC2D website uses the Materials Cloud REST API for curated metadata
(``mcxd-api``) and the AiiDA REST API for structures and provenance.  This
module combines both: ``overview`` supplies ``mc2d-N`` IDs and band gaps,
``structure-uuids`` maps those IDs to relaxed structure UUIDs, and the AiiDA
REST API provides the CIF structures.

The legacy OPTIMADE endpoint is intentionally not used here.  It exposes the
2018 snapshot with AiiDA UUIDs and incomplete band gaps.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from pymatgen.core import Composition, Structure

logger = logging.getLogger(__name__)

MC2D_API_BASE = "https://mcxd-api.materialscloud.org"
MC2D_OVERVIEW_URL = f"{MC2D_API_BASE}/mc2d/pbe-v1/overview"
MC2D_STRUCTURE_UUIDS_URL = f"{MC2D_API_BASE}/mc2d/pbe-v1/structure-uuids"
MC2D_CORE_URL = f"{MC2D_API_BASE}/mc2d/pbe-v1/core_base/{{mc2d_id}}"

AIIDA_API_BASE = "https://aiida.materialscloud.org/mc2d/api/v4"
AIIDA_NODE_URL = f"{AIIDA_API_BASE}/nodes/{{uuid}}"
AIIDA_ATTRS_URL = f"{AIIDA_NODE_URL}/contents/attributes"
AIIDA_CIF_URL = f"{AIIDA_NODE_URL}/download?download_format=cif"

HEADERS = {
    "User-Agent": "aiida-gw-mc2d-pbev1/1.0",
    "Accept": "application/json",
}


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def fetch_overview(timeout: int = 60, session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Return the ``pbe-v1`` overview (one entry per ``mc2d-N``)."""
    session = session or _session()
    response = session.get(MC2D_OVERVIEW_URL, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected overview payload: {type(data).__name__}")
    return data


def fetch_structure_uuids(timeout: int = 60, session: requests.Session | None = None) -> dict[str, str]:
    """Return the ``mc2d-N -> relaxed structure UUID`` mapping."""
    session = session or _session()
    response = session.get(MC2D_STRUCTURE_UUIDS_URL, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected structure-uuids payload: {type(data).__name__}")
    return {str(key): str(value) for key, value in data.items()}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def load_or_fetch_overview(
    cache_path: str | Path | None = None,
    timeout: int = 60,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Read the overview from *cache_path* when present, else fetch and save it."""
    if cache_path is not None:
        path = Path(cache_path)
        if path.is_file():
            logger.info("Reading overview from %s", path)
            data = read_json(path)
            if not isinstance(data, list):
                raise ValueError(f"{path}: expected a JSON list")
            return data
    data = fetch_overview(timeout=timeout, session=session)
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        logger.info("Saved overview to %s", path)
    return data


def load_or_fetch_structure_uuids(
    cache_path: str | Path | None = None,
    timeout: int = 60,
    session: requests.Session | None = None,
) -> dict[str, str]:
    """Read the structure-uuids map from *cache_path* when present, else fetch and save it."""
    if cache_path is not None:
        path = Path(cache_path)
        if path.is_file():
            logger.info("Reading structure UUIDs from %s", path)
            data = read_json(path)
            if not isinstance(data, dict):
                raise ValueError(f"{path}: expected a JSON object")
            return {str(key): str(value) for key, value in data.items()}
    data = fetch_structure_uuids(timeout=timeout, session=session)
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        logger.info("Saved structure UUIDs to %s", path)
    return data


def _elements(formula: str | None) -> set[str]:
    if not formula:
        return set()
    try:
        return {str(element) for element in Composition(formula).elements}
    except Exception:
        return set()


def _reduced_formula(formula: str | None) -> str | None:
    if not formula:
        return None
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return str(formula).strip()


def _gap_allowed(gap, min_band_gap: float | None, include_unknown_gap: bool) -> bool:
    if min_band_gap is None:
        return True
    if gap is None:
        return include_unknown_gap
    try:
        return float(gap) > min_band_gap
    except (TypeError, ValueError):
        return include_unknown_gap


def select_entries(
    overview: Iterable[dict[str, Any]],
    nelements: int | None = None,
    min_nelements: int | None = None,
    max_nelements: int | None = None,
    exclude_elements: Iterable[str] | None = None,
    formulas: Iterable[str] | None = None,
    min_band_gap: float | None = None,
    include_unknown_gap: bool = False,
    stable_only: bool = False,
    exclude_unstable: bool = False,
) -> list[dict[str, Any]]:
    """Filter overview entries by composition, band gap and stability.

    ``stable_only`` requires ``unstable is False`` and drops entries with no
    phonon data.  ``exclude_unstable`` only drops entries explicitly marked
    unstable and keeps unknown stability.
    """
    excluded = {str(element) for element in (exclude_elements or ())}
    wanted_formulas = None
    if formulas is not None:
        wanted_formulas = {_reduced_formula(formula) for formula in formulas}
        wanted_formulas.discard(None)

    selected = []
    for entry in overview:
        formula = entry.get("formula")
        elements = _elements(formula)
        if not elements:
            continue
        count = len(elements)
        if nelements is not None and count != nelements:
            continue
        if min_nelements is not None and count < min_nelements:
            continue
        if max_nelements is not None and count > max_nelements:
            continue
        if excluded and elements & excluded:
            continue
        if wanted_formulas is not None and _reduced_formula(formula) not in wanted_formulas:
            continue
        if stable_only and entry.get("unstable") is not False:
            continue
        if exclude_unstable and entry.get("unstable") is True:
            continue
        if not _gap_allowed(entry.get("bg"), min_band_gap, include_unknown_gap):
            continue
        selected.append(entry)
    return selected


def fetch_cif_structure(
    uuid: str,
    timeout: int = 60,
    session: requests.Session | None = None,
) -> Structure:
    """Download a structure from the AiiDA REST API and parse it as CIF."""
    session = session or _session()
    response = session.get(AIIDA_CIF_URL.format(uuid=uuid), timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return Structure.from_str(response.text, fmt="cif")


def build_pbev1_entry(
    overview_entry: dict[str, Any],
    structure: Structure,
    structure_uuid: str | None = None,
) -> dict[str, Any]:
    """Build a ``fetch-json`` compatible entry with pbe-v1 metadata."""
    mc2d_id = overview_entry.get("id")
    elements = sorted(str(element) for element in structure.composition.elements)
    return {
        "id": mc2d_id,
        "mc2d_id": mc2d_id,
        "formula": overview_entry.get("formula") or structure.composition.reduced_formula,
        "nelements": len(elements),
        "elements": elements,
        "nsites": len(structure),
        "structure": structure.as_dict(),
        "band_gap": overview_entry.get("bg"),
        "magnetic_state": overview_entry.get("mags"),
        "unstable": overview_entry.get("unstable"),
        "structure_uuid": structure_uuid,
        "dataset_version": "pbe-v1",
    }


def fetch_pbev1_entries(
    overview: Iterable[dict[str, Any]],
    structure_uuids: dict[str, str],
    nelements: int | None = None,
    min_nelements: int | None = None,
    max_nelements: int | None = None,
    exclude_elements: Iterable[str] | None = None,
    formulas: Iterable[str] | None = None,
    min_band_gap: float | None = None,
    include_unknown_gap: bool = False,
    stable_only: bool = False,
    exclude_unstable: bool = False,
    limit: int | None = None,
    sleep: float = 0.05,
    timeout: int = 60,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select overview entries and download their relaxed structures.

    Returns the downloaded entries and a small statistics dictionary.
    """
    selected = select_entries(
        overview,
        nelements=nelements,
        min_nelements=min_nelements,
        max_nelements=max_nelements,
        exclude_elements=exclude_elements,
        formulas=formulas,
        min_band_gap=min_band_gap,
        include_unknown_gap=include_unknown_gap,
        stable_only=stable_only,
        exclude_unstable=exclude_unstable,
    )
    stats = {"selected": len(selected), "fetched": 0, "no_uuid": 0, "failed": 0}
    if limit is not None:
        selected = selected[:limit]

    session = session or _session()
    entries: list[dict[str, Any]] = []
    for index, overview_entry in enumerate(selected, start=1):
        mc2d_id = overview_entry.get("id")
        uuid = structure_uuids.get(mc2d_id)
        if not uuid:
            stats["no_uuid"] += 1
            logger.warning("No structure UUID for %s", mc2d_id)
            continue
        try:
            structure = fetch_cif_structure(uuid, timeout=timeout, session=session)
        except Exception as exc:
            stats["failed"] += 1
            logger.warning("Failed to download %s (%s): %s", mc2d_id, uuid, exc)
        else:
            entries.append(build_pbev1_entry(overview_entry, structure, structure_uuid=uuid))
            stats["fetched"] += 1
        if index % 25 == 0:
            logger.info("Processed %d/%d entries (fetched %d)", index, len(selected), stats["fetched"])
        if sleep:
            time.sleep(sleep)
    return entries, stats


def save_by_nelements(
    entries: Iterable[dict[str, Any]],
    output_dir: str | Path = ".",
    structures_per_file: int | None = None,
) -> dict[int, str | list[str]]:
    """Write entries to ``mc2d_<n>elements.json`` files grouped by element count."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_nelements: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        count = entry.get("nelements")
        if count is None:
            logger.warning("Skipping %s with nelements=None", entry.get("id", "unknown"))
            continue
        by_nelements.setdefault(int(count), []).append(entry)

    paths: dict[int, str | list[str]] = {}
    for count, group in sorted(by_nelements.items()):
        if structures_per_file and len(group) > structures_per_file:
            file_list: list[str] = []
            for start in range(0, len(group), structures_per_file):
                chunk = group[start:start + structures_per_file]
                chunk_number = start // structures_per_file + 1
                path = out / f"mc2d_{count}elements_{chunk_number}.json"
                path.write_text(json.dumps(chunk, indent=2))
                logger.info("Saved %d structures to %s", len(chunk), path)
                file_list.append(str(path))
            paths[count] = file_list
        else:
            path = out / f"mc2d_{count}elements.json"
            path.write_text(json.dumps(group, indent=2))
            logger.info("Saved %d structures to %s", len(group), path)
            paths[count] = str(path)
    return paths


def write_selection_manifest(
    entries: Iterable[dict[str, Any]],
    structure_uuids: dict[str, str],
    path: str | Path,
) -> Path:
    """Write a CSV manifest of the selected overview entries (no structures)."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mc2d_id",
        "formula",
        "nelements",
        "band_gap",
        "magnetic_state",
        "unstable",
        "structure_uuid",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "mc2d_id": entry.get("id", ""),
                    "formula": entry.get("formula", ""),
                    "nelements": len(_elements(entry.get("formula"))),
                    "band_gap": "" if entry.get("bg") is None else entry.get("bg"),
                    "magnetic_state": entry.get("mags") or "",
                    "unstable": "" if entry.get("unstable") is None else entry.get("unstable"),
                    "structure_uuid": structure_uuids.get(entry.get("id"), ""),
                }
            )
    return path


def _sorted_lengths(matrix: Iterable[Iterable[float]]) -> tuple[float, ...]:
    return tuple(sorted(round(sum(float(x) ** 2 for x in vector) ** 0.5, 3) for vector in matrix))


def lattice_key_from_structure_dict(structure: dict[str, Any]) -> tuple[tuple[float, ...], int]:
    matrix = structure["lattice"]["matrix"]
    return _sorted_lengths(matrix), len(structure["sites"])


def lattice_key_from_cell(cell: dict[str, Any]) -> tuple[tuple[float, ...], int]:
    return _sorted_lengths(cell["cell"]), len(cell["sites"])


def lattice_key_from_structure(structure: Structure) -> tuple[tuple[float, ...], int]:
    return _sorted_lengths(structure.lattice.matrix), len(structure)


def inplane_key(key: tuple[tuple[float, ...], int]) -> tuple[tuple[float, ...], int]:
    """Keep the two shortest lattice lengths (the 2D plane) and the site count."""
    lengths, nsites = key
    return tuple(lengths[:2]), nsites


def geometry_match(a, b, tolerance: float, inplane: bool = False) -> bool:
    """Compare two lattice keys with a relative tolerance."""
    if a is None or b is None:
        return False
    if inplane:
        a = inplane_key(a)
        b = inplane_key(b)
    lengths_a, nsites_a = a
    lengths_b, nsites_b = b
    if nsites_a != nsites_b or len(lengths_a) != len(lengths_b):
        return False
    return all(abs(x - y) / max(y, 1e-6) <= tolerance for x, y in zip(lengths_a, lengths_b))


def build_uuid_index(structure_uuids: dict[str, str]) -> dict[str, str]:
    """Invert ``mc2d-N -> uuid`` into ``uuid -> mc2d-N``."""
    return {uuid: mc2d_id for mc2d_id, uuid in structure_uuids.items()}


def direct_mc2d_id(optimade_id: str | None, uuid_to_id: dict[str, str]) -> str | None:
    if not optimade_id:
        return None
    return uuid_to_id.get(optimade_id)
