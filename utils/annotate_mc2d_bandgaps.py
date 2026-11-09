#!/usr/bin/env python3
"""Crosswalk legacy MC2D records to current ``pbe-v1`` IDs and band gaps.

The legacy OPTIMADE endpoint used by the original 2-element GW runs exposes
AiiDA node UUIDs and a sparse ``_mcloud_band_gap``.  The current MC2D data
(``mc2d/pbe-v1``) exposes stable ``mc2d-N`` IDs and complete band gaps.

Matching is UUID-first: ~98% of legacy OPTIMADE IDs are exactly the pbe-v1
relaxed-structure UUIDs returned by ``/mc2d/pbe-v1/structure-uuids``.  The
remaining entries are matched by element set, site count and lattice geometry.

Two input modes are supported:

* JSON files from ``aiida-gw fetch-json`` (legacy OPTIMADE output).  The files
  can be annotated in place or copied to ``--output-dir``; ``--report-only``
  writes only the report.
* AiiDA nodes (``--nodes``) from ``--group`` / ``--node-pk``.  A sidecar CSV
  and JSON are written; **AiiDA nodes are never modified**.

The script does not write extras to nodes.  Use the sidecar mapping to join
legacy 2-element results with pbe-v1 band gaps.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from pymatgen.core import Structure

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aiida_gw.datasets.mc2d_pbev1 import (  # noqa: E402
    AIIDA_ATTRS_URL,
    MC2D_CORE_URL,
    build_uuid_index,
    direct_mc2d_id,
    geometry_match,
    inplane_key,
    lattice_key_from_cell,
    lattice_key_from_structure,
    lattice_key_from_structure_dict,
    load_or_fetch_overview,
    load_or_fetch_structure_uuids,
)

LOGGER = logging.getLogger("annotate_mc2d_bandgaps")

HEADERS = {"User-Agent": "aiida-gw-mc2d-crosswalk/2.0", "Accept": "application/json"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

FIELDNAMES = [
    "source",
    "group",
    "file",
    "pk",
    "structure_pk",
    "uuid",
    "input_id",
    "formula",
    "nelements",
    "mc2d_id",
    "structure_uuid",
    "band_gap",
    "magnetic_state",
    "unstable",
    "band_gap_match",
    "mc2d_candidates",
]

ANNOTATION_KEYS = (
    "mc2d_id",
    "band_gap",
    "magnetic_state",
    "unstable",
    "band_gap_match",
    "mc2d_candidates",
    "structure_uuid",
)


class Cache:
    """On-disk cache for core_base and structure-attribute responses."""

    def __init__(self, path: Path | None):
        self.path = path
        self.data = {"core": {}, "cell": {}}
        if path and path.is_file():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict):
                    self.data["core"] = loaded.get("core", {})
                    self.data["cell"] = loaded.get("cell", {})
            except Exception as exc:  # pragma: no cover - cache is best effort
                LOGGER.warning("Could not read cache %s: %s", path, exc)

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not write cache %s: %s", self.path, exc)

    def get_core(self, mc2d_id: str):
        return self.data["core"].get(mc2d_id)

    def set_core(self, mc2d_id: str, value) -> None:
        self.data["core"][mc2d_id] = value

    def get_cell(self, uuid: str):
        return self.data["cell"].get(uuid)

    def set_cell(self, uuid: str, value) -> None:
        self.data["cell"][uuid] = value


def _get_json(url: str, timeout: int = 60):
    response = SESSION.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.json()


def get_core(mc2d_id: str, cache: Cache, sleep: float = 0.0):
    cached = cache.get_core(mc2d_id)
    if cached is not None:
        return cached
    try:
        data = _get_json(MC2D_CORE_URL.format(mc2d_id=mc2d_id), timeout=30)
    except Exception as exc:
        LOGGER.debug("core_base %s failed: %s", mc2d_id, exc)
        return None
    cache.set_core(mc2d_id, data)
    if sleep:
        import time

        time.sleep(sleep)
    return data


def get_cell(uuid: str, cache: Cache):
    cached = cache.get_cell(uuid)
    if cached is not None:
        return cached
    try:
        data = _get_json(AIIDA_ATTRS_URL.format(uuid=uuid), timeout=30)
        attributes = data["data"]["attributes"]
        cell = {"cell": attributes["cell"], "sites": attributes["sites"]}
    except Exception as exc:
        LOGGER.debug("attributes %s failed: %s", uuid, exc)
        return None
    cache.set_cell(uuid, cell)
    return cell


def _overview_index(overview: list[dict]) -> dict[str, dict]:
    return {entry.get("id"): entry for entry in overview if entry.get("id")}


def _index_by_elements(overview: list[dict]) -> dict[tuple[str, ...], list[dict]]:
    index: dict[tuple[str, ...], list[dict]] = {}
    for entry in overview:
        elements = tuple(sorted(_elements(entry.get("formula"))))
        if elements:
            index.setdefault(elements, []).append(entry)
    return index


def _elements(formula: str | None):
    from pymatgen.core import Composition

    if not formula:
        return []
    try:
        return [str(element) for element in Composition(formula).elements]
    except Exception:
        return []


def _index_by_formula(overview: list[dict]) -> dict[str, list[dict]]:
    from pymatgen.core import Composition

    index: dict[str, list[dict]] = {}
    for entry in overview:
        formula = entry.get("formula")
        if not formula:
            continue
        try:
            reduced = Composition(formula).reduced_formula
        except Exception:
            reduced = formula
        index.setdefault(reduced, []).append(entry)
    return index


def _dedupe(candidates: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for candidate in candidates:
        key = candidate.get("id")
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _prepared_key(structure: Structure) -> tuple[tuple[float, ...], int]:
    try:
        primitive = structure.get_primitive_structure()
    except Exception:
        primitive = structure
    return inplane_key(lattice_key_from_structure(primitive))


def _entry_key(entry: dict) -> tuple[tuple[tuple[float, ...], int] | None, bool]:
    structure_dict = entry.get("structure")
    if not structure_dict:
        return None, bool(entry.get("_prepared"))
    try:
        structure = Structure.from_dict(structure_dict)
    except Exception:
        return None, bool(entry.get("_prepared"))
    if entry.get("_prepared"):
        return _prepared_key(structure), True
    return lattice_key_from_structure_dict(structure_dict), False


def _annotation_from_overview(
    mc2d_id: str,
    overview_by_id: dict[str, dict],
    match: str,
    structure_uuid: str | None = None,
) -> dict:
    overview_entry = overview_by_id.get(mc2d_id, {})
    return {
        "mc2d_id": mc2d_id,
        "band_gap": overview_entry.get("bg"),
        "magnetic_state": overview_entry.get("mags"),
        "unstable": overview_entry.get("unstable"),
        "band_gap_match": match,
        "structure_uuid": structure_uuid,
    }


def _candidate_metadata(core: dict, overview_entry: dict | None) -> tuple:
    band_gap = None
    magnetic_state = None
    unstable = None
    if overview_entry is not None:
        band_gap = overview_entry.get("bg")
        magnetic_state = overview_entry.get("mags")
        unstable = overview_entry.get("unstable")
    if band_gap is None:
        try:
            band_gap = core["electronic"]["band_gap"]["value"]
        except Exception:
            pass
    if magnetic_state is None:
        try:
            magnetic_state = core["electronic"].get("magnetic_state")
        except Exception:
            pass
    if unstable is None:
        try:
            unstable = core["vibrational"].get("unstable")
        except Exception:
            pass
    return band_gap, magnetic_state, unstable


def match_entry(
    entry: dict,
    overview_by_id: dict[str, dict],
    uuid_to_id: dict[str, str],
    by_elements: dict,
    by_formula: dict,
    cache: Cache,
    tolerance: float,
    sleep: float,
) -> dict:
    """Return annotation fields for one JSON entry or node-derived entry."""
    legacy_id = entry.get("id")
    direct_id = direct_mc2d_id(legacy_id, uuid_to_id)
    if direct_id:
        return _annotation_from_overview(
            direct_id,
            overview_by_id,
            match="uuid",
            structure_uuid=legacy_id,
        )

    legacy_key, inplane = _entry_key(entry)
    legacy_nsites = entry.get("nsites")
    if legacy_nsites is None and legacy_key is not None:
        legacy_nsites = legacy_key[1]

    formula = entry.get("formula")
    elements = tuple(sorted(entry.get("elements") or _elements(formula)))
    candidates = _dedupe(by_elements.get(elements, []))
    used_fallback = False
    if not candidates:
        from pymatgen.core import Composition

        try:
            reduced = Composition(formula).reduced_formula if formula else None
        except Exception:
            reduced = formula
        candidates = _dedupe(by_formula.get(reduced, [])) if reduced else []
        used_fallback = True

    hits = []
    for candidate in candidates:
        mc2d_id = candidate.get("id")
        if not mc2d_id:
            continue
        core = get_core(mc2d_id, cache, sleep=sleep)
        if not core:
            continue
        general = core.get("general", {})
        for geometry, key in (
            ("relaxed", "structure_relaxed_uuid"),
            ("extracted", "structure_extracted_uuid"),
        ):
            uuid = general.get(key)
            if not uuid:
                continue
            cell = get_cell(uuid, cache)
            if cell is None:
                continue
            if legacy_key is not None:
                matched = geometry_match(
                    legacy_key,
                    lattice_key_from_cell(cell),
                    tolerance,
                    inplane=inplane,
                )
            elif legacy_nsites is not None:
                matched = len(cell["sites"]) == legacy_nsites
            else:
                matched = False
            if matched:
                band_gap, magnetic_state, unstable = _candidate_metadata(core, candidate)
                hits.append(
                    {
                        "mc2d_id": mc2d_id,
                        "band_gap": band_gap,
                        "magnetic_state": magnetic_state,
                        "unstable": unstable,
                        "geometry": geometry,
                    }
                )
                break

    if len(hits) == 1:
        hit = hits[0]
        match = f"geometry:{hit['geometry']}"
        if inplane:
            match += ":prepared"
        if used_fallback:
            match += ":formula_fallback"
        return {
            "mc2d_id": hit["mc2d_id"],
            "band_gap": hit["band_gap"],
            "magnetic_state": hit["magnetic_state"],
            "unstable": hit["unstable"],
            "band_gap_match": match,
        }

    if len(hits) > 1:
        candidate_ids = [hit["mc2d_id"] for hit in hits]
        gaps = [hit["band_gap"] for hit in hits]
        if all(gap is not None for gap in gaps) and len(set(gaps)) == 1:
            return {
                "mc2d_id": None,
                "band_gap": gaps[0],
                "magnetic_state": None,
                "unstable": None,
                "band_gap_match": "ambiguous_same_gap",
                "mc2d_candidates": candidate_ids,
            }
        if all(gap is not None for gap in gaps):
            return {
                "mc2d_id": None,
                "band_gap": min(gaps),
                "magnetic_state": None,
                "unstable": None,
                "band_gap_match": "ambiguous_min_gap",
                "mc2d_candidates": candidate_ids,
            }
        return {
            "mc2d_id": None,
            "band_gap": None,
            "magnetic_state": None,
            "unstable": None,
            "band_gap_match": "ambiguous",
            "mc2d_candidates": candidate_ids,
        }

    if len(candidates) == 1:
        candidate = candidates[0]
        mc2d_id = candidate.get("id")
        core = get_core(mc2d_id, cache, sleep=sleep) if mc2d_id else None
        if core:
            general = core.get("general", {})
            nsites_match = False
            for key in ("structure_relaxed_uuid", "structure_extracted_uuid"):
                uuid = general.get(key)
                if not uuid:
                    continue
                cell = get_cell(uuid, cache)
                if cell is not None and legacy_nsites is not None and len(cell["sites"]) == legacy_nsites:
                    nsites_match = True
                    break
            if nsites_match:
                band_gap, magnetic_state, unstable = _candidate_metadata(core, candidate)
                return {
                    "mc2d_id": mc2d_id,
                    "band_gap": band_gap,
                    "magnetic_state": magnetic_state,
                    "unstable": unstable,
                    "band_gap_match": "formula_unique",
                }

    return {
        "mc2d_id": None,
        "band_gap": None,
        "magnetic_state": None,
        "unstable": None,
        "band_gap_match": "not_found",
        "mc2d_candidates": [candidate.get("id") for candidate in candidates if candidate.get("id")],
    }


def _row_for_entry(entry: dict, annotation: dict, source: str, **extra) -> dict:
    row = {field: "" for field in FIELDNAMES}
    row.update(extra)
    row.update(
        {
            "source": source,
            "input_id": entry.get("id", ""),
            "formula": entry.get("formula", ""),
            "nelements": entry.get("nelements", ""),
            "mc2d_id": annotation.get("mc2d_id") or "",
            "structure_uuid": annotation.get("structure_uuid") or "",
            "band_gap": "" if annotation.get("band_gap") is None else annotation.get("band_gap"),
            "magnetic_state": annotation.get("magnetic_state") or "",
            "unstable": "" if annotation.get("unstable") is None else annotation.get("unstable"),
            "band_gap_match": annotation.get("band_gap_match", ""),
            "mc2d_candidates": ";".join(annotation.get("mc2d_candidates") or []),
        }
    )
    return row


def _resolve_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(Path(item) for item in glob.glob(str(path / "mc2d_*.json"))))
        elif path.is_file():
            files.append(path)
        else:
            LOGGER.warning("Input not found: %s", raw)
    seen = set()
    unique = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def annotate_json_file(path: Path, entries: list[dict], **match_kwargs) -> list[dict]:
    rows = []
    counts: dict[str, int] = {}
    for index, entry in enumerate(entries):
        for key in ANNOTATION_KEYS:
            entry.pop(key, None)
        annotation = match_entry(entry, **match_kwargs)
        entry.update(annotation)
        match_type = annotation.get("band_gap_match", "")
        bucket = "uuid" if match_type == "uuid" else match_type.split(":")[0]
        counts[bucket] = counts.get(bucket, 0) + 1
        rows.append(_row_for_entry(entry, annotation, "json", file=str(path)))
        if (index + 1) % 25 == 0:
            LOGGER.info("  %s: %d/%d entries processed", path.name, index + 1, len(entries))
    LOGGER.info("  %s: %s", path.name, counts)
    return rows


def _node_structure(node):
    if hasattr(node, "get_pymatgen") and hasattr(node, "base"):
        return node
    inputs = getattr(node, "inputs", None)
    if inputs is None:
        return None
    for name in ("structure", "input_structure", "initial_structure", "base_structure"):
        try:
            candidate = inputs[name]
        except Exception:
            continue
        if hasattr(candidate, "get_pymatgen"):
            return candidate
    def search(mapping, depth=0):
        if depth > 4:
            return None
        try:
            for key in mapping:
                try:
                    value = mapping[key]
                except Exception:
                    continue
                if hasattr(value, "get_pymatgen") and hasattr(value, "base"):
                    return value
                found = search(value, depth + 1) if hasattr(value, "__getitem__") else None
                if found is not None:
                    return found
        except Exception:
            pass
        return None

    return search(inputs)


def node_to_entry(node) -> tuple[dict | None, object | None]:
    structure_node = _node_structure(node)
    if structure_node is None:
        return None, None
    try:
        pymatgen_structure = structure_node.get_pymatgen()
    except Exception:
        return None, structure_node
    extras = structure_node.base.extras
    optimade_id = extras.get("optimade_id") or extras.get("mc2d_id") or ""
    formula = extras.get("formula") or pymatgen_structure.composition.reduced_formula
    elements = sorted(str(element) for element in pymatgen_structure.composition.elements)
    entry = {
        "id": optimade_id or str(structure_node.uuid),
        "formula": formula,
        "elements": elements,
        "nelements": len(elements),
        "nsites": len(pymatgen_structure),
        "structure": pymatgen_structure.as_dict(),
        "_prepared": True,
    }
    return entry, structure_node


def load_nodes(groups: list[str], node_pks: list[int], limit: int | None = None) -> list[dict]:
    try:
        from aiida import load_profile
        from aiida.orm import Group, load_node

        try:
            load_profile()
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - depends on AiiDA environment
        raise RuntimeError(f"Loading AiiDA nodes requires an AiiDA environment: {exc}") from exc

    selected: list[dict] = []
    seen = set()
    for label in groups:
        group = Group.collection.get(label=label)
        for node in group.nodes:
            if node.pk in seen:
                continue
            seen.add(node.pk)
            selected.append({"node": node, "group": label})
    for pk in node_pks:
        node = load_node(pk)
        if node.pk in seen:
            continue
        seen.add(node.pk)
        selected.append({"node": node, "group": ""})
    if limit is not None:
        selected = selected[:limit]
    return selected


def annotate_nodes(selected: list[dict], **match_kwargs) -> list[dict]:
    rows = []
    for index, item in enumerate(selected):
        node = item["node"]
        entry, structure_node = node_to_entry(node)
        if entry is None:
            rows.append(
                {
                    **{field: "" for field in FIELDNAMES},
                    "source": "node",
                    "group": item["group"],
                    "pk": node.pk,
                    "uuid": node.uuid,
                    "band_gap_match": "no_structure",
                }
            )
            continue
        annotation = match_entry(entry, **match_kwargs)
        rows.append(
            _row_for_entry(
                entry,
                annotation,
                "node",
                group=item["group"],
                pk=node.pk,
                structure_pk=structure_node.pk,
                uuid=node.uuid,
            )
        )
        if (index + 1) % 25 == 0:
            LOGGER.info("  %d/%d nodes processed", index + 1, len(selected))
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote report %s", path)


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote sidecar %s", path)


def _summary(rows: list[dict]) -> None:
    uniquely = sum(1 for row in rows if row["mc2d_id"])
    known = 0
    low = 0
    for row in rows:
        if row["band_gap"] == "":
            continue
        try:
            value = float(row["band_gap"])
        except (TypeError, ValueError):
            continue
        known += 1
        if value <= 0.010:
            low += 1
    LOGGER.info(
        "Summary: %d/%d uniquely matched, %d with known gap, %d with gap <= 0.010 eV",
        uniquely,
        len(rows),
        known,
        low,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="JSON files or directories containing mc2d_*.json")
    parser.add_argument("--nodes", action="store_true", help="Crosswalk AiiDA nodes instead of JSON files")
    parser.add_argument("--group", action="append", default=[], help="AiiDA group label (repeatable)")
    parser.add_argument("--groups", default=None, help="Comma-separated AiiDA group labels")
    parser.add_argument("--node-pk", action="append", type=int, default=[], help="AiiDA node PK (repeatable)")
    parser.add_argument("--output-dir", default=None, help="Write annotated JSON here (default: in place)")
    parser.add_argument("--report", default=None, help="CSV report path")
    parser.add_argument("--json-output", default=None, help="Sidecar JSON path (node mode)")
    parser.add_argument("--overview-json", default=None, help="Cache/reuse the pbe-v1 overview JSON")
    parser.add_argument("--structure-uuids-json", default=None, help="Cache/reuse the mc2d-N -> UUID map")
    parser.add_argument("--cache", default=".mc2d_crosswalk_cache.json", help="Cache file ('none' to disable)")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Relative lattice-length tolerance")
    parser.add_argument("--sleep", type=float, default=0.02, help="Delay between core_base requests")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many entries/nodes")
    parser.add_argument("--report-only", action="store_true", help="Write reports but do not rewrite JSON files")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    groups = list(args.group)
    if args.groups:
        groups.extend(item.strip() for item in args.groups.split(",") if item.strip())
    groups = list(dict.fromkeys(groups))

    if args.nodes and not (groups or args.node_pk):
        LOGGER.error("--nodes requires --group/--groups or --node-pk")
        return 1
    if not args.nodes and not args.inputs:
        LOGGER.error("Provide JSON inputs, or use --nodes with --group/--node-pk")
        return 1

    overview = load_or_fetch_overview(args.overview_json)
    structure_uuids = load_or_fetch_structure_uuids(args.structure_uuids_json)
    overview_by_id = _overview_index(overview)
    uuid_to_id = build_uuid_index(structure_uuids)
    by_elements = _index_by_elements(overview)
    by_formula = _index_by_formula(overview)
    LOGGER.info(
        "Indexed %d pbe-v1 entries (%d direct UUID matches available)",
        len(overview),
        len(uuid_to_id),
    )

    cache_path = None if args.cache.lower() == "none" else Path(args.cache)
    cache = Cache(cache_path)
    match_kwargs = {
        "overview_by_id": overview_by_id,
        "uuid_to_id": uuid_to_id,
        "by_elements": by_elements,
        "by_formula": by_formula,
        "cache": cache,
        "tolerance": args.tolerance,
        "sleep": args.sleep,
    }

    if args.nodes:
        selected = load_nodes(groups, args.node_pk, limit=args.limit)
        LOGGER.info("Processing %d AiiDA node(s)", len(selected))
        rows = annotate_nodes(selected, **match_kwargs)
        report_path = Path(args.report) if args.report else Path("mc2d_node_crosswalk.csv")
        json_path = Path(args.json_output) if args.json_output else report_path.with_suffix(".json")
        write_csv(rows, report_path)
        write_json(rows, json_path)
    else:
        inputs = _resolve_inputs(args.inputs)
        if not inputs:
            LOGGER.error("No input JSON files found")
            return 1

        all_rows = []
        for path in inputs:
            entries = json.loads(path.read_text())
            if not isinstance(entries, list):
                LOGGER.error("%s: expected a JSON list of structures", path)
                return 1
            process_entries = entries[: args.limit] if args.limit else entries
            LOGGER.info("Processing %s (%d entries)", path, len(process_entries))
            all_rows.extend(annotate_json_file(path, process_entries, **match_kwargs))
            if not args.report_only:
                out_dir = Path(args.output_dir) if args.output_dir else path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / path.name
                out_path.write_text(json.dumps(entries, indent=2))
                LOGGER.info("Wrote %s", out_path)

        report_path = Path(args.report) if args.report else None
        if report_path is None:
            base_dir = Path(args.output_dir) if args.output_dir else inputs[0].parent
            report_path = base_dir / "mc2d_bandgap_report.csv"
        write_csv(all_rows, report_path)
        if args.json_output:
            write_json(all_rows, Path(args.json_output))
        rows = all_rows

    cache.save()
    _summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
