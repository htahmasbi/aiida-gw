"""Tests for the legacy-to-pbe-v1 crosswalk utility (no network access)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

UTIL_PATH = Path(__file__).resolve().parents[1] / "utils" / "annotate_mc2d_bandgaps.py"
SPEC = importlib.util.spec_from_file_location("annotate_mc2d_bandgaps", UTIL_PATH)
crosswalk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(crosswalk)


OVERVIEW = [
    {"id": "mc2d-5", "formula": "AgI", "bg": 2.15, "mags": "non-magnetic", "unstable": False},
    {"id": "mc2d-9", "formula": "C2", "bg": 1.5, "mags": "non-magnetic", "unstable": False},
]
OVERVIEW_BY_ID = {entry["id"]: entry for entry in OVERVIEW}
UUID_TO_ID = {"legacy-agi-uuid": "mc2d-5"}
BY_ELEMENTS = {("C",): [OVERVIEW[1]]}
BY_FORMULA = {"C2": [OVERVIEW[1]]}


@pytest.fixture
def structure():
    return Structure(
        Lattice.from_parameters(3.0, 3.0, 20.0, 90, 90, 90),
        ["C", "C"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _cache():
    return crosswalk.Cache(None)


def test_direct_uuid_match_uses_overview_metadata():
    entry = {"id": "legacy-agi-uuid", "formula": "AgI", "elements": ["Ag", "I"], "nsites": 4}
    annotation = crosswalk.match_entry(
        entry,
        OVERVIEW_BY_ID,
        UUID_TO_ID,
        BY_ELEMENTS,
        BY_FORMULA,
        _cache(),
        tolerance=0.02,
        sleep=0.0,
    )
    assert annotation["mc2d_id"] == "mc2d-5"
    assert annotation["band_gap"] == 2.15
    assert annotation["magnetic_state"] == "non-magnetic"
    assert annotation["band_gap_match"] == "uuid"
    assert annotation["structure_uuid"] == "legacy-agi-uuid"


def test_geometry_fallback(structure, monkeypatch):
    core = {
        "general": {"structure_relaxed_uuid": "relaxed-uuid"},
        "electronic": {"band_gap": {"value": 1.5}, "magnetic_state": "non-magnetic"},
        "vibrational": {"unstable": False},
    }
    cell = {"cell": structure.lattice.matrix.tolist(), "sites": [{}, {}]}
    monkeypatch.setattr(crosswalk, "get_core", lambda *args, **kwargs: core)
    monkeypatch.setattr(crosswalk, "get_cell", lambda *args, **kwargs: cell)

    entry = {
        "id": "unknown-legacy-uuid",
        "formula": "C2",
        "elements": ["C"],
        "nsites": 2,
        "structure": structure.as_dict(),
    }
    annotation = crosswalk.match_entry(
        entry,
        OVERVIEW_BY_ID,
        UUID_TO_ID,
        BY_ELEMENTS,
        BY_FORMULA,
        _cache(),
        tolerance=0.02,
        sleep=0.0,
    )
    assert annotation["mc2d_id"] == "mc2d-9"
    assert annotation["band_gap"] == 1.5
    assert annotation["band_gap_match"] == "geometry:relaxed"


def test_geometry_ambiguity_uses_conservative_min_gap(structure, monkeypatch):
    core_a = {
        "general": {"structure_relaxed_uuid": "uuid-a"},
        "electronic": {"band_gap": {"value": 0.0}},
        "vibrational": {"unstable": True},
    }
    core_b = {
        "general": {"structure_relaxed_uuid": "uuid-b"},
        "electronic": {"band_gap": {"value": 1.6}},
        "vibrational": {"unstable": False},
    }
    cores = {"mc2d-9": core_a, "mc2d-10": core_b}
    cell = {"cell": structure.lattice.matrix.tolist(), "sites": [{}, {}]}
    monkeypatch.setattr(crosswalk, "get_core", lambda mc2d_id, *args, **kwargs: cores[mc2d_id])
    monkeypatch.setattr(crosswalk, "get_cell", lambda *args, **kwargs: cell)

    overview = [
        {"id": "mc2d-9", "formula": "C2", "bg": 0.0, "mags": "non-magnetic", "unstable": True},
        {"id": "mc2d-10", "formula": "C2", "bg": 1.6, "mags": "non-magnetic", "unstable": False},
    ]
    overview_by_id = {entry["id"]: entry for entry in overview}
    by_elements = {("C",): [overview[0], overview[1]]}
    by_formula = {"C2": [overview[0], overview[1]]}
    entry = {
        "id": "unknown-legacy-uuid",
        "formula": "C2",
        "elements": ["C"],
        "nsites": 2,
        "structure": structure.as_dict(),
    }
    annotation = crosswalk.match_entry(
        entry,
        overview_by_id,
        UUID_TO_ID,
        by_elements,
        by_formula,
        _cache(),
        tolerance=0.02,
        sleep=0.0,
    )
    assert annotation["mc2d_id"] is None
    assert annotation["band_gap"] == 0.0
    assert annotation["band_gap_match"] == "ambiguous_min_gap"
    assert annotation["mc2d_candidates"] == ["mc2d-9", "mc2d-10"]


class _Extras:
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value


class _FakeNode:
    def __init__(self, structure, optimade_id=None, pk=1, uuid="node-uuid"):
        self.pk = pk
        self.uuid = uuid
        self._structure = structure
        self.base = SimpleNamespace(extras=_Extras())
        if optimade_id is not None:
            self.base.extras.set("optimade_id", optimade_id)

    def get_pymatgen(self):
        return self._structure


def test_node_to_entry(structure):
    node = _FakeNode(structure, optimade_id="legacy-agi-uuid")
    entry, structure_node = crosswalk.node_to_entry(node)
    assert structure_node is node
    assert entry["id"] == "legacy-agi-uuid"
    assert entry["elements"] == ["C"]
    assert entry["nsites"] == 2
    assert entry["_prepared"] is True


def test_annotate_nodes_sidecar_row(structure):
    node = _FakeNode(structure, optimade_id="legacy-agi-uuid", pk=42, uuid="workflow-uuid")
    rows = crosswalk.annotate_nodes(
        [{"node": node, "group": "gw_runs_2el"}],
        overview_by_id=OVERVIEW_BY_ID,
        uuid_to_id=UUID_TO_ID,
        by_elements=BY_ELEMENTS,
        by_formula=BY_FORMULA,
        cache=_cache(),
        tolerance=0.02,
        sleep=0.0,
    )
    assert len(rows) == 1
    assert rows[0]["source"] == "node"
    assert rows[0]["group"] == "gw_runs_2el"
    assert rows[0]["pk"] == 42
    assert rows[0]["mc2d_id"] == "mc2d-5"
    assert rows[0]["band_gap_match"] == "uuid"


def test_annotate_json_file_clears_stale_fields(structure, tmp_path):
    entry = {
        "id": "legacy-agi-uuid",
        "formula": "AgI",
        "elements": ["Ag", "I"],
        "nsites": 2,
        "structure": structure.as_dict(),
        "mc2d_id": "stale",
        "mc2d_candidates": ["stale"],
        "band_gap_match": "stale",
    }
    rows = crosswalk.annotate_json_file(
        tmp_path / "mc2d_2elements.json",
        [entry],
        overview_by_id=OVERVIEW_BY_ID,
        uuid_to_id=UUID_TO_ID,
        by_elements=BY_ELEMENTS,
        by_formula=BY_FORMULA,
        cache=_cache(),
        tolerance=0.02,
        sleep=0.0,
    )
    assert entry["mc2d_id"] == "mc2d-5"
    assert "mc2d_candidates" not in entry
    assert entry["band_gap_match"] == "uuid"
    assert rows[0]["mc2d_id"] == "mc2d-5"
