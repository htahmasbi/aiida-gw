"""Tests for submission-time band-gap filtering in ``aiida_gw.cli``."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

from aiida_gw.cli import _band_gap_allowed, _load_structures_from_json


@pytest.mark.parametrize(
    ("band_gap", "min_gap", "include_unknown", "expected"),
    [
        (0.5, 0.01, False, True),
        (0.01, 0.01, False, False),
        (0.0, 0.01, False, False),
        (None, 0.01, False, False),
        (None, 0.01, True, True),
        ("2.1", 0.01, False, True),
        ("bad", 0.01, False, False),
        ("bad", 0.01, True, True),
        (None, None, False, True),
    ],
)
def test_band_gap_allowed(band_gap, min_gap, include_unknown, expected):
    assert _band_gap_allowed(band_gap, min_gap, include_unknown) is expected


class _Extras:
    def __init__(self):
        self._data = {}

    def set(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


class _FakeStructure:
    def __init__(self, pymatgen=None):
        self.pymatgen = pymatgen
        self.base = SimpleNamespace(extras=_Extras())

    def store(self):
        return self


@pytest.fixture
def aiida_orm(monkeypatch):
    """Return ``aiida.orm``, installing a temporary stub when aiida is absent."""
    try:
        import aiida.orm as orm

        return orm
    except ImportError:
        pass
    aiida = types.ModuleType("aiida")
    orm = types.ModuleType("aiida.orm")
    orm.StructureData = object
    aiida.orm = orm
    monkeypatch.setitem(sys.modules, "aiida", aiida)
    monkeypatch.setitem(sys.modules, "aiida.orm", orm)
    return orm


@pytest.fixture
def sample_json(tmp_path):
    structure = Structure(Lattice.cubic(3.0), ["C"], [[0.0, 0.0, 0.0]]).as_dict()
    entries = [
        {"id": "low", "formula": "C", "elements": ["C"], "structure": structure, "mc2d_id": "mc2d-1", "band_gap": 0.0},
        {"id": "ok", "formula": "C", "elements": ["C"], "structure": structure, "mc2d_id": "mc2d-2", "band_gap": 0.5},
        {"id": "unknown", "formula": "C", "elements": ["C"], "structure": structure},
    ]
    path = Path(tmp_path) / "mc2d_2elements.json"
    path.write_text(json.dumps(entries))
    return path


def test_loader_filters_low_and_unknown_gaps(sample_json, monkeypatch, aiida_orm):
    monkeypatch.setattr(aiida_orm, "StructureData", _FakeStructure)

    nodes = _load_structures_from_json(
        [sample_json],
        None,
        None,
        "X",
        10,
        object(),
        "no structures",
        min_band_gap=0.01,
        include_unknown_gap=False,
    )

    assert len(nodes) == 1
    assert nodes[0].base.extras.get("mc2d_id") == "mc2d-2"
    assert nodes[0].base.extras.get("band_gap") == 0.5


def test_loader_keeps_unknown_when_requested(sample_json, monkeypatch, aiida_orm):
    monkeypatch.setattr(aiida_orm, "StructureData", _FakeStructure)

    nodes = _load_structures_from_json(
        [sample_json],
        None,
        None,
        "X",
        10,
        object(),
        "no structures",
        min_band_gap=0.01,
        include_unknown_gap=True,
    )

    by_id = {node.base.extras.get("optimade_id"): node for node in nodes}
    assert set(by_id) == {"ok", "unknown"}
    assert by_id["ok"].base.extras.get("mc2d_id") == "mc2d-2"
    assert by_id["unknown"].base.extras.get("band_gap") is None


def test_loader_without_filter_keeps_all(sample_json, monkeypatch, aiida_orm):
    monkeypatch.setattr(aiida_orm, "StructureData", _FakeStructure)

    nodes = _load_structures_from_json([sample_json], None, None, "X", 10, object(), "no structures")
    assert len(nodes) == 3


def test_loader_pbev1_extras(tmp_path, monkeypatch, aiida_orm):
    monkeypatch.setattr(aiida_orm, "StructureData", _FakeStructure)
    structure = Structure(Lattice.cubic(3.0), ["C"], [[0.0, 0.0, 0.0]]).as_dict()
    entries = [
        {
            "id": "mc2d-9",
            "dataset_version": "pbe-v1",
            "mc2d_id": "mc2d-9",
            "structure_uuid": "relaxed-uuid",
            "formula": "C",
            "elements": ["C"],
            "structure": structure,
            "band_gap": 1.5,
            "magnetic_state": "non-magnetic",
            "unstable": False,
        }
    ]
    path = Path(tmp_path) / "mc2d_3elements.json"
    path.write_text(json.dumps(entries))

    nodes = _load_structures_from_json([path], None, None, "X", 10, object(), "no structures")

    extras = nodes[0].base.extras._data
    assert extras["source"] == "mc2d_pbev1"
    assert extras["mc2d_id"] == "mc2d-9"
    assert extras["structure_uuid"] == "relaxed-uuid"
    assert extras["dataset_version"] == "pbe-v1"
    assert extras["band_gap"] == 1.5
    assert "optimade_id" not in extras
