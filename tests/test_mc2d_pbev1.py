"""Tests for the MC2D pbe-v1 REST helpers (no network access)."""

import csv
import json

import pytest
from pymatgen.core import Lattice, Structure

from aiida_gw.datasets import mc2d_pbev1 as mc2d


def _overview():
    return [
        {"id": "mc2d-1", "formula": "BN", "bg": 4.5, "mags": "non-magnetic", "unstable": False},
        {"id": "mc2d-2", "formula": "MoS2", "bg": 1.6, "mags": "non-magnetic", "unstable": False},
        {"id": "mc2d-3", "formula": "MgO", "bg": None, "mags": "non-magnetic", "unstable": None},
        {"id": "mc2d-4", "formula": "NaCl", "bg": 0.0, "mags": "non-magnetic", "unstable": True},
        {"id": "mc2d-5", "formula": "KBr", "bg": 2.0, "mags": "non-magnetic", "unstable": None},
        {"id": "mc2d-6", "formula": "LiCoO2", "bg": 1.0, "mags": "ferromagnetic", "unstable": False},
        {"id": "mc2d-7", "formula": "LaNiO3", "bg": 0.5, "mags": "non-magnetic", "unstable": False},
    ]


def test_select_entries_by_nelements():
    selected = mc2d.select_entries(_overview(), nelements=2)
    assert {entry["id"] for entry in selected} == {"mc2d-1", "mc2d-2", "mc2d-3", "mc2d-4", "mc2d-5"}

    selected = mc2d.select_entries(_overview(), min_nelements=3, max_nelements=3)
    assert {entry["id"] for entry in selected} == {"mc2d-6", "mc2d-7"}


def test_select_entries_excludes_elements_and_formulas():
    selected = mc2d.select_entries(_overview(), exclude_elements={"La"})
    assert "mc2d-7" not in {entry["id"] for entry in selected}

    selected = mc2d.select_entries(_overview(), formulas=["MoS2", "BN"])
    assert {entry["id"] for entry in selected} == {"mc2d-1", "mc2d-2"}


def test_select_entries_band_gap_semantics():
    selected = mc2d.select_entries(_overview(), min_band_gap=0.01)
    ids = {entry["id"] for entry in selected}
    assert "mc2d-3" not in ids  # unknown
    assert "mc2d-4" not in ids  # zero
    assert "mc2d-2" in ids

    selected = mc2d.select_entries(_overview(), min_band_gap=0.01, include_unknown_gap=True)
    assert "mc2d-3" in {entry["id"] for entry in selected}


def test_select_entries_stability_semantics():
    selected = mc2d.select_entries(_overview(), exclude_unstable=True)
    ids = {entry["id"] for entry in selected}
    assert "mc2d-4" not in ids  # explicitly unstable
    assert "mc2d-3" in ids  # unknown kept

    selected = mc2d.select_entries(_overview(), stable_only=True)
    ids = {entry["id"] for entry in selected}
    assert "mc2d-4" not in ids
    assert "mc2d-3" not in ids  # unknown dropped


def test_uuid_index_and_direct_match():
    uuids = {"mc2d-1": "aaa", "mc2d-2": "bbb"}
    index = mc2d.build_uuid_index(uuids)
    assert index == {"aaa": "mc2d-1", "bbb": "mc2d-2"}
    assert mc2d.direct_mc2d_id("bbb", index) == "mc2d-2"
    assert mc2d.direct_mc2d_id("ccc", index) is None
    assert mc2d.direct_mc2d_id(None, index) is None


def test_geometry_match_full_and_inplane():
    a = ((3.0, 3.0, 20.0), 3)
    b = ((3.0, 3.01, 20.0), 3)
    c = ((3.0, 3.0, 25.0), 3)
    assert mc2d.geometry_match(a, b, tolerance=0.02)
    assert not mc2d.geometry_match(a, c, tolerance=0.02)
    assert mc2d.geometry_match(a, c, tolerance=0.02, inplane=True)
    assert not mc2d.geometry_match(a, ((3.0, 3.0, 20.0), 4), tolerance=0.02)


def test_build_pbev1_entry():
    structure = Structure(Lattice.cubic(3.0), ["C"], [[0.0, 0.0, 0.0]])
    entry = mc2d.build_pbev1_entry(
        {"id": "mc2d-2", "formula": "C", "bg": 1.23, "mags": "non-magnetic", "unstable": False},
        structure,
        structure_uuid="abc",
    )
    assert entry["id"] == "mc2d-2"
    assert entry["mc2d_id"] == "mc2d-2"
    assert entry["band_gap"] == 1.23
    assert entry["elements"] == ["C"]
    assert entry["nsites"] == 1
    assert entry["structure_uuid"] == "abc"
    assert entry["dataset_version"] == "pbe-v1"
    assert entry["structure"]["lattice"]["matrix"][0][0] == pytest.approx(3.0)


def test_save_by_nelements(tmp_path):
    entries = [
        {"id": "mc2d-1", "nelements": 2, "formula": "BN"},
        {"id": "mc2d-2", "nelements": 2, "formula": "MoS2"},
        {"id": "mc2d-3", "nelements": 3, "formula": "LiCoO2"},
    ]
    paths = mc2d.save_by_nelements(entries, tmp_path)
    assert set(paths) == {2, 3}
    assert json.loads((tmp_path / "mc2d_2elements.json").read_text())[0]["id"] == "mc2d-1"
    assert json.loads((tmp_path / "mc2d_3elements.json").read_text())[0]["id"] == "mc2d-3"


def test_save_by_nelements_splits_files(tmp_path):
    entries = [{"id": f"mc2d-{i}", "nelements": 2} for i in range(5)]
    paths = mc2d.save_by_nelements(entries, tmp_path, structures_per_file=2)
    assert isinstance(paths[2], list)
    assert len(paths[2]) == 3


def test_write_selection_manifest(tmp_path):
    path = mc2d.write_selection_manifest(
        [{"id": "mc2d-2", "formula": "MoS2", "bg": 1.6, "mags": "non-magnetic", "unstable": False}],
        {"mc2d-2": "abc"},
        tmp_path / "manifest.csv",
    )
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["mc2d_id"] == "mc2d-2"
    assert rows[0]["band_gap"] == "1.6"
    assert rows[0]["structure_uuid"] == "abc"
