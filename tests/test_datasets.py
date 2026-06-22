"""Tests for MC2D OPTIMADE dataset fetching and JSON save/load."""

import json
import tempfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

from aiida_gw.datasets.mc2d_optimade import (
    MC2D_STRUCTURES_URL,
    _get_structure_elements,
    _structure_has_unsupported_elements,
    fetch_all_mc2d,
    fetch_mc2d_structures,
    optimade_entry_to_pymatgen,
    save_mc2d_by_nelements,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_ENTRY = {
    "id": "mc2d_001",
    "type": "structures",
    "attributes": {
        "chemical_formula_reduced": "BN",
        "chemical_formula_descriptive": "B1 N1",
        "elements": ["B", "N"],
        "nelements": 2,
        "nsites": 2,
        "lattice_vectors": [[2.5, 0.0, 0.0], [-1.25, 2.165, 0.0], [0.0, 0.0, 15.0]],
        "cartesian_site_positions": [[0.0, 0.0, 0.0], [1.25, 0.722, 0.0]],
        "species_at_sites": ["B", "N"],
        "species": [{"name": "B"}, {"name": "N"}],
        "space_group_symbol_hermann_mauguin": "P6m2",
        "space_group_it_number": 187,
    },
}

SAMPLE_ENTRY_MOS2 = {
    "id": "mc2d_002",
    "type": "structures",
    "attributes": {
        "chemical_formula_reduced": "MoS2",
        "chemical_formula_descriptive": "Mo1 S2",
        "elements": ["Mo", "S"],
        "nelements": 2,
        "nsites": 3,
        "lattice_vectors": [[3.16, 0.0, 0.0], [-1.58, 2.736, 0.0], [0.0, 0.0, 20.0]],
        "cartesian_site_positions": [[0.0, 0.0, 0.0], [1.58, 0.912, 0.0], [1.58, 0.912, 3.0]],
        "species_at_sites": ["Mo", "S", "S"],
        "species": [{"name": "Mo"}, {"name": "S"}],
        "space_group_symbol_hermann_mauguin": "P-6m2",
        "space_group_it_number": 187,
    },
}

SAMPLE_ENTRY_LA = {
    "id": "mc2d_003",
    "type": "structures",
    "attributes": {
        "chemical_formula_reduced": "LaN",
        "chemical_formula_descriptive": "La1 N1",
        "elements": ["La", "N"],
        "nelements": 2,
        "nsites": 2,
        "lattice_vectors": [[2.5, 0.0, 0.0], [-1.25, 2.165, 0.0], [0.0, 0.0, 15.0]],
        "cartesian_site_positions": [[0.0, 0.0, 0.0], [1.25, 0.722, 0.0]],
        "species_at_sites": ["La", "N"],
        "species": [{"name": "La"}, {"name": "N"}],
        "space_group_symbol_hermann_mauguin": "P6m2",
        "space_group_it_number": 187,
    },
}


def _mock_response(data_list, next_url=None):
    """Create a mock requests.Response for the OPTIMADE API."""
    resp = MagicMock()
    body = {"data": data_list, "links": {}}
    if next_url:
        body["links"]["next"] = next_url
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


# ── optimade_entry_to_pymatgen ────────────────────────────────────────────────

def test_optimade_entry_to_pymatgen():
    struct = optimade_entry_to_pymatgen(SAMPLE_ENTRY)
    assert struct is not None
    assert struct.composition.reduced_formula == "BN"
    assert struct.num_sites == 2


# ── _get_structure_elements ───────────────────────────────────────────────────

def test_get_structure_elements_from_entry():
    item = {"entry": {"attributes": {"elements": ["B", "N"]}}}
    assert _get_structure_elements(item) == {"B", "N"}


def test_get_structure_elements_from_structure():
    from pymatgen.core import Lattice, Structure

    s = Structure(
        Lattice([[2.5, 0, 0], [0, 2.5, 0], [0, 0, 15]]),
        ["B", "N"],
        [[0, 0, 0], [1.25, 1.25, 0]],
    )
    item = {"structure": s}
    assert _get_structure_elements(item) == {"B", "N"}


def test_get_structure_elements_empty():
    assert _get_structure_elements({}) == set()


# ── _structure_has_unsupported_elements ───────────────────────────────────────

def test_no_exclude_or_supported():
    item = {"entry": {"attributes": {"elements": ["B", "N"]}}}
    assert _structure_has_unsupported_elements(item, None, None) is False


def test_excluded_elements_present():
    item = {"entry": {"attributes": {"elements": ["B", "N"]}}}
    assert _structure_has_unsupported_elements(item, {"N"}, None) is True


def test_excluded_elements_absent():
    item = {"entry": {"attributes": {"elements": ["B", "N"]}}}
    assert _structure_has_unsupported_elements(item, {"La"}, None) is False


def test_unsupported_elements():
    item = {"entry": {"attributes": {"elements": ["B", "N"]}}}
    assert _structure_has_unsupported_elements(item, None, {"B"}) is True


def test_all_supported():
    item = {"entry": {"attributes": {"elements": ["B", "N"]}}}
    assert _structure_has_unsupported_elements(item, None, {"B", "N"}) is False


# ── fetch_mc2d_structures ─────────────────────────────────────────────────────

@patch("aiida_gw.datasets.mc2d_optimade.requests.get")
def test_fetch_single_page(mock_get):
    mock_get.return_value = _mock_response([SAMPLE_ENTRY])
    results = fetch_mc2d_structures(max_structures=5)
    assert len(results) == 1
    assert results[0]["id"] == "mc2d_001"
    assert results[0]["formula"] == "BN"
    mock_get.assert_called_once_with(
        MC2D_STRUCTURES_URL, params=ANY, timeout=60
    )


@patch("aiida_gw.datasets.mc2d_optimade.requests.get")
def test_fetch_limit(mock_get):
    mock_get.return_value = _mock_response([SAMPLE_ENTRY, SAMPLE_ENTRY_MOS2])
    results = fetch_mc2d_structures(max_structures=1)
    assert len(results) == 1


@patch("aiida_gw.datasets.mc2d_optimade.requests.get")
def test_fetch_pagination(mock_get):
    next_url = f"{MC2D_STRUCTURES_URL}?page_offset=100"
    mock_get.side_effect = [
        _mock_response([SAMPLE_ENTRY], next_url=next_url),
        _mock_response([SAMPLE_ENTRY_MOS2]),
    ]
    results = fetch_mc2d_structures(max_structures=5)
    assert len(results) == 2
    assert results[0]["id"] == "mc2d_001"
    assert results[1]["id"] == "mc2d_002"


@patch("aiida_gw.datasets.mc2d_optimade.requests.get")
def test_fetch_retry_on_failure(mock_get):
    from requests import ConnectionError

    mock_get.side_effect = [
        ConnectionError("timeout"),
        ConnectionError("timeout"),
        _mock_response([SAMPLE_ENTRY]),
    ]
    results = fetch_mc2d_structures(max_structures=5)
    assert len(results) == 1
    assert mock_get.call_count == 3


@patch("aiida_gw.datasets.mc2d_optimade.requests.get")
def test_fetch_retry_exhausted(mock_get):
    from requests import ConnectionError

    mock_get.side_effect = ConnectionError("always fails")
    with pytest.raises(ConnectionError):
        fetch_mc2d_structures(max_structures=5)
    assert mock_get.call_count == 3


@patch("aiida_gw.datasets.mc2d_optimade.requests.get")
def test_fetch_with_filter(mock_get):
    mock_get.return_value = _mock_response([SAMPLE_ENTRY])
    results = fetch_mc2d_structures(
        optimade_filter='elements HAS ALL "B","N" AND nelements=2',
        max_structures=5,
    )
    assert len(results) == 1


@patch("aiida_gw.datasets.mc2d_optimade.requests.get")
def test_fetch_max_atoms_filter(mock_get):
    mock_get.return_value = _mock_response([SAMPLE_ENTRY, SAMPLE_ENTRY_MOS2])
    results = fetch_mc2d_structures(max_atoms=2, max_structures=5)
    assert len(results) == 1  # only BN (2 atoms) passes


@patch("aiida_gw.datasets.mc2d_optimade.requests.get")
def test_fetch_min_atoms_filter(mock_get):
    mock_get.return_value = _mock_response([SAMPLE_ENTRY, SAMPLE_ENTRY_MOS2])
    results = fetch_mc2d_structures(min_atoms=3, max_structures=5)
    assert len(results) == 1  # only MoS2 (3 atoms) passes


# ── fetch_all_mc2d ────────────────────────────────────────────────────────────

@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_fetch_all_basic(mock_fetch):
    mock_fetch.return_value = [
        {"id": "1", "formula": "BN", "entry": SAMPLE_ENTRY, "structure": MagicMock()},
    ]
    mock_fetch.return_value[0]["structure"].as_dict.return_value = {"lattice": {}, "sites": []}
    mock_fetch.return_value[0]["entry"]["attributes"]["nelements"] = 2
    mock_fetch.return_value[0]["entry"]["attributes"]["elements"] = ["B", "N"]
    mock_fetch.return_value[0]["nsites"] = 2

    results = fetch_all_mc2d()
    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert results[0]["nelements"] == 2
    assert results[0]["elements"] == ["B", "N"]
    assert "structure" in results[0]


@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_fetch_all_exclude_elements(mock_fetch):
    mock_fetch.return_value = [
        {"id": "1", "formula": "BN", "entry": SAMPLE_ENTRY, "structure": MagicMock(), "nsites": 2},
        {"id": "2", "formula": "LaN", "entry": SAMPLE_ENTRY_LA, "structure": MagicMock(), "nsites": 2},
    ]
    for r in mock_fetch.return_value:
        r["structure"].as_dict.return_value = {"lattice": {}, "sites": []}

    results = fetch_all_mc2d(exclude_elements={"La"})
    assert len(results) == 1
    assert results[0]["id"] == "1"


@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_fetch_all_supported_elements(mock_fetch):
    mock_fetch.return_value = [
        {"id": "1", "formula": "BN", "entry": SAMPLE_ENTRY, "structure": MagicMock(), "nsites": 2},
        {"id": "2", "formula": "MoS2", "entry": SAMPLE_ENTRY_MOS2, "structure": MagicMock(), "nsites": 3},
    ]
    for r in mock_fetch.return_value:
        r["structure"].as_dict.return_value = {"lattice": {}, "sites": []}

    results = fetch_all_mc2d(supported_elements={"B", "N"})
    assert len(results) == 1
    assert results[0]["id"] == "1"


# ── save_mc2d_by_nelements ────────────────────────────────────────────────────

@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_save_mc2d_by_nelements_basic(mock_fetch):
    from pymatgen.core import Lattice, Structure

    bn_struct = Structure(
        Lattice([[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 15]]),
        ["B", "N"],
        [[0, 0, 0], [1.25, 0.722, 0]],
    )

    mock_fetch.return_value = [
        {
            "id": "mc2d_001",
            "formula": "BN",
            "entry": {"attributes": {"nelements": 2, "elements": ["B", "N"]}},
            "structure": bn_struct,
            "nsites": 2,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_mc2d_by_nelements(tmpdir)

        assert 2 in paths
        filepath = Path(paths[2])
        assert filepath.exists()
        assert filepath.name == "mc2d_2elements.json"

        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["id"] == "mc2d_001"
        assert data[0]["formula"] == "BN"
        assert data[0]["nelements"] == 2
        assert data[0]["elements"] == ["B", "N"]
        assert "structure" in data[0]

        # Verify structure can be loaded back
        from pymatgen.core import Structure as PmStructure
        restored = PmStructure.from_dict(data[0]["structure"])
        assert restored.composition.reduced_formula == "BN"


@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_save_mc2d_by_nelements_grouped(mock_fetch):
    from pymatgen.core import Lattice, Structure

    bn_struct = Structure(
        Lattice([[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 15]]),
        ["B", "N"],
        [[0, 0, 0], [1.25, 0.722, 0]],
    )
    mos2_struct = Structure(
        Lattice([[3.16, 0, 0], [-1.58, 2.736, 0], [0, 0, 20]]),
        ["Mo", "S", "S"],
        [[0, 0, 0], [1.58, 0.912, 0], [1.58, 0.912, 3.0]],
    )

    mock_fetch.return_value = [
        {
            "id": "mc2d_001",
            "formula": "BN",
            "entry": {"attributes": {"nelements": 2, "elements": ["B", "N"]}},
            "structure": bn_struct,
            "nsites": 2,
        },
        {
            "id": "mc2d_002",
            "formula": "MoS2",
            "entry": {"attributes": {"nelements": 3, "elements": ["Mo", "S"]}},
            "structure": mos2_struct,
            "nsites": 3,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_mc2d_by_nelements(tmpdir)

        assert 2 in paths
        assert 3 in paths
        assert Path(paths[2]).name == "mc2d_2elements.json"
        assert Path(paths[3]).name == "mc2d_3elements.json"

        with open(paths[2]) as f:
            data2 = json.load(f)
        assert len(data2) == 1
        assert data2[0]["formula"] == "BN"

        with open(paths[3]) as f:
            data3 = json.load(f)
        assert len(data3) == 1
        assert data3[0]["formula"] == "MoS2"


@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_save_mc2d_skips_none_nelements(mock_fetch):
    from pymatgen.core import Lattice, Structure

    bn_struct = Structure(
        Lattice([[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 15]]),
        ["B", "N"],
        [[0, 0, 0], [1.25, 0.722, 0]],
    )

    mock_fetch.return_value = [
        {
            "id": "mc2d_001",
            "formula": "BN",
            "entry": {"attributes": {"nelements": None, "elements": ["B", "N"]}},
            "structure": bn_struct,
            "nsites": 2,
        },
        {
            "id": "mc2d_002",
            "formula": "BN",
            "entry": {"attributes": {"nelements": 2, "elements": ["B", "N"]}},
            "structure": bn_struct,
            "nsites": 2,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_mc2d_by_nelements(tmpdir)

        assert len(paths) == 1  # only nelements=2 group
        with open(paths[2]) as f:
            data = json.load(f)
        assert len(data) == 1
        # The None nelements entry was skipped
        assert all(d["nelements"] is not None for d in data)


@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_save_mc2d_exclude_elements(mock_fetch):
    from pymatgen.core import Lattice, Structure

    bn_struct = Structure(
        Lattice([[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 15]]),
        ["B", "N"],
        [[0, 0, 0], [1.25, 0.722, 0]],
    )
    la_struct = Structure(
        Lattice([[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 15]]),
        ["La", "N"],
        [[0, 0, 0], [1.25, 0.722, 0]],
    )

    mock_fetch.return_value = [
        {
            "id": "mc2d_001",
            "formula": "BN",
            "entry": {"attributes": {"nelements": 2, "elements": ["B", "N"]}},
            "structure": bn_struct,
            "nsites": 2,
        },
        {
            "id": "mc2d_002",
            "formula": "LaN",
            "entry": {"attributes": {"nelements": 2, "elements": ["La", "N"]}},
            "structure": la_struct,
            "nsites": 2,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_mc2d_by_nelements(tmpdir, exclude_elements={"La"})
        with open(paths[2]) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["id"] == "mc2d_001"


@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_save_mc2d_supported_elements(mock_fetch):
    from pymatgen.core import Lattice, Structure

    bn_struct = Structure(
        Lattice([[2.5, 0, 0], [-1.25, 2.165, 0], [0, 0, 15]]),
        ["B", "N"],
        [[0, 0, 0], [1.25, 0.722, 0]],
    )

    mock_fetch.return_value = [
        {
            "id": "mc2d_001",
            "formula": "BN",
            "entry": {"attributes": {"nelements": 2, "elements": ["B", "N"]}},
            "structure": bn_struct,
            "nsites": 2,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_mc2d_by_nelements(tmpdir, supported_elements={"B", "N"})
        with open(paths[2]) as f:
            data = json.load(f)
        assert len(data) == 1


@patch("aiida_gw.datasets.mc2d_optimade.fetch_mc2d_structures")
def test_save_mc2d_creates_output_dir(mock_fetch):
    mock_fetch.return_value = []

    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "subdir" / "nested"
        paths = save_mc2d_by_nelements(nested)
        assert nested.exists()
        assert paths == {}
