"""Tests for CP2K data file reader and basis/potential resolution."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from aiida_gw.codes.cp2k.data_reader import (
    _extract_accuracy,
    _first_token,
    parse_cp2k_data_file,
    resolve_orbital_basis_name,
    resolve_potential_name,
    resolve_ri_basis_name,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_data_file(lines: list[str], suffix: str = ".data") -> str:
    """Write a temporary CP2K data file and return its path."""
    with NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.writelines(lines)
        return f.name


def _make_ri_basis_file(entries: list[tuple[str, str]]) -> str:
    """Write a temporary RI basis file from (element, name) pairs."""
    lines = []
    for element, name in entries:
        lines.append(f"{element}  {name}\n")
        lines.append("  1 0 0.0 0.0 0.0\n")
    return _make_data_file(lines, ".RI")


def _make_potential_file(entries: list[tuple[str, str]]) -> str:
    """Write a temporary potential file from (element, name) pairs."""
    lines = []
    for element, name in entries:
        lines.append(f"{element}  {name}\n")
        lines.append("  0.5 0.5 0.5 1.0 1.0\n")
    return _make_data_file(lines, ".POT")


def _make_basis_file(entries: list[tuple[str, str]]) -> str:
    """Write a temporary basis file from (element, name) pairs."""
    lines = []
    for element, name in entries:
        lines.append(f"{element}  {name}\n")
        lines.append("  2 1 0.5 0.5\n")
    return _make_data_file(lines, ".BASIS")


# ── _extract_accuracy ────────────────────────────────────────────────────────

def test_extract_accuracy_error_underscore():
    assert _extract_accuracy("error_4.3e-08") == 4.3e-08


def test_extract_accuracy_error_space():
    assert _extract_accuracy("error 1.2e-05") == 1.2e-05


def test_extract_accuracy_error_colon():
    assert _extract_accuracy("error: 5.0e-06") == 5.0e-06


def test_extract_accuracy_relative_accuracy():
    assert _extract_accuracy("relative accuracy of RI-MP2 1e-5") == 1e-5


def test_extract_accuracy_relative_accuracy_colon():
    assert _extract_accuracy("relative accuracy of RI-MP2: 8.5e-05") == 8.5e-05


def test_extract_accuracy_none():
    assert _extract_accuracy("") is None
    assert _extract_accuracy("some random text") is None
    assert _extract_accuracy(None) is None  # type: ignore[arg-type]


# ── _first_token ─────────────────────────────────────────────────────────────

def test_first_token_single():
    assert _first_token("GTH-PBE-q3") == "GTH-PBE-q3"


def test_first_token_multi():
    assert _first_token("GTH-PBE-q3 GTH-PBE") == "GTH-PBE-q3"


def test_first_token_ri():
    assert (
        _first_token("RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08")
        == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"
    )


# ── parse_cp2k_data_file ─────────────────────────────────────────────────────

SAMPLE_DATA = [
    "# comment line\n",
    "# accuracy error 1e-5\n",
    "B  GTH-PBE-q3 GTH-PBE\n",
    "  0.5 0.3 0.2\n",
    "  0.1 0.2 0.3\n",
    "# another entry\n",
    "N  GTH-PBE-q5 GTH-PBE\n",
    "  0.6 0.4 0.2\n",
]

SAMPLE_DATA_WITH_ACCURACY = [
    "# relative accuracy of RI-MP2: 1.2e-05\n",
    "B  RI-aug-SZV-MOLOPT-GTH-tier-1\n",
    "  1 0 0.0 0.0 0.0\n",
    "# relative accuracy of RI-MP2: 4.3e-08\n",
    "B  RI-aug-SZV-MOLOPT-GTH-tier-1_high\n",
    "  1 0 0.0 0.0 0.0\n",
]


def test_parse_cp2k_data_file_elements():
    path = _make_data_file(SAMPLE_DATA)
    try:
        data = parse_cp2k_data_file(path)
        assert set(data.keys()) == {"B", "N"}
    finally:
        Path(path).unlink()


def test_parse_cp2k_data_file_entry_names():
    path = _make_data_file(SAMPLE_DATA)
    try:
        data = parse_cp2k_data_file(path)
        assert data["B"][0].name == "GTH-PBE-q3 GTH-PBE"
        assert data["N"][0].name == "GTH-PBE-q5 GTH-PBE"
    finally:
        Path(path).unlink()


def test_parse_cp2k_data_file_comment():
    path = _make_data_file(SAMPLE_DATA)
    try:
        data = parse_cp2k_data_file(path)
        assert "accuracy error 1e-5" in data["B"][0].comment
    finally:
        Path(path).unlink()


def test_parse_cp2k_data_file_accuracy_from_comment():
    path = _make_data_file(SAMPLE_DATA_WITH_ACCURACY)
    try:
        data = parse_cp2k_data_file(path)
        assert data["B"][0].accuracy == 1.2e-05
        assert data["B"][1].accuracy == 4.3e-08
    finally:
        Path(path).unlink()


def test_parse_cp2k_data_file_empty():
    path = _make_data_file([], ".data")
    try:
        data = parse_cp2k_data_file(path)
        assert data == {}
    finally:
        Path(path).unlink()


def test_parse_cp2k_data_file_comments_only():
    path = _make_data_file(["# just a comment\n", "# another\n"], ".data")
    try:
        data = parse_cp2k_data_file(path)
        assert data == {}
    finally:
        Path(path).unlink()


# ── resolve_potential_name ───────────────────────────────────────────────────

POTENTIAL_ENTRIES = [
    ("B", "GTH-PBE-q3 GTH-PBE"),
    ("B", "GTH-BLYP-q3 GTH-BLYP"),
    ("N", "GTH-PBE-q5 GTH-PBE"),
    ("Na", "GTH-PBE-q1 GTH-PBE"),
    ("Na", "GTH-PBE-q9 GTH-PBE"),
    ("Au", "GTH-PBE-q11 GTH-PBE"),
    ("Au", "All-Electron-Au"),  # not GTH-
]


def test_resolve_potential_single_entry():
    """Single match returns the first token (no alias)."""
    path = _make_potential_file([("He", "GTH-PBE-q2 GTH-PBE")])
    try:
        result = resolve_potential_name(path, "He")
        assert result == "GTH-PBE-q2"
    finally:
        Path(path).unlink()


def test_resolve_potential_picks_first_token():
    """Alias after the primary name should be stripped."""
    path = _make_potential_file(POTENTIAL_ENTRIES)
    try:
        result = resolve_potential_name(path, "B")
        assert result == "GTH-PBE-q3"
    finally:
        Path(path).unlink()


def test_resolve_potential_largest_q():
    """For Na with q1 and q9, picks q9 (most valence electrons)."""
    path = _make_potential_file(POTENTIAL_ENTRIES)
    try:
        result = resolve_potential_name(path, "Na")
        assert result == "GTH-PBE-q9"
    finally:
        Path(path).unlink()


def test_resolve_potential_filters_non_gth():
    """All-Electron entries (no GTH-) should be excluded."""
    path = _make_potential_file(POTENTIAL_ENTRIES)
    try:
        result = resolve_potential_name(path, "Au")
        assert result == "GTH-PBE-q11"
    finally:
        Path(path).unlink()


def test_resolve_potential_not_found():
    path = _make_potential_file([("B", "GTH-PBE-q3")])
    try:
        result = resolve_potential_name(path, "H")
        assert result is None
    finally:
        Path(path).unlink()


# ── resolve_orbital_basis_name ───────────────────────────────────────────────

BASIS_ENTRIES = [
    ("B", "DZVP-MOLOPT-PBE-GTH-q3 DZVP-MOLOPT-GGA-GTH-q3"),
    ("B", "TZV2P-MOLOPT-PBE-GTH-q3 TZV2P-MOLOPT-GGA-GTH-q3"),
    ("N", "DZVP-MOLOPT-PBE-GTH-q5 DZVP-MOLOPT-GGA-GTH-q5"),
]


def test_resolve_orbital_no_filter():
    """Without orb_basis filter, returns first entry's first token."""
    path = _make_basis_file(BASIS_ENTRIES)
    try:
        result = resolve_orbital_basis_name(path, "B")
        assert result == "DZVP-MOLOPT-PBE-GTH-q3"
    finally:
        Path(path).unlink()


def test_resolve_orbital_with_filter():
    """With orb_basis filter, returns first matching entry's first token."""
    path = _make_basis_file(BASIS_ENTRIES)
    try:
        result = resolve_orbital_basis_name(path, "B", orb_basis="TZV2P")
        assert result == "TZV2P-MOLOPT-PBE-GTH-q3"
    finally:
        Path(path).unlink()


def test_resolve_orbital_not_found():
    path = _make_basis_file(BASIS_ENTRIES)
    try:
        result = resolve_orbital_basis_name(path, "H")
        assert result is None
    finally:
        Path(path).unlink()


# ── resolve_ri_basis_name (core scenarios) ───────────────────────────────────

SAMPLE_RI_ENTRIES = [
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_2.1e-07"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.0e-06"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_5.0e-06"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.2e-05"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_8.5e-05"),
    ("N", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.5e-05"),
]


def test_resolve_ri_target_none_picks_most_accurate():
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=None, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"
    finally:
        Path(path).unlink()


def test_resolve_ri_target_1e5():
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-5, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.2e-05"
    finally:
        Path(path).unlink()


def test_resolve_ri_target_1e6():
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-6, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.0e-06"
    finally:
        Path(path).unlink()


def test_resolve_ri_target_1e7():
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-7, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"
    finally:
        Path(path).unlink()


def test_resolve_ri_target_1e8():
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-8, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"
    finally:
        Path(path).unlink()


def test_resolve_ri_no_orb_basis():
    """Without orb_basis filter, all entries are considered."""
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(path, "B")
        assert result is not None
    finally:
        Path(path).unlink()


def test_resolve_ri_no_match():
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(path, "H")
        assert result is None
    finally:
        Path(path).unlink()


def test_resolve_ri_different_elements():
    """Each element gets its own resolution."""
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        n_result = resolve_ri_basis_name(
            path, "N", accuracy_target=1e-5, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert n_result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.5e-05"
    finally:
        Path(path).unlink()


# ── edge cases ───────────────────────────────────────────────────────────────

def test_resolve_potential_no_gth_pattern():
    """When no GTH- entries exist, returns first entry's first token."""
    entries = [("B", "All-Electron-B")]
    path = _make_potential_file(entries)
    try:
        result = resolve_potential_name(path, "B")
        assert result is None  # filtered out by GTH- pattern
    finally:
        Path(path).unlink()


def test_resolve_ri_no_accuracy_metadata():
    """Entries without accuracy metadata fall back to first entry."""
    entries = [("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_basic")]
    path = _make_ri_basis_file(entries)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-5, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_basic"
    finally:
        Path(path).unlink()
