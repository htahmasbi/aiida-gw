"""Tests for CP2K data file reader and basis/potential resolution."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from aiida_gw.codes.cp2k.data_reader import (
    resolve_ri_basis_name,
)


def _make_ri_basis_file(entries: list[tuple[str, str]]) -> str:
    """Write a temporary RI basis file from (element, name) pairs and return its path."""
    lines = []
    for element, name in entries:
        lines.append(f"{element}  {name}\n")
        lines.append("  1 0 0.0 0.0 0.0\n")
    with NamedTemporaryFile(mode="w", suffix=".RI", delete=False) as f:
        f.writelines(lines)
        return f.name


SAMPLE_RI_ENTRIES = [
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_2.1e-07"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.0e-06"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_5.0e-06"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.2e-05"),
    ("B", "RI_aug-SZV-MOLOPT-GTH-tier-1_error_8.5e-05"),
]


def test_resolve_ri_target_none_picks_most_accurate():
    """accuracy_target=None should pick the smallest error (most accurate)."""
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=None, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"
    finally:
        Path(path).unlink()


def test_resolve_ri_target_1e5():
    """accuracy_target=1e-5 → closest to 0.00001 → entry with error 1.2e-05."""
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-5, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.2e-05"
    finally:
        Path(path).unlink()


def test_resolve_ri_target_1e6():
    """accuracy_target=1e-6 → closest to 0.000001 → entry with error 1.0e-06."""
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-6, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_1.0e-06"
    finally:
        Path(path).unlink()


def test_resolve_ri_target_1e7():
    """accuracy_target=1e-7 → closest to 0.0000001 → entry with error 4.3e-08 (diff 5.7e-08 < 1.1e-07)."""
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-7, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"
    finally:
        Path(path).unlink()


def test_resolve_ri_target_1e8():
    """accuracy_target=1e-8 → closest to 1e-08 → entry with error 4.3e-08."""
    path = _make_ri_basis_file(SAMPLE_RI_ENTRIES)
    try:
        result = resolve_ri_basis_name(
            path, "B", accuracy_target=1e-8, orb_basis="aug-SZV-MOLOPT-GTH-tier-1"
        )
        assert result == "RI_aug-SZV-MOLOPT-GTH-tier-1_error_4.3e-08"
    finally:
        Path(path).unlink()
