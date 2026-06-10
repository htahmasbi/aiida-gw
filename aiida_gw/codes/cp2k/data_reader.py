"""Read and resolve CP2K data files (basis sets, potentials, RI basis sets)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

_ACCURACY_RE = re.compile(
    r"(?:relative\s+accuracy\s+of\s+RI-MP2|error)[:\s_]+([\d.eE+-]+)"
)


def _extract_accuracy(text: str) -> float | None:
    """Extract a relative accuracy value from *text*."""
    if not text:
        return None
    m = _ACCURACY_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


@dataclass
class BasisEntry:
    """A single entry in a CP2K data file."""
    name: str
    header: str
    comment: str = ""
    accuracy: float | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.accuracy is None:
            self.accuracy = _extract_accuracy(self.comment) or _extract_accuracy(self.name)


def parse_cp2k_data_file(file: str | Path | IO) -> dict[str, list[BasisEntry]]:
    """Parse a CP2K basis set / potential / RI data file.

    Returns a mapping ``element → [BasisEntry, ...]`` in file order.
    """
    if isinstance(file, (str, Path)):
        with open(file) as fh:
            lines = fh.readlines()
    else:
        lines = file.readlines()

    entries: dict[str, list[BasisEntry]] = {}
    current_element: str | None = None
    current_header: str | None = None
    current_comment: str = ""
    pending_name: str | None = None
    pending_numbers: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Comment line — may contain accuracy metadata
        if stripped.startswith("#"):
            if current_element is None:
                # Could be "Element: N" or "Basis set for N"
                el_match = re.search(
                    r"(?:Element|element|atom|Atom)\s*(?::|#)\s*([A-Z][a-z]?)",
                    stripped,
                )
                if el_match:
                    current_element = el_match.group(1)
            current_comment = line
            continue

        if not stripped:
            continue

        # First non-comment line after an element: should be the basis name
        if pending_name is None:
            pending_name = stripped
            pending_numbers = []
            continue

        # Lines with numbers (shell data)
        if re.match(r"^\s*\d+", stripped):
            pending_numbers.append(stripped)
            continue

        # If we got here without matching, we may have missed a name.
        # Flush pending entry first, then treat as new name.
        if pending_name is not None:
            _flush_entry(entries, current_element, pending_name, current_header, current_comment)
            current_comment = ""
            pending_name = None
            pending_numbers = []

        pending_name = stripped
        pending_numbers = []

    # Flush last entry
    if pending_name is not None:
        _flush_entry(entries, current_element, pending_name, current_header, current_comment)

    return entries


def _flush_entry(
    entries: dict[str, list[BasisEntry]],
    element: str | None,
    name: str,
    header: str | None,
    comment: str,
) -> None:
    """Add a basis entry to the dictionary."""
    if element is not None:
        entries.setdefault(element, []).append(
            BasisEntry(name=name, header=header or "", comment=comment)
        )


def list_basis_entries(
    file: str | Path,
    element: str,
) -> list[BasisEntry]:
    """Return all basis entries for *element* in *file*."""
    data = parse_cp2k_data_file(file)
    return data.get(element, [])


def resolve_ri_basis_name(
    ri_basis_file: str | Path,
    element: str,
    accuracy_target: float | None = None,
    orb_basis: str | None = None,
) -> str | None:
    """Return the RI auxiliary basis name for *element*.

    When *accuracy_target* is ``None``, picks the entry with the **largest**
    accuracy value (i.e., the cheapest / largest-error basis). When set,
    picks the cheapest basis whose accuracy still meets the target.
    """
    entries = list_basis_entries(ri_basis_file, element)

    if orb_basis:
        prefix = f"RI_{orb_basis}"
        entries = [e for e in entries if prefix in e.name]

    if not entries:
        return None

    # Pick the entry with the largest accuracy (cheapest, largest error)
    candidates = [e for e in entries if e.accuracy is not None]
    if candidates:
        if accuracy_target is not None:
            within = [e for e in candidates if e.accuracy >= accuracy_target]
            if within:
                return max(within, key=lambda e: e.accuracy).name
            return min(candidates, key=lambda e: e.accuracy).name
        # No target → pick the cheapest (largest error)
        return max(candidates, key=lambda e: e.accuracy).name

    return entries[0].name


def resolve_potential_name(
    potential_file: str | Path,
    element: str,
) -> str | None:
    """Return the potential name for *element* from a CP2K POTENTIAL file."""
    entries = list_basis_entries(potential_file, element)
    return entries[0].name if entries else None
