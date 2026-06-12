"""Read and resolve CP2K data files (basis sets, potentials, RI basis sets)."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

_ACCURACY_RE = re.compile(
    r"(?:relative\s+accuracy\s+of\s+RI-MP2|error)[:\s_]+([\d.eE+-]+)"
)
_HEADER_RE = re.compile(r"^([A-Z][a-z]?)\s+(.+)")


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


def _parse_lines(lines: list[str]) -> dict[str, list[BasisEntry]]:
    """Parse CP2K data file lines into ``element → [BasisEntry, ...]``."""
    entries: dict[str, list[BasisEntry]] = {}
    current_element: str | None = None
    current_name: str | None = None
    current_header: str | None = None
    prev_comment: list[str] = []
    next_comment: list[str] = []

    def _save() -> None:
        if current_element is not None and current_name is not None:
            comment = " ".join(prev_comment).strip()
            entries.setdefault(current_element, []).append(
                BasisEntry(
                    name=current_name,
                    header=current_header or "",
                    comment=comment,
                )
            )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            comment_text = stripped.lstrip("#").strip()
            if comment_text:
                next_comment.append(comment_text)
            continue

        m = _HEADER_RE.match(stripped)
        if m:
            _save()
            prev_comment = next_comment
            next_comment = []
            current_element = m.group(1)
            current_name = m.group(2)
            current_header = stripped

    _save()
    return entries


@functools.lru_cache(maxsize=128)
def _parse_cached(file: str | Path) -> dict[str, list[BasisEntry]]:
    """Cached version of parse_cp2k_data_file for file paths."""
    with open(file) as fh:
        return _parse_lines(fh.readlines())


def parse_cp2k_data_file(file: str | Path | IO) -> dict[str, list[BasisEntry]]:
    """Parse a CP2K data file into ``element → [BasisEntry, ...]``.

    Results for file paths (str/Path) are cached. IO objects are not cached.

    Format::

        # comment with accuracy
        Element  BasisName
          <data lines>
    """
    if isinstance(file, IO):
        return _parse_lines(file.readlines())
    return _parse_cached(file)


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

    When *orb_basis* is given (e.g. ``"aug-SZV-MOLOPT-GTH-tier-1"``),
    only entries whose name contains ``RI_{orb_basis}`` are considered,
    ensuring consistency with the ORB basis set.

    When *accuracy_target* is given (e.g. ``1e-5``), picks the entry
    whose accuracy is closest to *target*.  When ``None``, picks the
    entry with the smallest error (most accurate).
    """
    entries = list_basis_entries(ri_basis_file, element)

    if orb_basis:
        prefix = f"RI_{orb_basis}"
        entries = [e for e in entries if prefix in e.name]

    if not entries:
        return None
    if accuracy_target is not None:
        return _select_basis_by_accuracy(entries, accuracy_target)
    candidates = [e for e in entries if e.accuracy is not None]
    if candidates:
        return min(candidates, key=lambda e: e.accuracy).name
    return entries[0].name


def _select_basis_by_accuracy(
    entries: list[BasisEntry], target: float
) -> str | None:
    """Pick the entry whose accuracy is closest to *target*."""
    if not entries:
        return None
    candidates = [e for e in entries if e.accuracy is not None]
    if not candidates:
        return entries[0].name
    return min(candidates, key=lambda e: abs(e.accuracy - target)).name


def _first_token(name: str) -> str:
    """Return the first token of *name* (CP2K ignores aliases after the first)."""
    return name.split()[0]


def resolve_orbital_basis_name(
    basis_file: str | Path,
    element: str,
    orb_basis: str | None = None,
) -> str | None:
    """Return the orbital basis set name for *element* from a CP2K basis file.

    When *orb_basis* is given, picks the first entry whose name contains it
    (e.g. ``"aug-SZV-MOLOPT-GTH-tier-1"``). Otherwise returns the first entry.
    """
    entries = list_basis_entries(basis_file, element)
    if orb_basis:
        entries = [e for e in entries if orb_basis in e.name]
    return _first_token(entries[0].name) if entries else None


_Q_RE = re.compile(r"-q(\d+)")


def _q_number(name: str) -> int:
    """Extract the number of valence electrons from a GTH potential *name*."""
    m = _Q_RE.search(name)
    return int(m.group(1)) if m else 0


def resolve_potential_name(
    potential_file: str | Path,
    element: str,
    pattern: str | None = "GTH-",
) -> str | None:
    """Return the potential name for *element* from a CP2K POTENTIAL file.

    When *pattern* is given (default ``"GTH-"``), only names containing
    it are considered — this avoids picking an all-electron entry when
    a pseudopotential is wanted.

    When multiple entries match (e.g. ``GTH-PBE-q1`` and ``GTH-PBE-q9``
    for Na), the one with the **largest** q-number (most valence
    electrons) is selected, which is generally preferred for GW.
    """
    entries = list_basis_entries(potential_file, element)
    names = [_first_token(e.name) for e in entries]
    if pattern:
        names = [n for n in names if pattern in n]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    return max(names, key=_q_number)
