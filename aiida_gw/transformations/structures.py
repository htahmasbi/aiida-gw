from __future__ import annotations

import logging

import numpy as np
from pymatgen.core import Lattice, Structure

logger = logging.getLogger(__name__)


def rotate_2d_to_xy(structure: Structure) -> Structure:
    """Rotate a 2D material structure so the normal is along Y (vacuum direction)."""
    lattice = structure.lattice
    matrix = lattice.matrix.copy()

    norms = np.linalg.norm(matrix, axis=1)
    max_idx = np.argmin(norms)

    if max_idx == 0:
        new_lattice = Lattice([matrix[2], matrix[1], matrix[0]])
    elif max_idx == 1:
        new_lattice = Lattice([matrix[0], matrix[2], matrix[1]])
    else:
        new_lattice = lattice

    frac_coords = structure.frac_coords
    species = structure.species

    rotated = Structure(new_lattice, species, frac_coords)
    return rotated


def add_vacuum(structure: Structure, vacuum: float = 20.0, axis: int = 1) -> Structure:
    """Add vacuum along a given axis (default: Y = axis 1)."""
    matrix = structure.lattice.matrix.copy()
    old_length = np.linalg.norm(matrix[axis])
    scale = (old_length + vacuum) / old_length
    matrix[axis] *= scale
    new_lattice = Lattice(matrix)
    return Structure(new_lattice, structure.species, structure.frac_coords)


def make_supercell(structure: Structure, scaling: list[int]) -> Structure:
    """Make a supercell from a structure."""
    return structure.copy() * scaling


def prepare_2d_structure(
    structure: Structure,
    vacuum: float = 20.0,
    supercell: list[int] | None = None,
) -> Structure:
    """Prepare a 2D structure for GW: rotate, supercell, add vacuum."""
    s = rotate_2d_to_xy(structure)
    if supercell:
        s = make_supercell(s, supercell)
    s = add_vacuum(s, vacuum=vacuum, axis=1)
    return s
