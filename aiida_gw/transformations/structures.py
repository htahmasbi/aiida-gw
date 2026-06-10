from __future__ import annotations

import logging

import numpy as np
from pymatgen.core import Lattice, Structure

logger = logging.getLogger(__name__)


def _center_slab(structure: Structure, axis: int = 1) -> Structure:
    """Center slab in the cell along a given axis using circular statistics."""
    result = structure.copy()
    frac = np.array(result.frac_coords)[:, axis]
    mean_angle = np.arctan2(
        np.mean(np.sin(2 * np.pi * frac)),
        np.mean(np.cos(2 * np.pi * frac)),
    )
    mean = mean_angle / (2 * np.pi)
    shift = 0.5 - mean
    vec = [0, 0, 0]
    vec[axis] = shift
    result.translate_sites(list(range(len(result))), vec)
    return result


def rotate_xy_to_xz(structure: Structure, vacuum: float = 20.0) -> Structure:
    """Rotate from XY plane (Z vacuum) to XZ plane (Y vacuum).

    Mapping: x_new = x_old, y_new = z_old, z_new = y_old.
    Atoms are centered in the new vacuum gap.
    """
    old_lattice = structure.lattice.matrix

    a_old = old_lattice[0]
    b_old = old_lattice[1]

    a_new = [a_old[0], 0.0, a_old[1]]
    b_new = [0.0, vacuum, 0.0]
    c_new = [b_old[0], 0.0, b_old[1]]

    new_lattice = Lattice([a_new, b_new, c_new])

    species = []
    coords = []
    for site in structure:
        x, y, z = site.coords
        species.append(site.species)
        coords.append([x, z, y])

    result = Structure(
        lattice=new_lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=True,
    )

    result = _center_slab(result, axis=1)
    return result


def make_supercell(structure: Structure, scaling: list[int]) -> Structure:
    """Return a supercell scaled by *scaling* (e.g. ``[3, 3, 1]``)."""
    s = structure.copy()
    s.make_supercell(scaling)
    return s


def prepare_2d_structure(
    structure: Structure,
    vacuum: float = 20.0,
    supercell: list[int] | None = None,
) -> Structure:
    """Prepare a 2D structure for GW: supercell, rotate to XZ, add Y vacuum."""
    if supercell:
        structure = make_supercell(structure, supercell)
    structure = rotate_xy_to_xz(structure, vacuum=vacuum)
    return structure
