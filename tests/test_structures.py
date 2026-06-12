"""Tests for pure 2D structure transformation functions."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from aiida_gw.transformations.structures import (
    _center_slab,
    make_supercell,
    rotate_xy_to_xz,
)


@pytest.fixture
def bn_structure():
    """A simple hexagonal BN monolayer in XY-plane with Z vacuum."""
    a = 2.5
    lattice = Lattice([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 15]])
    return Structure(lattice, ["B", "N"], [[0, 0, 0], [a / 2, a * np.sqrt(3) / 6, 0]])


class TestCenterSlab:
    def test_centers_along_axis(self, bn_structure):
        centered = _center_slab(bn_structure, axis=1)
        # slab should be near center of cell (~0.5) after centering
        assert abs(np.mean(centered.frac_coords[:, 1]) - 0.5) < 0.1

    def test_does_not_break_structure(self, bn_structure):
        centered = _center_slab(bn_structure, axis=1)
        assert centered.num_sites == bn_structure.num_sites
        assert centered.species == bn_structure.species


class TestRotateXyToXz:
    def test_vacuum_becomes_y(self, bn_structure):
        rotated = rotate_xy_to_xz(bn_structure, vacuum=15)
        cell = rotated.lattice.matrix
        # Y dimension should be ~vacuum (vacuum), Z should be small (in-plane)
        assert cell[1, 1] == pytest.approx(15.0, abs=0.1)
        assert cell[2, 2] < 5.0

    def test_inplane_preserved(self, bn_structure):
        rotated = rotate_xy_to_xz(bn_structure, vacuum=15)
        cell = rotated.lattice.matrix
        # In-plane components X and Z should have non-zero magnitude
        assert abs(cell[0, 0]) > 0
        assert abs(cell[2, 0]) > 0 or abs(cell[0, 2]) > 0

    def test_slab_centered(self, bn_structure):
        rotated = rotate_xy_to_xz(bn_structure, vacuum=15)
        frac = rotated.frac_coords
        # Atoms should be near center of Y (vacuum direction)
        assert abs(np.mean(frac[:, 1]) - 0.5) < 0.1


class TestMakeSupercell:
    def test_supercell_size(self, bn_structure):
        sc = make_supercell(bn_structure, [3, 3, 1])
        assert sc.num_sites == bn_structure.num_sites * 9

    def test_supercell_lattice(self, bn_structure):
        sc = make_supercell(bn_structure, [2, 1, 1])
        assert sc.lattice.matrix[0, 0] == pytest.approx(bn_structure.lattice.matrix[0, 0] * 2)

    def test_identity(self, bn_structure):
        sc = make_supercell(bn_structure, [1, 1, 1])
        assert sc.num_sites == bn_structure.num_sites
