"""Tests for pure builder functions."""

import numpy as np
import pytest

from aiida_gw.core.builders import (
    _classify_from_vectors,
    _is_cp2k_key,
    _strip_invalid_keys,
    dict_merge,
    get_cube_print_section,
)


class TestIsCp2kKey:
    def test_uppercase(self):
        assert _is_cp2k_key("FORCE_EVAL") is True
        assert _is_cp2k_key("BASIS_SET ORB") is True

    def test_underscore(self):
        assert _is_cp2k_key("_") is True

    def test_hash_prefix(self):
        assert _is_cp2k_key("### JOB_TYPE") is True

    def test_lowercase(self):
        assert _is_cp2k_key("basis_set") is False

    def test_mixed(self):
        assert _is_cp2k_key("Scf") is False

    def test_empty(self):
        assert _is_cp2k_key("") is False


class TestStripInvalidKeys:
    def test_flat_dict(self):
        d = {"FORCE_EVAL": 1, "scf": 2, "_": 3}
        _strip_invalid_keys(d)
        assert d == {"FORCE_EVAL": 1, "_": 3}

    def test_nested(self):
        d = {"FORCE_EVAL": {"DFT": {"scf": 1, "CUTOFF": 2}, "bad": 3}}
        _strip_invalid_keys(d)
        assert d == {"FORCE_EVAL": {"DFT": {"CUTOFF": 2}}}

    def test_empty_result(self):
        d = {"foo": 1, "bar": {"baz": 2}}
        _strip_invalid_keys(d)
        assert d == {}

    def test_all_valid(self):
        d = {"FORCE_EVAL": {"DFT": {"CUTOFF": 400}}, "### META": "x"}
        _strip_invalid_keys(d)
        assert d == {"FORCE_EVAL": {"DFT": {"CUTOFF": 400}}, "### META": "x"}


class TestDictMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        merge = {"b": 3, "c": 4}
        dict_merge(base, merge)
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"FORCE_EVAL": {"DFT": {"CUTOFF": 400}, "METHOD": "QS"}}
        merge = {"FORCE_EVAL": {"DFT": {"REL_CUTOFF": 50}}}
        dict_merge(base, merge)
        assert base == {"FORCE_EVAL": {"DFT": {"CUTOFF": 400, "REL_CUTOFF": 50}, "METHOD": "QS"}}

    def test_overwrite_nested(self):
        base = {"a": {"x": 1, "y": 2}}
        merge = {"a": {"y": 99}}
        dict_merge(base, merge)
        assert base == {"a": {"x": 1, "y": 99}}

    def test_merge_into_empty(self):
        base = {}
        merge = {"a": 1, "b": {"c": 2}}
        dict_merge(base, merge)
        assert base == {"a": 1, "b": {"c": 2}}

    def test_no_mutation_of_merge(self):
        base = {"a": 1}
        merge = {"a": 2}
        dict_merge(base, merge)
        assert merge == {"a": 2}  # unchanged


class TestGetCubePrintSection:
    def test_structure(self):
        section = get_cube_print_section()
        assert "PRINT" in section
        dft_print = section["PRINT"]
        assert "V_HARTREE_CUBE" in dft_print
        assert "E_DENSITY_CUBE" in dft_print

    def test_sections_enabled(self):
        section = get_cube_print_section()
        for name in ("V_HARTREE_CUBE", "E_DENSITY_CUBE"):
            assert section["PRINT"][name]["_"] == "ON"
            assert section["PRINT"][name]["STRIDE"] == "1 1 1"

    def test_keys_survive_strip(self):
        section = get_cube_print_section()
        _strip_invalid_keys(section)
        assert section == get_cube_print_section()

    def test_mergeable_into_dft(self):
        dft = {"CUTOFF": 400}
        dict_merge(dft, get_cube_print_section())
        assert dft["PRINT"]["V_HARTREE_CUBE"]["_"] == "ON"


class TestClassifyFromVectors:
    def make_vec(self, x, y, z=0.0):
        return np.array([x, y, z], dtype=float)

    def test_square(self):
        a = self.make_vec(3.0, 0.0)
        b = self.make_vec(0.0, 3.0)
        assert _classify_from_vectors(a, b) == "square"

    def test_rectangular(self):
        a = self.make_vec(4.0, 0.0)
        b = self.make_vec(0.0, 2.0)
        assert _classify_from_vectors(a, b) == "rectangular"

    def test_hexagonal(self):
        a = self.make_vec(3.0, 0.0)
        b = self.make_vec(1.5, 2.598076211, 0.0)  # 60 deg, same length
        assert _classify_from_vectors(a, b) == "hexagonal"

    def test_oblique(self):
        a = self.make_vec(3.0, 0.0)
        b = self.make_vec(1.0, 2.0, 0.0)  # arbitrary angle
        assert _classify_from_vectors(a, b) == "oblique"
