"""Tests for pure builder functions and Cp2kBuilder."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aiida_gw.core.builders import (
    Cp2kBuilder,
    _classify_from_vectors,
    _is_cp2k_key,
    _strip_invalid_keys,
    dict_merge,
    get_cube_print_section,
)


class TestMetadataOptionsValidators:
    """Tests for MetadataOptions pydantic field validators."""

    def test_partition_valid(self):
        from aiida_gw.core.config import MetadataOptions
        for name in ("cpu-genoa", "gpu-a100", "debug", "compute", "cpu_gen6", "a_b"):
            opts = MetadataOptions(partition=name)
            assert opts.partition == name

    def test_partition_invalid_characters(self):
        from aiida_gw.core.config import MetadataOptions
        from pydantic import ValidationError

        for name in ("cpu genoa", "partition!", "queue@cluster", "part/queue"):
            with pytest.raises(ValidationError, match="Invalid partition name"):
                MetadataOptions(partition=name)

    def test_memory_format_valid(self):
        from aiida_gw.core.config import MetadataOptions
        for val in ("600G", "38400M", "128K", "1G", "512M", "600", "0", "38400"):
            opts = MetadataOptions(memory_per_machine=val)
            assert opts.memory_per_machine == val

    def test_memory_format_invalid_unit(self):
        from aiida_gw.core.config import MetadataOptions
        from pydantic import ValidationError

        for val in ("600X", "10GB", "8T", "12.5G"):
            with pytest.raises(ValidationError, match="must end with G/M/K|must be a number"):
                MetadataOptions(memory_per_machine=val)

    def test_memory_format_invalid_number(self):
        from aiida_gw.core.config import MetadataOptions
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must be a number"):
            MetadataOptions(memory_per_machine="abcG")

    def test_partition_and_memory_none(self):
        from aiida_gw.core.config import MetadataOptions

        opts = MetadataOptions()
        assert opts.partition is None
        assert opts.memory_per_machine is None


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


class TestCp2kBuilder:
    """Integration tests for Cp2kBuilder.build_scf_inputs and build_gw_inputs."""

    @pytest.fixture
    def mock_structure(self):
        """Minimal H2 molecule structure."""
        from ase import Atoms

        mol = Atoms("HH", positions=[(0, 0, 0), (0.74, 0, 0)], cell=[10, 10, 10], pbc=[True, True, True])
        with patch("aiida_gw.core.builders.StructureData") as MockSD:
            mock = MagicMock()
            mock.get_ase.return_value = mol
            mock.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
            MockSD.return_value = mock
            yield mock

    @pytest.fixture
    def mock_code(self):
        return MagicMock(label="cp2k@localhost")

    @pytest.fixture
    def mock_config(self):
        from aiida_gw.core.config import Cp2kConfig, GwConfig, MetadataOptions, ProjectConfig

        metadata = MetadataOptions()
        cp2k_cfg = Cp2kConfig()
        gw_cfg = GwConfig(
            resolve_from_files=False,
            xc_functional="PBE",
        )
        return ProjectConfig(metadata_options=metadata, cp2k=cp2k_cfg, gw=gw_cfg)

    def test_build_scf_inputs_returns_builder(self, mock_structure, mock_code, mock_config):
        builder = Cp2kBuilder(mock_config).build_scf_inputs(
            structure=mock_structure,
            code=mock_code,
            protocol_section="single_point",
            protocol_name="protocol_SIRIUS.yml",
            kpoints_mesh=[2, 2, 2],
        )
        assert builder is not None
        assert hasattr(builder, "cp2k")
        assert builder.cp2k.code is mock_code
        assert builder.cp2k.structure is mock_structure

    def test_build_scf_inputs_sets_parameters(self, mock_structure, mock_code, mock_config):
        builder = Cp2kBuilder(mock_config).build_scf_inputs(
            structure=mock_structure,
            code=mock_code,
            protocol_section="single_point",
            protocol_name="protocol_SIRIUS.yml",
        )
        params = builder.cp2k.parameters.get_dict()
        assert "FORCE_EVAL" in params
        assert "GLOBAL" in params

    def test_build_scf_inputs_sets_retrieve_list(self, mock_structure, mock_code, mock_config):
        builder = Cp2kBuilder(mock_config).build_scf_inputs(
            structure=mock_structure,
            code=mock_code,
            protocol_section="single_point",
            protocol_name="protocol_SIRIUS.yml",
        )
        settings = builder.cp2k.settings.get_dict()
        retrieve_list = settings["additional_retrieve_list"]
        assert isinstance(retrieve_list, list)
        assert "aiida.out" in retrieve_list
        assert "aiida-BANDSTRUCTURE*" in retrieve_list
        assert "bandstructure_SCF_and_G0W0" in retrieve_list
        assert "DOS_PDOS_SCF.out" in retrieve_list

    def test_build_scf_inputs_kpoints_injected(self, mock_structure, mock_code, mock_config):
        builder = Cp2kBuilder(mock_config).build_scf_inputs(
            structure=mock_structure,
            code=mock_code,
            protocol_section="single_point",
            protocol_name="protocol_SIRIUS.yml",
            kpoints_mesh=[4, 4, 1],
        )
        params = builder.cp2k.parameters.get_dict()
        dft = params.get("FORCE_EVAL", {}).get("DFT", {})
        kp = dft.get("KPOINTS", {})
        assert "MONKHORST-PACK 4 4 1" in kp.get("SCHEME", "")

    def test_build_gw_inputs_returns_builder(self, mock_structure, mock_code, mock_config):
        builder = Cp2kBuilder(mock_config).build_gw_inputs(
            structure=mock_structure,
            code=mock_code,
        )
        assert builder is not None
        assert hasattr(builder, "cp2k")
        assert builder.cp2k.code is mock_code

    def test_build_gw_inputs_gw_section_present(self, mock_structure, mock_code, mock_config):
        builder = Cp2kBuilder(mock_config).build_gw_inputs(
            structure=mock_structure,
            code=mock_code,
        )
        params = builder.cp2k.parameters.get_dict()
        bs = params.get("FORCE_EVAL", {}).get("PROPERTIES", {}).get("BANDSTRUCTURE", {})
        assert "GW" in bs
        assert "NUM_TIME_FREQ_POINTS" in bs["GW"]
        assert "KPOINTS_W" in bs["GW"]
