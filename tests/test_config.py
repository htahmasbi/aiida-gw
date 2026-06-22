"""Tests for config models."""

import pytest
from pydantic import ValidationError

from aiida_gw.core.config import Cp2kConfig, ElementOverride, GwConfig, MetadataOptions


class TestMetadataOptions:
    def test_to_dict(self):
        opts = MetadataOptions()
        d = opts.to_dict()
        assert d == {
            "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 8},
            "max_wallclock_seconds": 36000,
            "withmpi": True,
        }

    def test_to_dict_custom(self):
        opts = MetadataOptions(num_machines=4, num_mpiprocs_per_machine=16, max_wallclock_seconds=72000, withmpi=False)
        d = opts.to_dict()
        assert d == {
            "resources": {"num_machines": 4, "num_mpiprocs_per_machine": 16},
            "max_wallclock_seconds": 72000,
            "withmpi": False,
        }

    def test_to_dict_with_memory(self):
        opts = MetadataOptions(
            num_machines=2, num_mpiprocs_per_machine=64, memory_per_machine="38400M"
        )
        d = opts.to_dict()
        assert d["resources"]["num_machines"] == 2
        assert d["resources"]["num_mpiprocs_per_machine"] == 64
        assert "#SBATCH --mem=38400M" in d["custom_scheduler_commands"]

    def test_to_dict_memory_gb(self):
        opts = MetadataOptions(memory_per_machine="600G")
        d = opts.to_dict()
        assert d["custom_scheduler_commands"] == "#SBATCH --mem=600G\n"

    def test_to_dict_with_partition(self):
        opts = MetadataOptions(partition="cpu-genoa")
        d = opts.to_dict()
        assert d["custom_scheduler_commands"] == "#SBATCH --partition=cpu-genoa\n"

    def test_to_dict_memory_and_partition(self):
        opts = MetadataOptions(memory_per_machine="600G", partition="cpu-genoa")
        d = opts.to_dict()
        assert "#SBATCH --mem=600G" in d["custom_scheduler_commands"]
        assert "#SBATCH --partition=cpu-genoa" in d["custom_scheduler_commands"]

    def test_to_dict_no_scheduler_commands(self):
        opts = MetadataOptions()
        d = opts.to_dict()
        assert "custom_scheduler_commands" not in d


class TestCp2kConfig:
    def test_kpoints_mesh_valid(self):
        cfg = Cp2kConfig(kpoints_mesh=[4, 1, 4])
        assert cfg.kpoints_mesh == [4, 1, 4]

    def test_kpoints_mesh_none(self):
        cfg = Cp2kConfig(kpoints_mesh=None)
        assert cfg.kpoints_mesh is None

    def test_kpoints_mesh_invalid_length(self):
        with pytest.raises(ValidationError):
            Cp2kConfig(kpoints_mesh=[4, 1])

    def test_kpoints_mesh_zero_element(self):
        with pytest.raises(ValidationError):
            Cp2kConfig(kpoints_mesh=[4, 0, 4])

    def test_kpoints_mesh_negative(self):
        with pytest.raises(ValidationError):
            Cp2kConfig(kpoints_mesh=[-1, 1, 1])


class TestElementOverride:
    def test_defaults(self):
        ovr = ElementOverride()
        assert ovr.orb_basis is None
        assert ovr.potential is None
        assert ovr.ri_basis is None

    def test_all_fields(self):
        ovr = ElementOverride(orb_basis="TZV2P", potential="GTH-PBE-q3", ri_basis="RI_TZV2P")
        assert ovr.orb_basis == "TZV2P"
        assert ovr.potential == "GTH-PBE-q3"
        assert ovr.ri_basis == "RI_TZV2P"

    def test_partial(self):
        ovr = ElementOverride(orb_basis="DZVP")
        assert ovr.orb_basis == "DZVP"
        assert ovr.potential is None
        assert ovr.ri_basis is None


class TestGwConfigElementSettings:
    def test_empty_by_default(self):
        cfg = GwConfig()
        assert cfg.element_settings == {}

    def test_with_overrides(self):
        cfg = GwConfig(
            element_settings={
                "B": {"orb_basis": "DZVP-MOLOPT-PBE-GTH-q3", "potential": "GTH-PBE-q3"},
                "N": {"ri_basis": "RI_TZV2P"},
            }
        )
        assert "B" in cfg.element_settings
        assert cfg.element_settings["B"].orb_basis == "DZVP-MOLOPT-PBE-GTH-q3"
        assert cfg.element_settings["B"].potential == "GTH-PBE-q3"
        assert cfg.element_settings["B"].ri_basis is None
        assert cfg.element_settings["N"].ri_basis == "RI_TZV2P"


