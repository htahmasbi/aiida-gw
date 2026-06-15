"""Tests for config models."""

from pydantic import ValidationError

from aiida_gw.core.config import Cp2kConfig, MetadataOptions


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
            num_machines=2, num_mpiprocs_per_machine=64, memory_per_machine_mb=38400
        )
        d = opts.to_dict()
        assert d["resources"]["num_machines"] == 2
        assert d["resources"]["num_mpiprocs_per_machine"] == 64
        assert d["custom_scheduler_commands"] == "#SBATCH --mem=38400M\n"

    def test_to_dict_memory_none(self):
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
        import pytest
        with pytest.raises(ValidationError):
            Cp2kConfig(kpoints_mesh=[4, 1])

    def test_kpoints_mesh_zero_element(self):
        import pytest
        with pytest.raises(ValidationError):
            Cp2kConfig(kpoints_mesh=[4, 0, 4])

    def test_kpoints_mesh_negative(self):
        import pytest
        with pytest.raises(ValidationError):
            Cp2kConfig(kpoints_mesh=[-1, 1, 1])
