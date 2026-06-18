# aiida-gw

![Python Version](https://img.shields.io/badge/python-≥%203.10-blue)
![AiiDA Version](https://img.shields.io/badge/AiiDA-≥%202.5-orange)
[![CI](https://github.com/htahmasbi/aiida-gw/actions/workflows/ci.yml/badge.svg)](https://github.com/htahmasbi/aiida-gw/actions/workflows/ci.yml)

AiiDA plugin for CP2K GW workflows targeting 2D materials.

Forked from [aiida-datagen](https://github.com/hmhoseini/aiida-datagen), refactored to focus on GW calculations with CP2K only.

## Installation

```bash
git clone https://github.com/htahmasbi/aiida-gw.git
cd aiida-gw
pip install -e .
```

## Configuration

Create a `.env` file or set environment variables. All variables use the prefix `AIIDA_GW_` with nested keys separated by `__`:

| Variable | Default | Description |
| :--- | :---: | :--- |
| **AiiDA Core** | | |
| `AIIDA_GW_PROFILE` | *Required* | AiiDA profile name |
| `AIIDA_GW_CODE_LABEL` | `cp2k@localhost` | Code label for CP2K |
| **Cluster Resources** | | |
| `AIIDA_GW_NUM_MACHINES` | `1` | Number of compute nodes |
| `AIIDA_GW_NUM_MPIPROCS` | `8` | MPI processes per node |
| `AIIDA_GW_WALLTIME` | `36000` | Max wallclock time in seconds (10 hours) |
| `AIIDA_GW_METADATA_OPTIONS__MEMORY_PER_MACHINE` | *Optional* | Memory per node (e.g., `600G` or `38400M` $\rightarrow$ `#SBATCH --mem`) |
| `AIIDA_GW_METADATA_OPTIONS__PARTITION` | *Optional* | SLURM partition ($\rightarrow$ `#SBATCH --partition`) |
| **CP2K Physics** | | |
| `AIIDA_GW_CUTOFF` | `400` | CP2K plane-wave cutoff (Ry) |
| `AIIDA_GW_REL_CUTOFF` | `50` | Relative cutoff (Ry) |
| `AIIDA_GW_VACUUM` | `20` | Vacuum gap (Å) |
| `AIIDA_GW_SUPERCELL` | `2,2,1` | Supercell dimensions ($x, y, z$) |
| `AIIDA_GW_KPOINTS` | `12,1,12` | Ground-state K-point mesh |
| `AIIDA_GW_GW_KPOINTS` | `12,1,12` | GW step K-point mesh |
| `AIIDA_GW_GW_KPOINTS_W` | *Optional* | GW `KPOINTS_W` mesh (if different from `GW_KPOINTS`) |
| **Files & Basis Sets** | | |
| `AIIDA_GW_RESOLVE_FROM_FILES` | `True` | Automatically resolve orbital/potential/RI from data files |
| `AIIDA_GW_ORB_BASIS` | `aug-SZV-MOLOPT-GTH-tier-1` | Orbital basis set name |
| `AIIDA_GW_BASIS_SET_FILE` | `BASIS_GTH_MOLOPT_AUG_for_excited_states` | Primary orbital basis set file (cluster path) |
| `AIIDA_GW_RI_BASIS_SET_FILE` | `BASIS_GTH_MOLOPT_AUG_for_excited_states_RI` | RI basis set file (cluster path) |
| `AIIDA_GW_POTENTIAL_FILE` | `POTENTIAL_UZH` | Potential file (cluster path) |
| `AIIDA_GW_RI_BASIS_ACCURACY_TARGET` | *Optional* | Numerical accuracy target for automated RI selection |

Alternatively, place a ``config.toml`` file in the project root, ``~/.config/aiida-gw/``,
or the current directory:

```toml
[metadata_options]
num_machines = 2
num_mpiprocs_per_machine = 64
max_wallclock_seconds = 86400
memory_per_machine = "600G"
partition = "cpu-genoa"

[gw]
basis_set_file = "/path/to/BASIS_GTH_MOLOPT_AUG_for_excited_states"
ri_basis_set_file = "/path/to/BASIS_GTH_MOLOPT_AUG_for_excited_states_RI"
potential_file = "/path/to/POTENTIAL_UZH"
orb_basis = "aug-SZV-MOLOPT-GTH-tier-1"
resolve_from_files = true
kpoints_mesh = [6, 1, 6]
vacuum = 20.0
supercell = [3, 3, 1]
```

## Usage

```bash
# Show configuration
aiida-gw config-show

# Run a single-point calculation (SIRIUS mode)
aiida-gw run --mode single_point --structure structure.cif --code cp2k@localhost

# Run a relaxation (SIRIUS mode)
aiida-gw run --mode relax --structure structure.cif --code cp2k@localhost

# Run GW calculation (QS/GPW mode)
aiida-gw run --mode gw --structure structure.cif --code cp2k@localhost --vacuum 20 --supercell 3,3,1

# Run GW with explicit k-points
aiida-gw run --mode gw --structure bn.cif --kpoints 6,1,6 --kpoints-w 6,1,6

# Fetch structures from MC2D via OPTIMADE (with element exclusion)
aiida-gw fetch --group mc2d_structures --max 10 --elements B,N
aiida-gw fetch --group gw_ready --max 50 --exclude-elements La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu

# Save all MC2D structures grouped by element count into JSON files
aiida-gw fetch-json --output ./structures

# Run GW on OPTIMADE structures (auto-detects unsupported elements from data files)
aiida-gw run --mode gw --group mc2d_structures --code cp2k@localhost

# Run GW with explicit element exclusion
aiida-gw run --mode gw --group mc2d_structures --code cp2k@localhost --exclude-elements La,Ce,Pr
```

## Project Structure

```
aiida-gw/
├── aiida_gw/
│   ├── cli.py                  # Typer CLI entry point
│   ├── core/
│   │   ├── config.py           # Pydantic configuration
│   │   ├── builders.py         # CP2K input builder (protocol-driven)
│   │   ├── enums.py            # Calculation modes
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── logging.py          # Python logging setup
│   ├── codes/
│   │   └── cp2k/
│   │       ├── __init__.py     # Exports CP2K_FILES_PATH
│   │       ├── cp2k_parsers.py # AiiDA output parsers
│   │       ├── parsers.py      # CP2K output parsing utilities
│   │       ├── data_reader.py  # CP2K data file reader (basis/potential/RI resolution)
│   │       └── cp2k_files/     # Protocol YAMLs, basis sets, pseudopotentials
│   ├── datasets/
│   │   ├── mc2d_optimade.py    # OPTIMADE structure fetcher
│   ├── transformations/
│   │   └── structures.py       # 2D structure helpers (vacuum, rotation, supercell)
│   └── workflows/
│       ├── single_point.py     # SinglePointWorkChain
│       ├── relaxation.py       # RelaxWorkChain
│       ├── gw.py               # GwWorkChain
│       └── archive/            # Original pipeline (step1/2/3, main, settings)
├── setup.json
└── README.md
```

## Workflows

- **SinglePointWorkChain** — Single-point energy/force calculation using `Cp2kBaseWorkChain`
- **RelaxWorkChain** — Geometry relaxation using `Cp2kBaseWorkChain`
- **GwWorkChain** — GW calculation (SCF → GW) with structure preparation (vacuum, supercell)

## License

MIT

## Contact

h.tahmasb@gmail.com
