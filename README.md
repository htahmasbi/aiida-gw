# aiida-gw

![Python Version](https://img.shields.io/badge/python-≥%203.10-blue)
![AiiDA Version](https://img.shields.io/badge/AiiDA-≥%202.5-orange)

AiiDA plugin for CP2K GW workflows targeting 2D materials.

Forked from [aiida-datagen](https://github.com/hmhoseini/aiida-datagen), refactored to focus on GW calculations with CP2K only.

## Installation

```bash
git clone https://github.com/htahmasbi/aiida-gw.git
cd aiida-gw
pip install -e .
```

## Configuration

Create a `.env` file or set environment variables:

| Variable | Default | Description |
|---|---|---|
| AIIDA_GW_CODE_LABEL | cp2k@localhost | Code label for CP2K |
| AIIDA_GW_PROFILE | profile_name | AiiDA profile name |
| AIIDA_GW_NUM_MACHINES | 1 | Number of nodes |
| AIIDA_GW_NUM_MPIPROCS | 128 | MPI processes per node |
| AIIDA_GW_WALLTIME | 43200 | Max wallclock seconds |
| AIIDA_GW_CUTOFF | 800 | CP2K cutoff (Ry) |
| AIIDA_GW_KPOINTS | 6,6,4 | K-point mesh |
| AIIDA_GW_GW_KPOINTS | 6,6,4 | GW k-point mesh |
| AIIDA_GW_VACUUM | 20 | Vacuum gap (Å) |
| AIIDA_GW_SUPERCELL | 3,3,1 | Supercell size |

## Usage

```bash
# Show configuration
aiida-gw config-show

# Run a single-point calculation
aiida-gw run --mode single_point --structure structure.cif --code cp2k@localhost

# Run a relaxation
aiida-gw run --mode relax --structure structure.cif --code cp2k@localhost

# Run GW calculation
aiida-gw run --mode gw --structure structure.cif --code cp2k@localhost --vacuum 20 --supercell 3,3,1

# Fetch structures from MC2D via OPTIMADE
aiida-gw fetch --group mc2d_structures --max 10 --elements B,N

# Run GW on OPTIMADE structures
aiida-gw run --mode gw --optimade-group mc2d_structures --code cp2k@localhost
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
│   │       ├── parsers.py      # Parsing utilities
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
