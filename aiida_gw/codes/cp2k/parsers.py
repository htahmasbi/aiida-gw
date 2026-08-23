import re
import numpy as np

def read_structure(content):
    """ Parse the structure from the restart file
    """
    match = re.search(r"\n\s*&COORD\n(.*?)\n\s*&END COORD\n", content, re.DOTALL)
    coord_lines = [line.strip().split() for line in match.group(1).splitlines()]

    symbols = []
    tags = []
    for atomic_kind in [line[0] for line in coord_lines]:
        symbols.append("".join([s for s in atomic_kind if not s.isdigit()]))
        try:
            tag = int("".join([s for s in atomic_kind if s.isdigit()]))
        except ValueError:
            tag = 0
        tags.append(tag)

    positions_str = [line[1:] for line in coord_lines]
    positions = np.array(positions_str, np.float64)

    match = re.search(r"\n\s*&CELL\n(.*?)\n\s*&END CELL\n", content, re.DOTALL)
    cell_lines = [line.strip().split() for line in match.group(1).splitlines()]
    cell_str = [line[1:] for line in cell_lines if line[0] in "ABC"]
    cell = np.array(cell_str, np.float64)

    cell_pbc = [True, True, True]

    return {
        "symbols": symbols,
        "positions": positions,
        "cell": cell,
        "tags": tags,
        "pbc": cell_pbc,
    }

def parse_cp2k_output_simple(fstring):
    """ Parse CP2K output into a dictionary
    """
    lines = fstring.splitlines()
    SIRIUS = False
    if 'SIRIUS' in fstring:
        SIRIUS = True
    result_dict = {"SIRIUS": SIRIUS}
    result_dict["cp2k_version"] = None
    result_dict["sirius_version"] = None
    energy = None
    bohr2ang = 0.529177208590000
    Eh2eV = 27.211324570273
    _bs_levels = {
        "SCF": "scf",
        "SCF+SOC": "scf_soc",
        "G0W0": "g0w0",
        "G0W0+SOC": "g0w0_soc",
        "Hartree-Fock with SCF orbitals": "hf",
    }
    _bs_quantities = {
        "valence band maximum": "vbm",
        "conduction band minimum": "cbm",
        "indirect band gap": "gap_indirect",
        "direct band gap": "gap_direct",
    }

    for i_line, line in enumerate(lines):
        if line.startswith(" CP2K| version string:"):
            cp2k_version = ' '.join(line.split()[3:])
            result_dict["cp2k_version"] = cp2k_version

        if line.startswith("SIRIUS "):
            sirius_version = ' '.join(line.split())
            result_dict["sirius_version"] = sirius_version

        if line.startswith(" GLOBAL| Run type"):
            result_dict["run_type"] = line.split()[-1]

        if line.startswith("[unit cell] lattice vectors"):
            lattice_vertor_A = [bohr2ang*float(lines[i_line+1].split()[4]),
                                bohr2ang*float(lines[i_line+1].split()[5]),
                                bohr2ang*float(lines[i_line+1].split()[6])]
            lattice_vertor_B = [bohr2ang*float(lines[i_line+2].split()[4]),
                                bohr2ang*float(lines[i_line+2].split()[5]),
                                bohr2ang*float(lines[i_line+2].split()[6])]
            lattice_vertor_C = [bohr2ang*float(lines[i_line+3].split()[4]),
                                bohr2ang*float(lines[i_line+3].split()[5]),
                                bohr2ang*float(lines[i_line+3].split()[6])]
            result_dict["lattice_vectors"] = np.array([lattice_vertor_A,
                                                       lattice_vertor_B,
                                                       lattice_vertor_C], np.float64)
        if "The number of warnings for this run is" in line:
            result_dict["nwarnings"] = int(line.split()[-1])

        if line.startswith(" ENERGY| ") and "free" in line and "SIRIUS" in line:
            energy = float(line.split()[9])
            result_dict["energy"] = energy*Eh2eV
            result_dict["energy_units"] = "eV"

        if line.startswith(" ENERGY| ") and "energy" in line and "QS" in line:
            energy = float(line.split()[8])
            result_dict["energy"] = energy*Eh2eV
            result_dict["energy_units"] = "eV"

        if line.startswith(" "):
            for level, level_key in _bs_levels.items():
                if not line.startswith(f" {level} "):
                    continue
                for quantity, quantity_key in _bs_quantities.items():
                    if f"{quantity} (eV)" in line:
                        result_dict[f"{level_key}_{quantity_key}"] = float(line.split()[-1])
                break

        if "run_type" in result_dict.keys():
            # Initialization
            if "motion_step_info" not in result_dict:
                result_dict["motion_opt_converged"] = False
                result_dict["motion_step_info"] = {
                    "step": [],  # MOTION step
                    "energy_eV": [],  # total energy
                    "scf_converged": [],  # SCF converged in this motions step (bool)
                }
                step = 0
                energy = None
                if SIRIUS:
                    scf_converged = False
                else:
                    scf_converged = True
            print_now = False
            data = line.split()
            if re.search(r"SCF run NOT converged", line):
                scf_converged = False
            if re.search(r"converged after", line):
                scf_converged = True
            if result_dict["run_type"] in ["ENERGY_FORCE"]:
                if energy is not None and not result_dict["motion_step_info"]["step"]:
                    print_now = True
                    if "converged after" in fstring:
                        scf_converged = True
            if result_dict["run_type"] in ["GEO_OPT", "CELL_OPT"]:
                # Note: with CELL_OPT/LBFGS there is no "STEP 0", while there is with CELL_OPT/BFGS
                if re.search(r"Informations at step", line):
                    step = int(data[5])
                elif re.search(r"Step number", line):
                    step = int(data[3])
                if (len(data) == 1 and data[0] == "---------------------------------------------------") or\
                   re.search(r"Estimated peak process memory after this step", line):
                    print_now = True
                if re.search(
                    r"Reevaluating energy at the minimum", line):
                    result_dict["motion_opt_converged"] = True

            if print_now and energy is not None:
                if step == 0 and result_dict["run_type"] in ["GEO_OPT", "CELL_OPT"]: #BFGS or CS
                    continue
                result_dict["motion_step_info"]["step"].append(step)
                result_dict["motion_step_info"]["energy_eV"].append(energy*Eh2eV)
                result_dict["motion_step_info"]["scf_converged"].append(scf_converged)
                if SIRIUS:
                    scf_converged = False
                else:
                    scf_converged = True
    return result_dict

def parse_lines(lines, start, end):
    parsed_lines = []
    for line in lines[start:end]:
        parsed_lines.append(line.split()[-3:])
    return parsed_lines

def read_positions(content):
    start_line = []
    positions = []
    pattern = re.compile("^\s[i]", re.MULTILINE)
    lines = content.splitlines()
    for i_line, line in enumerate(lines):
        for match in re.finditer(pattern, line):
            start_line.append(i_line+1)
    nlines = start_line[1]-start_line[0]-2
    for a_s_l in start_line:
        parsed_lines = parse_lines(lines, a_s_l, a_s_l+nlines)
        positions.append(np.array(parsed_lines, np.float64))
    return positions

def read_coordinates(content):
    """ ENERGY FORCE calculations
    """
    coordinates_str = []
    symbols = []
    lines = content.splitlines()[2:]
    for line in lines:
        coordinates_str.append(line.split()[-3:])
        symbols.append(line.split()[0])
    coordinates = [np.array(coordinates_str, np.float64)]
    return symbols, coordinates

def read_s_p_forces(content):
    """ ENERGY FORCE calculations
    """
    HaB2eVA = 51.42208619083232
    lines = content.splitlines()
    for i_line, line in enumerate(lines):
        if line.startswith(" # Atom"):
            start_line = i_line + 1
        if line.startswith(" SUM"):
            end_line = i_line
    parsed_lines = parse_lines(lines, start_line, end_line)
    s_p_forces = [(np.array(parsed_lines, np.float64)) * HaB2eVA]
    return s_p_forces

def read_s_p_stress_tensor(content):
    """ ENERGY FORCE calculations
    """
    stress_tensor = []
    lines = content.splitlines()[3:6]
    for line in lines:
        stress_tensor.extend(np.array(line.split()[2:], np.float64) * 1000) # [bar]
    return stress_tensor

def read_forces(content):
    HaB2eVA = 51.42208619083232
    start_lines = []
    forces = []
    pattern = re.compile("^\s[i]", re.MULTILINE)
    lines = content.splitlines()
    for i_line, line in enumerate(lines):
        for match in re.finditer(pattern, line):
            start_lines.append(i_line+1)
    nlines = start_lines[1]-start_lines[0]-2
    for a_s_l in start_lines:
        parsed_lines = parse_lines(lines, a_s_l, a_s_l+nlines)
        forces.append(np.array(parsed_lines, np.float64) * HaB2eVA)
    return forces

def read_stress_tensor(content):
    stress_tensor = []
    lines = content.splitlines()[1:]
    for line in lines:
        stress_tensor.append(np.array(line.split()[2:], np.float64)) # xx, xy, xz, yx, yy, yz, zx, zy, zz [bar]
    return stress_tensor

def read_cell_parameters(content):
    cell_parameters = []
    lines = content.splitlines()[1:]
    for line in lines:
        cell_a = line.split()[2:5] # Angstrom
        cell_b = line.split()[5:8]
        cell_c = line.split()[8:11]
        cell_parameters.append(np.array([cell_a, cell_b, cell_c], np.float64))
    return cell_parameters

def read_lattice_parameters(content):
    match = re.search(r"\n\s*&CELL\n(.*?)\n\s*&END CELL\n", content, re.DOTALL)
    if match is None:
        raise ValueError("No &CELL section found in CP2K input")
    cell = np.full((3, 3), np.nan)
    for raw_line in match.group(1).splitlines():
        tokens = raw_line.split("#")[0].strip().split()
        if len(tokens) >= 4 and tokens[0].upper() in ("A", "B", "C"):
            idx = "ABC".index(tokens[0].upper())
            cell[idx, :] = np.array(tokens[1:4], np.float64)
    if np.isnan(cell).any():
        raise ValueError("Could not parse A/B/C lattice vectors from &CELL section")
    return cell


_FLOAT_RE = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
_KPOINT_HEADER_RE = re.compile(
    r"kpoint:\s*\d+\s+coordinate:\s*"
    rf"({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})",
    re.IGNORECASE,
)
_BAND_ROW_RE = re.compile(r"^\s*(\d+)\s+\((occ|vir)\)\s+(\d+)\s+(.*)$")
_SPIN_RE = re.compile(r"SPIN:\s*(\d+)")
_DAT_COORDS_RE = re.compile(_FLOAT_RE)


def _level_tag(label):
    """Map a column label from the xTP header to an energy-level name."""
    upper = label.upper()
    for needle, tag in (
        ("G0W0+SOC", "G0W0+SOC"),
        ("G0W0", "G0W0"),
        ("GW", "G0W0"),
        ("DFT", "DFT"),
    ):
        if needle in upper:
            return tag
    return None


def read_bandstructure(content):
    """Parse CP2K bandstructure output files.

    Supports two formats:

    1. CP2K xTP (GW) files like 'bandstructure_SCF_and_G0W0', consisting of
       repeated blocks::

           kpoint:       1             coordinate:     0.0000    0.0000    0.0000
               n           k   <epsilon>_nk^DFT (eV)  ...  <epsilon>_nk^G0W0 (eV)
               1 (occ)     1         -75.519    ...       -86.246

       The '^'-tagged header columns define the energy levels returned
       (e.g. 'DFT', 'G0W0'); each data row is ``n (occ|vir) k <ncols floats>``.

    2. Standard CP2K '.dat' band files like 'aiida-BANDSTRUCTURE_1-1.dat'
       with '# KPOINT'/'# SET: .. SPIN: n' comments and one eigenvalue per
       line; levels are named 'spin_<n>'.

    Returns:
        dict with keys:
            - kpoints: (nkpoints x 3) array of fractional coordinates
            - kpoint_labels: list (empty strings; xTP files carry no labels)
            - eigenvalues: dict of {level_name: (nkpoints x nbands) array}
            - units: str ("eV")

    Raises:
        ValueError: on structurally inconsistent data (ragged bands, column
            count mismatch).
    """
    lines = content.splitlines()
    kpoints = []
    kpoint_labels = []
    eigenvalues = {}

    level_columns = []
    n_value_cols = 0
    pending = {}
    open_kpt = False
    current_spin = None

    def _flush():
        nonlocal pending, open_kpt
        if not open_kpt:
            return
        for level, rows in pending.items():
            eigenvalues.setdefault(level, []).append(rows)
        pending = {}
        open_kpt = False

    for raw_line in lines:
        line_stripped = raw_line.strip()
        if not line_stripped:
            continue

        match = _KPOINT_HEADER_RE.search(line_stripped)
        if match is not None:
            _flush()
            kpoints.append([float(match.group(i)) for i in (1, 2, 3)])
            kpoint_labels.append("")
            open_kpt = True
            continue

        if line_stripped.startswith("#"):
            upper = line_stripped.upper()
            if "KPOINT" in upper:
                _flush()
                coords = _DAT_COORDS_RE.findall(line_stripped.split(":")[-1])
                if len(coords) >= 3:
                    kpoints.append([float(c) for c in coords[:3]])
                    kpoint_labels.append("")
                    open_kpt = True
            else:
                spin_match = _SPIN_RE.search(upper)
                if spin_match is not None:
                    current_spin = f"spin_{spin_match.group(1)}"
            continue

        tokens = line_stripped.split()

        if "^" in line_stripped and "EV" in line_stripped.upper():
            level_columns = [_level_tag(tok) for tok in tokens if "^" in tok]
            n_value_cols = len(level_columns)
            continue

        band_match = _BAND_ROW_RE.match(line_stripped)
        if band_match is not None and open_kpt:
            try:
                vals = [float(tok) for tok in band_match.group(4).split()]
            except ValueError:
                continue
            if len(vals) != n_value_cols:
                raise ValueError(
                    f"Band row has {len(vals)} value(s), expected {n_value_cols}: "
                    f"'{line_stripped}'"
                )
            for tag, val in zip(level_columns, vals):
                if tag is not None:
                    pending.setdefault(tag, []).append(val)
            continue

        if open_kpt:
            try:
                vals = [float(tok) for tok in tokens]
            except ValueError:
                continue
            if vals:
                level = current_spin or "default"
                pending.setdefault(level, []).extend(vals)

    _flush()

    for level, rows in eigenvalues.items():
        eigenvalues[level] = np.array(rows, dtype=np.float64)

    return {
        "kpoints": np.array(kpoints, dtype=np.float64),
        "kpoint_labels": kpoint_labels,
        "eigenvalues": eigenvalues,
        "units": "eV",
    }


def read_dos_pdos(content):
    """Parse CP2K DOS/PDOS output file.

    Parses files like 'DOS_PDOS_G0W0.out' which contain:
    - Energy values (eV)
    - Total DOS
    - Projected DOS per atom/orbital

    Returns:
        dict with keys:
            - energy: 1D array of energy values (eV)
            - total_dos: 1D array of total DOS values
            - pdos: dict of {label: 1D array} for projected DOS
            - fermi_energy: float (eV) or None
            - units: str ("eV", "1/eV")
    """
    lines = content.splitlines()
    energy = []
    total_dos = []
    pdos = {}
    current_pdos_label = None
    fermi_energy = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            continue

        if "E (eV)" in line_stripped or "Energy" in line_stripped:
            continue

        if "Fermi energy" in line_stripped or "Fermi" in line_stripped:
            try:
                fermi_energy = float(line_stripped.split()[-1])
            except (ValueError, IndexError):
                pass
            continue

        tokens = line_stripped.split()
        if len(tokens) >= 2:
            try:
                energy.append(float(tokens[0]))
                total_dos.append(float(tokens[1]))
            except ValueError:
                pass

        if len(tokens) >= 4 and not tokens[0].replace(".", "").replace("-", "").isdigit():
            label = " ".join(tokens[:-2])
            try:
                val = float(tokens[-1])
                if label not in pdos:
                    pdos[label] = []
                pdos[label].append(val)
            except ValueError:
                pass

    result = {
        "energy": np.array(energy),
        "total_dos": np.array(total_dos),
        "pdos": {k: np.array(v) for k, v in pdos.items()},
        "units": {"energy": "eV", "dos": "1/eV"},
    }
    if fermi_energy is not None:
        result["fermi_energy"] = fermi_energy

    return result
