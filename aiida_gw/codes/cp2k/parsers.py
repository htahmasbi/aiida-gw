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
    cell_lines = [line.strip().split() for line in match.group(1).splitlines()]
    cell_str = [line[2:] for line in cell_lines if line[0] in "ABC"]
    cell = np.array(cell_str, np.float64)
    return cell


def read_bandstructure(content):
    """Parse CP2K GW bandstructure output file.

    Parses files like 'bandstructure_SCF_and_G0W0' which contain:
    - K-point coordinates and labels
    - Eigenvalues at each k-point for SCF, GW, etc.

    Returns:
        dict with keys:
            - kpoints: list of k-point coordinates (nx3 arrays)
            - kpoint_labels: list of k-point labels (e.g. "GAMMA", "K", "M")
            - eigenvalues: dict of {level_name: 2D array (nkpoints x nbands)}
              e.g. {"SCF": np.array, "G0W0": np.array, "G0W0+SOC": np.array}
            - units: str ("eV")
    """
    lines = content.splitlines()
    kpoints = []
    kpoint_labels = []
    eigenvalues = {}
    current_level = None
    nkpoints = 0
    nbands = 0
    eig_data = []

    for i_line, line in enumerate(lines):
        line_stripped = line.strip()

        if not line_stripped:
            continue

        if "SCF bands:" in line_stripped or "G0W0" in line_stripped or "SOC" in line_stripped:
            if current_level is not None and eig_data:
                eigenvalues[current_level] = np.array(eig_data)
                eig_data = []

            if "SCF bands:" in line_stripped:
                current_level = "SCF"
            elif "G0W0+SOC" in line_stripped:
                current_level = "G0W0+SOC"
            elif "G0W0" in line_stripped:
                current_level = "G0W0"
            elif "SOC" in line_stripped:
                current_level = "SOC"
            continue

        if line_stripped.startswith("#") or line_stripped.startswith("k-point"):
            continue

        tokens = line_stripped.split()
        if len(tokens) >= 4:
            try:
                kx, ky, kz = float(tokens[0]), float(tokens[1]), float(tokens[2])
                kpoints.append([kx, ky, kz])
                if len(tokens) > 4:
                    kpoint_labels.append(tokens[4])
                else:
                    kpoint_labels.append("")
            except ValueError:
                pass

        if current_level is not None and len(tokens) >= 3:
            try:
                eig_vals = [float(t) for t in tokens if t not in ["eV", "Ry"]]
                if eig_vals:
                    eig_data.append(eig_vals)
            except ValueError:
                pass

    if current_level is not None and eig_data:
        eigenvalues[current_level] = np.array(eig_data)

    return {
        "kpoints": np.array(kpoints),
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
