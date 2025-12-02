import numpy as np
from random import randint, uniform
from pyxtal.crystal import random_crystal
from pymatgen.core.structure import Structure
from aiida_datagen.codes.utils import is_structure_valid
from aiida_datagen.workflows.settings import inputs

class random_crystal_3d(random_crystal):
    def set_volume(self):
        self.volume = self.factor

def generate_random_bulk(ini_spg, elements, n_element, vpas, attempts):
    min_d_prefactor = min(0.90, inputs['min_distance_prefactor'])
    random_structures = []
    failed_attempts = 0
    spg = ini_spg
    while len(random_structures) < attempts:
        vol = uniform(min(vpas), max(vpas)) * sum(n_element)
        try:
            r_b_3d = random_crystal_3d(3, spg, elements, n_element, vol)
        except:
            spg = randint(1, 230)
            continue
        lattice = r_b_3d.lattice.matrix
        coords = None
        species = []
        try:
            for site in r_b_3d.atom_sites:
                species.extend([site.specie] * site.multiplicity)
                coords = site.coords if coords is None else np.append(coords, site.coords, axis=0)
        except:
            spg = randint(1, 230)
            continue
        pymatgen_structure = Structure(lattice, species, coords)
        if is_structure_valid(pymatgen_structure, False, min_d_prefactor, True, False, False)[0]:
            random_structures.append(pymatgen_structure.as_dict())
            failed_attempts = 0
            spg = ini_spg
        else:
            failed_attempts = failed_attempts + 1
        if failed_attempts > 5:
            spg = randint(1, 230)
    return random_structures
