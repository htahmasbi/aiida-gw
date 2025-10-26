import sys
import yaml
import numpy as np
from time import sleep
from collections import defaultdict
from random import randint, uniform
from pyxtal.crystal import random_crystal
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from aiida.orm import Group, Dict
from aiida_datagen.codes.utils import get_allowed_n_atom_for_compositions, get_time, store_calculation_nodes
from aiida_datagen.codes.flame.flame_launch_calculations import GenSymCrysSubmissionController
from aiida_datagen.workflows.core import log_write, previous_run_exist_check, group_is_empty_check
from aiida_datagen.workflows.settings import inputs, job_script, steps_status
from aiida_datagen.codes.utils import is_structure_valid
from aiida_datagen.codes.flame.core import conf2pymatgenstructure

class random_crystal_3d(random_crystal):
    def set_volume(self):
        self.volume = self.factor

def collect_random_structures(outfile):
    random_bulk_structures = []
    try:
        confs = yaml.load_all(outfile, Loader=yaml.SafeLoader)
    except:
        return None, None
    pymatgen_structures = conf2pymatgenstructure(confs)
    nat = len(pymatgen_structures[0].sites)
    for a_pymatgen_structure in pymatgen_structures:
        if is_structure_valid(a_pymatgen_structure, False, False, True, False, False)[0]:
            random_bulk_structures.append(a_pymatgen_structure.as_dict())
    return nat, random_bulk_structures

def store_step2_results(results_dict):
    random_structures_group = Group.collection.get(label='random_structures')
    random_structures_group.clear()
    for a_key in results_dict.keys():
        a_node = Dict({a_key: results_dict[a_key]}).store()
        a_node.label = a_key
        random_structures_group.add_nodes(a_node)
        log_write(f'{len(results_dict[a_key])} random bulk structures with {a_key} atoms are generated'+'\n')

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

def step2_pyxtal(composition_list):
    results_dict = defaultdict(list)
    vpas = []
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'vpas' in a_node.label:
            vpas = a_node.get_list()
    for a_comp in composition_list:
        allowed_n_atom = get_allowed_n_atom_for_compositions([a_comp])
        attempts = max(round(2*(inputs['number_of_bulk_structures']/len(inputs['bulk_number_of_atoms']))/230), 1)
        elements = []
        n_elmnt = []
        for e, n in Composition(a_comp).items():
            elements.append(str(e))
            n_elmnt.append(int(n))
        for a_n_a in allowed_n_atom:
            random_bulk_structures = []
            n_element = [n_el*int(a_n_a/sum(n_elmnt)) for n_el in n_elmnt]
            for spg in range(1, 231):
                random_bulk_structures.extend(generate_random_bulk(spg, elements, n_element, vpas, attempts))
            results_dict[str(a_n_a)].extend(random_bulk_structures)
    # store resutls
    store_step2_results(results_dict)

def step2_flame(composition_list):
    data_dict = {}
    results_dict = defaultdict(list)

    for a_comp in composition_list:
        allowed_n_atom = get_allowed_n_atom_for_compositions([a_comp])
        attempts = max(round(5*(inputs['number_of_bulk_structures']/len(inputs['bulk_number_of_atoms']))/230), 1)
        data_dict[a_comp] = [allowed_n_atom, attempts]
    # submit jobs
    controller = GenSymCrysSubmissionController(
        group_label='wf_step2',
        max_concurrent=job_script['gensymcrys']['number_of_jobs'],
        data_dict=data_dict)
    # wait until all jobs are done
    while controller.num_to_run > 0 or controller.num_active_slots > 0:
        if controller.num_to_run > 0:
            controller.submit_new_batch(dry_run=False)
        sleep(60)
    # extract data
    wf_step2_group = Group.collection.get(label='wf_step2')
    for a_wf_node in wf_step2_group.nodes:
        a_node = a_wf_node.called[-1]
        if not a_node.is_finished_ok:
            continue
        output_folder = a_node.outputs.retrieved
        with output_folder.open('posout.yaml', 'rb') as fhandle:
            nat, random_bulk_structures = collect_random_structures(fhandle)
        if nat and random_bulk_structures:
            results_dict[str(nat)].extend(random_bulk_structures)
    # store resutls
    store_step2_results(results_dict)

def step_2():
    """ Step 2
    """
    log_write("---------------------------------------------------------------------------------------------------"+'\n')
    log_write('STEP 2'+'\n')
    log_write(f'start time: {get_time()}'+'\n')
    if 'scratch' in inputs['calculation_type']:
        # check
        previous_run_exist_check()
        group_is_empty_check('wf_step2')

        composition_list = inputs['Chemical_formula']
        if len(composition_list) ==  0:
            log_write('>>> ERROR: no composition is provided <<<'+'\n')
            sys.exit()
        if len(inputs['bulk_number_of_atoms']) < 1 or not inputs['number_of_bulk_structures']:
            log_write('>>> ERROR: data for step 2 is not complete'+'\n')
            sys.exit()
        if inputs['random_structure_generator'] in ['PyXtal', 'pyxtal']:
            log_write('random structure generation with pyxtal'+'\n')
            step2_pyxtal(composition_list)
        else:
            log_write('random structure generation with gensymcrys'+'\n')
            step2_flame(composition_list)
    log_write('STEP 2 ended'+'\n')
    log_write(f'end time: {get_time()}'+'\n')
    if not steps_status[2]:
        store_calculation_nodes()
        log_write('End of the step 2. Bye!'+'\n')
    return steps_status[2]
