import sys
from collections import defaultdict
from pymatgen.core.composition import Composition
from aiida.orm import Group, Dict
from aiida_gw.codes.utils import get_allowed_n_atom_for_compositions, get_time, store_calculation_nodes
from aiida_gw.workflows.core import log_write, previous_run_exist_check, group_is_empty_check
from aiida_gw.workflows.settings import inputs, steps_status

def store_step2_results(results_dict):
    random_structures_group = Group.collection.get(label='random_structures')
    random_structures_group.clear()
    for a_key in results_dict.keys():
        a_node = Dict({a_key: results_dict[a_key]}).store()
        a_node.label = a_key
        random_structures_group.add_nodes(a_node)
        log_write(f'{len(results_dict[a_key])} random bulk structures with {a_key} atoms are generated'+'\n')

def step2_pyxtal(composition_list):
    from aiida_gw.codes.pxtl import generate_random_bulk
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
        if not composition_list:
            log_write('>>> ERROR: no composition is provided <<<'+'\n')
            sys.exit()
        if not inputs['number_of_bulk_structures']:
            log_write('>>> ERROR: data for step 2 is not complete'+'\n')
            sys.exit()
        if len(inputs['bulk_number_of_atoms']) < 1:
            log_write('>>> ERROR: data for step 2 is not complete'+'\n')
            sys.exit()
        log_write('random structure generation with pyxtal'+'\n')
        step2_pyxtal(composition_list)
    log_write('STEP 2 ended'+'\n')
    log_write(f'end time: {get_time()}'+'\n')
    if not steps_status[2]:
        store_calculation_nodes()
        log_write('End of the step 2. Bye!'+'\n')
    return steps_status[2]
