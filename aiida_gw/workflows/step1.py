import os
import sys
from time import sleep
from itertools import combinations_with_replacement
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from aiida.orm import Group, List, Dict, load_node, QueryBuilder, WorkChainNode, CalcJobNode
from aiida.plugins import DataFactory
from aiida_gw.codes.utils import get_element_list, get_structures_from_mpdb, get_reference_structures, get_time, store_calculation_nodes
from aiida_gw.workflows.core import log_write, previous_run_exist_check, report
from aiida_gw.workflows.settings import inputs, output_dir, groups, steps_status, job_script

def store_step1_results(vpas_db):
    """ Store results
    """
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if a_node.label in ['dimers', 'vpas', 'epas']:
            known_structures_group.remove_nodes([load_node(a_node.pk)])
    calculation_nodes_group = Group.collection.get(label='calculation_nodes')
    for a_node in calculation_nodes_group.nodes:
        if 'step_1' in a_node.label:
            calculation_nodes_group.remove_nodes([load_node(a_node.pk)])

    builder = QueryBuilder()
    builder.append(Group, filters={'label': 'results_step1'}, tag='results_step1_group')
    builder.append(WorkChainNode, with_group='results_step1_group', tag='wf_nodes')
    builder.append(CalcJobNode, with_incoming='wf_nodes', tag= 'calcjob_nodes', project='*')
    calcjob_nodes = builder.all(flat=True)

    dimers = {}
    epas = []
    minmaxepa = []
    vpas_dimer = []
    calculation_nodes = []
    covalent_radius = CovalentRadius.radius
    composition_list = inputs['Chemical_formula']
    element_list = get_element_list()
    for a_node in calcjob_nodes:
        if not a_node.is_finished_ok:
            continue
        if 'VASP' in inputs['ab_initio_code']:
            if not a_node.outputs.misc.dict.run_status['electronic_converged']:
                continue
            coords = []
            species = []
            lattice = a_node.outputs.structure.cell
            for a_site in a_node.outputs.structure.sites:
                species.append(a_site.kind_name)
                coords.append(a_site.position)
            pymatgen_structure = Structure(lattice, species, coords, to_unit_cell=True, coords_are_cartesian=True)
            epot = float(a_node.outputs.energies.get_array('energy_extrapolated_electronic')[-1])
        if 'SIRIUS' in inputs['ab_initio_code'] or 'QS' in inputs['ab_initio_code']:
            output_parameters = a_node.base.links.get_outgoing(link_label_filter='output_parameters').all_nodes()[0]
            motion_step = output_parameters['motion_step_info']
            if not motion_step['scf_converged']:
                continue
            coords = []
            species = []
            lattice = a_node.outputs.output_structure.cell
            for a_site in a_node.outputs.output_structure.sites:
                species.append(a_site.kind_name)
                coords.append(a_site.position)
            pymatgen_structure = Structure(lattice, species, coords, to_unit_cell=True, coords_are_cartesian=True)
            epot = float(a_node.outputs.output_parameters.dict.energy)
        if 'dimer' in a_node.label:
            labels = pymatgen_structure.labels
            a_dimer = '-'.join(labels)
            d = pymatgen_structure.distance_matrix[0, 1]
            if d <= covalent_radius[labels[0]] + covalent_radius[labels[-1]]:
                dimers[a_dimer]= d
                if pymatgen_structure.composition.is_element:
                    dimers[labels[0]] = d/2
                else:
                    a_r_dimer = '-'.join(list(reversed(labels)))
                    dimers[a_r_dimer] = d
            continue
        nat = len(pymatgen_structure.sites)
        epas.append(epot/nat)
        calculation_nodes.append(a_node.pk)
    # dimers
    for a_combination in combinations_with_replacement(element_list, 2):
        a_dimer = '-'.join(a_combination)
        if a_dimer in dimers.keys():
            log_write(f'{a_dimer}: {round(dimers[a_dimer], 2)} A (calculated from dimers)'+'\n')
        else:
            d = covalent_radius[a_combination[0]] + covalent_radius[a_combination[-1]]
            if a_combination[0] == a_combination[-1]:
                dimers[a_dimer] = 0.85 * d
                dimers[a_combination[0]] = d/2
            else:
                dimers[a_dimer] = 0.85 * d
                a_r_dimer = '-'.join(list(reversed(a_combination)))
                dimers[a_r_dimer] = 0.85 * d
            log_write(f'{a_dimer}: {round(dimers[a_dimer], 2)} A (calculated from covalent radii)'+ '\n')
    # vpas_db
    if vpas_db:
        if round(max(vpas_db), 2) != round(min(vpas_db), 2):
            log_write(f'min. and max. vpas from databases: {round(min(vpas_db), 2)} and {round(max(vpas_db), 2)} A^3/atom'+'\n')
        else:
            log_write(f'vpa calculated from database: {round(vpas_db[0], 2)} A^3/atom'+'\n')
    # vpas_dimer
    if len(vpas_db) < 3:
        c1= 27.5143396016886; c2= -15.1276388491108; c3= 2.73176264584233; a1= 1.62251902284598; a2= 3.06984018288488; a3= 4.79934235307803
        for a_composition in composition_list:
            elements = []
            nelement = []
            for elmnt, nelmnt in Composition(a_composition).items():
                elements.append(str(elmnt))
                nelement.append(int(nelmnt))
            vol = 0
            for i in range(len(elements)):
                vol += (c1 * dimers[elements[i]]**a1 + c2 * dimers[elements[i]]**a2 + c3 * dimers[elements[i]]**a3) * nelement[i]
            vpas_dimer.append(vol/sum(nelement))
        if round(max(vpas_dimer), 2) != round(min(vpas_dimer), 2):
            log_write(f'min. and max. vpas calculated from dimers: {round(min(vpas_dimer), 2)} and {round(max(vpas_dimer), 2)} A^3/atom'+'\n')
        else:
            log_write(f'vpa calculated from dimers: {round(vpas_dimer[0], 2)} A^3/atom'+'\n')
        vpas = vpas_db + vpas_dimer
    else:
        vpas = vpas_db
    # more vpas
    vpas.append(max(vpas)*1.05)
    vpas.append(min(vpas)*0.75)
    minmaxvpa = [min(vpas), max(vpas)]
    log_write(f'min. and max. vpas: {round(min(vpas), 2)} and {round(max(vpas), 2)} A^3/atom'+'\n')
    # epas
    if epas:
        log_write(f'min_epa: {round(min(epas), 8)}'+'\n')
    else:
        log_write('>>> WARNING: no minimum energy per atom is calculated (epa = 0.0 eV) <<<'+'\n')
        epas = [0.0]
    minmaxepa = [min(epas), max(epas)]
    # store
    a_node = Dict(dimers).store()
    a_node.label = 'dimers'
    known_structures_group.add_nodes(a_node)
    a_node = List(minmaxvpa).store()
    a_node.label = 'vpas'
    known_structures_group.add_nodes(a_node)
    a_node = List(minmaxepa).store()
    a_node.label = 'epas'
    known_structures_group.add_nodes(a_node)
    a_node = List(calculation_nodes).store()
    a_node.label = 'step_1'
    calculation_nodes_group.add_nodes(a_node)

def add_structures_to_parent_group(structures):
    """ Add structures to parent groups
    """
    # reference structures
    StructureData = DataFactory('structure')
    pg_step1_group = Group.collection.get(label='pg_step1')
    for i, a_structure in enumerate(structures):
        a_pymatgen_structure = Structure.from_dict(a_structure)
        nat = len(a_pymatgen_structure.sites)
        a_node = StructureData(pymatgen=a_pymatgen_structure).store()
        a_node.label = 'reference'
        a_node.base.extras.set('job', 'reference-'+str(i+1)+'_'+str(nat)+'-atoms')
        pg_step1_group.add_nodes(a_node)
   # dimers
#    covalent_radius = CovalentRadius.radius
#    element_list = get_element_list()
#    for i, a_dimer in enumerate(combinations_with_replacement(element_list, 2)):
#        d = covalent_radius[a_dimer[0]] + covalent_radius[a_dimer[-1]]
#        molecule = Molecule(a_dimer, [[0,0,0], [0,0,d]])
#        boxed_molecule = molecule.get_boxed_structure(15, 15, 15)
#        a_node = StructureData(pymatgen=boxed_molecule).store()
#        a_node.label = 'dimer'
#        a_node.base.extras.set('job', 'dimer-'+str(i+1))
#        pg_step1_group.add_nodes(a_node)

def step_1():
    """ Step 1
    """
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    log_write('Starting datagen'+'\n')
    log_write(f'type of calculation: {inputs["calculation_type"]}'+'\n')
    log_write('STEP 1'+'\n')
    log_write(f'start time: {get_time()}'+'\n')
    # check
    previous_run_exist_check()
    # create/clear groups
    for group_list in groups.values():
        for a_group_label in group_list:
            group, _ = Group.collection.get_or_create(a_group_label)
            group.clear()

    if 'scratch' in inputs['calculation_type']:
        if not inputs['Chemical_formula'] and not inputs['Chemical_system']:
            log_write('>>> ERROR: niether composition nor chemical system is provided <<<'+'\n')
            sys.exit()

        if inputs['Chemical_formula']:
            composition_list = inputs['Chemical_formula']
            log_write(f'Composition list: {composition_list}'+'\n')
            # get_known_structures
            l = get_structures_from_mpdb()
            if l == 0:
                log_write('>>> WARNING: no bulk structure was found in the MPD <<<'+'\n')
            else:
                log_write(f'Number of atomic structures from the MPD: {l}'+'\n')
            # Reference structures
            reference_structures, vpas_db = get_reference_structures(True)
            if reference_structures:
                log_write(f'Number of reference atomic structures from the MPD: {len(reference_structures)}'+'\n')
            else:
                log_write('>>> WARNING: no reference bulk structure was found in the MPD <<<'+'\n')
            # submit jobs
            if 'SIRIUS' in inputs['ab_initio_code'] or 'QS' in inputs['ab_initio_code']:
                # add structures to the parent group
                add_structures_to_parent_group([])
                from aiida_gw.codes.cp2k.cp2k_launch_calculations import CP2KSubmissionController
                log_write(f'Reference calculations with {inputs["ab_initio_code"]}'+'\n')
                controller = CP2KSubmissionController(
                    parent_group_label='pg_step1',
                    group_label='wf_step1',
                    max_concurrent=job_script['geopt']['number_of_jobs'],
                    QSorSIRIUS=inputs['ab_initio_code'])
            elif inputs['ab_initio_code']=='VASP':
                # add structures to the parent group
                add_structures_to_parent_group(reference_structures)
                from aiida_gw.codes.vasp.vasp_launch_calculations import VASPSubmissionController
                log_write('Reference calculations with VASP'+'\n')
                controller = VASPSubmissionController(
                    parent_group_label='pg_step1',
                    group_label='wf_step1',
                    max_concurrent=job_script['geopt']['number_of_jobs'])
            else:
                log_write('>>> ERROR: no ab_initio code is provided <<<'+'\n')
                sys.exit()
            # wait until all jobs are done
            while controller.num_to_run > 0 or controller.num_active_slots > 0:
                if controller.num_to_run > 0:
                    controller.submit_new_batch(dry_run=False)
                sleep(60)
            # report
            total_computing_time, submitted_jobs, finished_job = report('wf_step1')
            log_write(f'submitted jobs: {submitted_jobs}, succesful jobs: {finished_job}'+'\n')
            log_write(f'total computing time: {round(total_computing_time, 2)} core-hours'+'\n')
            # store step1 resutls
            store_step1_results(vpas_db)
    log_write('STEP 1 ended'+'\n')
    log_write(f'end time: {get_time()}'+'\n')
    if not steps_status[1]:
        store_calculation_nodes()
        log_write('End of the step 1. Bye!'+'\n')
    return steps_status[1]
