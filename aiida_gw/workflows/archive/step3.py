import os
import sys
import gzip
from random import sample
from time import sleep
import json
import yaml
import numpy as np
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.structure_matcher import StructureMatcher
from aiida.plugins import DataFactory
from aiida.orm import List, Group, load_node, QueryBuilder, WorkChainNode, CalcJobNode
from aiida_gw.codes.utils import get_time, get_reference_structures, get_allowed_n_atom_for_compositions, is_structure_valid, store_calculation_nodes
from aiida_gw.workflows.core import log_write, previous_run_exist_check, group_is_empty_check, report
from aiida_gw.workflows.settings import inputs, steps_status, job_script, run_dir, output_dir
from utils.export_data import add_input_node, add_author_data, add_protocol, add_known_structures, add_calculation_nodes, add_nodes
from utils.extract import get_author_data, get_input_data, collect_data, get_protocol, plot_1, plot_2, plot_3

def export_data(dir_path):
    with open(os.path.join(output_dir, 'datagen.log'), 'r', encoding='utf-8') as fhandle:
        lines = fhandle.readlines()
    computing_time = 0.0
    for a_line in lines:
        if a_line.startswith('total'):
            computing_time += float(a_line.split()[3])
    with open(os.path.join(run_dir, 'author_data.yaml'), 'r', encoding='utf-8') as fhandle:
        author_data = yaml.safe_load(fhandle)
    author_data['computing_time'] = f'{computing_time} core-hours'
    with open(os.path.join(run_dir, 'author_data.yaml'), 'w', encoding='utf-8') as fhandle:
        yaml.dump(author_data, fhandle, default_flow_style=False)
    if 'single' in inputs['calculation_type']:
        file_path = os.path.join(dir_path, 'single_point'+'.aiida')
    elif inputs['Chemical_formula']:
        file_path = os.path.join(dir_path, inputs['Chemical_formula'][0]+'.aiida')
    elif inputs['Chemical_system']:
        file_path = os.path.join(dir_path, inputs['Chemical_system'][0]+'.aiida')
    pks = []
    pks, inputs_data = add_input_node()
    pks.append(add_author_data())
    pks.append(add_protocol(inputs_data))
    pks.extend(add_known_structures())
    pks.extend(add_calculation_nodes())
    add_nodes(pks)
    os.system(f"verdi archive create --no-call-calc-backward --no-call-work-backward --no-create-backward {file_path} --groups tmp_group")
    with open('node_pks.dat', 'w', encoding='utf-8') as fhandle:
        for a_pk in pks:
            fhandle.write(f'{a_pk}'+'\n')
    tmp_group, _ = Group.collection.get_or_create('tmp_group')
    Group.collection.delete(tmp_group.pk)

def extract_data(dir_path):
    group, _ = Group.collection.get_or_create('imported_calculation_nodes')
    group.clear()
    if 'single' in inputs['calculation_type']:
        file_path = os.path.join(dir_path, 'single_point'+'.aiida')
    elif inputs['Chemical_formula']:
        file_path = os.path.join(dir_path, inputs['Chemical_formula'][0]+'.aiida')
    elif inputs['Chemical_system']:
        file_path = os.path.join(dir_path, inputs['Chemical_system'][0]+'.aiida')

    os.system(f"verdi archive import -G imported_calculation_nodes {file_path}")
    author_data = get_author_data()
    input_data = get_input_data()
    code = input_data['ab_initio_code']
    collected_data, pps, input_parameters, code_version = collect_data(code)
    protocol = get_protocol(input_parameters, code)
    todump = {'Author data': author_data}

    prefix= "UPF "
    for k in pps:
        if pps[k].startswith(prefix):
            pps[k]= pps[k][len(prefix):]
    pps_string= json.dumps(pps)
    if 'single' in inputs['calculation_type']:
        todump.update(
            {'number of data': len(collected_data[0]),
             'number of bulks': len(collected_data[2]),
             'number of clusters': len(collected_data[5]),
             'code': inputs['ab_initio_code'],
             'code_version': code_version,
             'pps': pps_string,
             'protocol': protocol
            })
    elif inputs['Chemical_formula']:
        todump.update(
            {'Chemical formula': inputs['Chemical_formula'][0],
             'number of data': len(collected_data[0]),
             'number of bulks': len(collected_data[2]),
             'number of clusters': len(collected_data[5]),
             'code': inputs['ab_initio_code'],
             'code_version': code_version,
             'pps': pps_string,
             'protocol': protocol
            })
    elif inputs['Chemical_system']:
        todump.update(
            {'Chemical system': inputs['Chemical_system'][0],
             'number of data': len(collected_data[0]),
             'number of bulks': len(collected_data[2]),
             'number of clusters': len(collected_data[5]),
             'code': inputs['ab_initio_code'],
             'code_version': code_version,
             'pps': pps_string,
             'protocol': protocol
            })
    # store
    outputfilename= os.path.join(dir_path,'DATASET.json')
    trainingdatafilename= os.path.join(dir_path, 'training_data.json.gz')

    with open(outputfilename, 'w', encoding='utf-8') as fhandle:
        json.dump(todump, fhandle, indent=4)
    with gzip.open(trainingdatafilename, 'wt', encoding='UTF-8') as fhandle:
        json.dump(collected_data[0], fhandle)
    # plots
    plot_1(collected_data[2], collected_data[3], collected_data[4], collected_data[5], collected_data[6], dirname=dir_path)
    plot_2(collected_data[7], dirname=dir_path)
    plot_3(collected_data[0], dirname=dir_path)

#    pngfilelist= [file for file in os.listdir(dir_path) if file.endswith('.png')]
#    readme_file = f"# Dataset {inputs['Chemical_formula'][0]}\n\n"
#    readme_file += f"number of data: {todump['number of data']}, "
#    readme_file += f"number of bulks: {todump['number of bulks']}, "
#    readme_file += f"number of clusters: {todump['number of clusters']}\n\n"
#
#    readme_file += f"Generated with {todump['code']}\n\n"
#
#    for p in pngfilelist:
#        readme_file += f"![{p}]({p})\n\n"
#
#    with open(os.path.join(dir_path, "README.md"), 'w', encoding='utf-8') as outfile:
#        outfile.write(readme_file)

def store_step3_results():
    calculation_nodes = []
    group_label = 'results_step3'
    builder = QueryBuilder()
    builder.append(Group, filters={'label': group_label}, tag='results_group')
    builder.append(WorkChainNode, with_group='results_group', tag='wf_nodes')
    builder.append(CalcJobNode, with_incoming='wf_nodes', project='*')
    calcjob_nodes = builder.all(flat=True)
    if 'SIRIUS' in inputs['ab_initio_code'] or 'QS' in inputs['ab_initio_code']:
        for a_node in calcjob_nodes:
            if a_node.exit_status != 0:
                continue
            output_parameters = a_node.base.links.get_outgoing(link_label_filter='output_parameters').all_nodes()[0]
            motion_step = output_parameters['motion_step_info']
            if 'False' in motion_step['scf_converged']:
                continue
            calculation_nodes.append(a_node.pk)

    calculation_nodes_group = Group.collection.get(label='calculation_nodes')
    for a_node in calculation_nodes_group.nodes:
        if 'step_3' in a_node.label:
            calculation_nodes_group.remove_nodes([load_node(a_node.pk)])
    a_node = List(calculation_nodes).store()
    a_node.label = 'step_3'
    calculation_nodes_group.add_nodes(a_node)

def get_structures_from_nodes():
    """ Read known and random bulk structures
    """
    random_structures_group = Group.collection.get(label='random_structures')
    random_bulk_structures_dict = {}
    composition_list = inputs['Chemical_formula']
    if not composition_list:
        log_write('>>> ERROR: no composition is provided <<<'+'\n')
        sys.exit()

    allowed_n_atom_bulk = get_allowed_n_atom_for_compositions(composition_list)

    if inputs['bulk_number_of_atoms'] and inputs['number_of_bulk_structures'] != 0 and allowed_n_atom_bulk:
        n_struct_geopt = int(inputs['number_of_bulk_structures']/len(allowed_n_atom_bulk))
    else:
        log_write('>>> ERROR: data for step 3 is not complete. Check input.yaml <<<'+'\n')
        sys.exit()
    # read random bulk structures
    for a_node in random_structures_group.nodes:
        if int(a_node.label) in allowed_n_atom_bulk:
            random_bulk_structures_dict[int(a_node.label)] = a_node.get_dict()[a_node.label]
    return n_struct_geopt, random_bulk_structures_dict

def get_structures_references():
    bulk_structures = []
    # references
    if os.path.exists(os.path.join(run_dir,'local_db','bulk_structures.json')):
        with open(os.path.join(run_dir,'local_db','bulk_structures.json'), 'r', encoding='utf-8') as fhandle:
            bulk_structures = json.loads(fhandle.read())
    log_write(f'Number of reference bulk structures from the local database: {len(bulk_structures)}'+'\n')
    return bulk_structures

def get_structures_singlepoint():
    bulk_structures = []
    cluster_structures = []
    if os.path.exists(os.path.join(run_dir,'local_db','bulk_structures.json')):
        with open(os.path.join(run_dir,'local_db','bulk_structures.json'), 'r', encoding='utf-8') as fhandle:
            bulk_structures = json.loads(fhandle.read())
    if inputs['cluster_calculation'] and os.path.exists(os.path.join(run_dir,'local_db','molecule_structures.json')):
        with open(os.path.join(run_dir,'local_db','molecule_structures.json'), 'r', encoding='utf-8') as fhandle:
            cluster_structures = json.loads(fhandle.read())

    if os.path.exists(os.path.join(run_dir,'local_db','training_data.json.gz')):
        with gzip.open(os.path.join(run_dir,'local_db','training_data.json.gz'), 'rb') as fhandle:
            data_from_file = json.loads(fhandle.read())
        for a_data in data_from_file:
            if 'free' in a_data['bc'] and inputs['cluster_calculation']:
                cluster_structures.append(a_data['structure'])
            if 'bulk' in a_data['bc']:
                bulk_structures.append(a_data['structure'])
    log_write(f'Number of bulk structures from the local database: {len(bulk_structures)}'+'\n')
    log_write(f'Number of cluster structures from the local database: {len(cluster_structures)}'+'\n')
    return bulk_structures, cluster_structures

def get_structures_finetuning():
    # get reference structures, if any
    reference_structures, _ = get_reference_structures(EAH=False)
    low_energy_bulk_structures = []
    if os.path.exists(os.path.join(run_dir, 'local_db','training_data.json.gz')):
        with gzip.open(os.path.join(run_dir, 'local_db','training_data.json.gz'), 'rb') as fhandle:
            data_from_file = json.loads(fhandle.read())
        epas = []
        structures = []
        for a_data in data_from_file:
            if 'free' in a_data['bc']:
                continue
            pymatgen_structure = Structure.from_dict(a_data['structure'])
            nat = len(pymatgen_structure.sites)
            epas.append(float(a_data['energy'])/nat)
            structures.append(a_data['structure'])

#        low_energy_indices = np.argsort(epas)[:inputs['number_of_bulk_structures']]
        low_energy_indices = np.argsort(epas)[:200]

        for i in low_energy_indices:
            print(i)
            for j in low_energy_indices:
                if i == j:
                    continue
                s1 = Structure.from_dict(structures[i])
                s2 = Structure.from_dict(structures[j])
                if len(s1.sites) != len(s2.sites):
                    continue
                matcher = StructureMatcher(ltol = 0.2, stol = 0.3, angle_tol = 5, primitive_cell = False, scale = True, attempt_supercell = False, allow_subset = False) #, comparator = SpeciesComparator)
                if matcher.fit(s1, s2):
                    break
            else:
                low_energy_bulk_structures.append(s1.as_dict())
            continue
    log_write(f'Number of bulk structures from the local database: {len(low_energy_bulk_structures)}'+'\n')
    return sample(low_energy_bulk_structures, 20 - len(reference_structures)) + reference_structures

def add_structures_to_parent_group_finetuning():
    pg_step3_group = Group.collection.get(label='pg_step3')
    StructureData = DataFactory('structure')
    low_energy_bulk_structures = get_structures_finetuning()
    ref_structures = get_structures_references()

    for i, l_e_struct in enumerate(low_energy_bulk_structures + ref_structures):
        a_structure = Structure.from_dict(l_e_struct)
        nat = len(a_structure.sites)
        lestrct_node = StructureData(pymatgen=a_structure).store()
        lestrct_node.label = 'scheme3'
        lestrct_node.base.extras.set('job', 'scheme3-'+str(i+1)+'_'+str(nat)+'-atoms')
        pg_step3_group.add_nodes(lestrct_node)

def add_structures_to_parent_group_singlepoint():
    pg_singlepoint_group = Group.collection.get(label='pg_singlepoint')
    StructureData = DataFactory('structure')
    bulk_structures, molecule_structures = get_structures_singlepoint()
    for i, bulk_sp in enumerate(bulk_structures):
        a_structure = Structure.from_dict(bulk_sp)
        nat = len(a_structure.sites)
        bulk_sp_node = StructureData(pymatgen=a_structure).store()
        bulk_sp_node.label = 'bulk-sp'
        bulk_sp_node.base.extras.set('job', 'bulk-sp'+str(i+1)+'_'+str(nat)+'-atoms')
        pg_singlepoint_group.add_nodes(bulk_sp_node)
    if molecule_structures:
        boxed_molecule = []
        for i, molecule in enumerate(molecule_structures):
            boxed_molecule = []
            a_struct = Structure.from_dict(molecule)
            cart_coords = a_struct.cart_coords
            maxx = max(cart_coords[:,0:1])[0]
            minx = min(cart_coords[:,0:1])[0]
            maxy = max(cart_coords[:,1:2])[0]
            miny = min(cart_coords[:,1:2])[0]
            maxz = max(cart_coords[:,2:3])[0]
            minz = min(cart_coords[:,2:3])[0]
            a_cluster = maxx-minx+inputs['vacuum_length']
            b_cluster = maxy-miny+inputs['vacuum_length']
            c_cluster = maxz-minz+inputs['vacuum_length']
            if max(a_cluster, b_cluster, c_cluster) > 50: #max. box size 50 A
                continue
            molecule = Molecule(a_struct.species, cart_coords)
            boxed_molecule.append(molecule.get_boxed_structure(a_cluster,b_cluster,c_cluster))
        for i, a_boxed_molecule in enumerate(boxed_molecule):
            nat = len(a_boxed_molecule.sites)
            bmstrct_node = StructureData(pymatgen=a_boxed_molecule).store()
            bmstrct_node.label = 'moleculei-sp'
            bmstrct_node.base.extras.set('job', 'molecule-sp'+str(i)+'_'+str(nat)+'-atoms')
            pg_singlepoint_group.add_nodes(bmstrct_node)

def add_structures_to_parent_group():
    """ add structures to parent groups
    """
    pg_step3_group = Group.collection.get(label='pg_step3')
    StructureData = DataFactory('structure')

    n_struct_geopt, random_bulk_structures_dict = get_structures_from_nodes()
    bulk_structures = get_structures_references()

    cluster_list = []
    for a_key in random_bulk_structures_dict.keys():
        indices_schm1_geopt = []
        indices_schm2_geopt = []

        indices = list(range(len(random_bulk_structures_dict[a_key])))
        n_struct_schm1_geopt = int(0.5 * n_struct_geopt)
        n_struct_schm2_geopt = n_struct_geopt - n_struct_schm1_geopt
        if len(indices) >= n_struct_schm1_geopt:
            indices_schm1_geopt = sample(indices, n_struct_schm1_geopt)
        else:
            log_write(f'>>> WARNING: not enough structures with {str(a_key)} atoms for optimization with scheme 1 <<<'+'\n')
            sys.exit()

        for rem in indices_schm1_geopt:
            indices.remove(rem)

        if len(indices) >= n_struct_schm2_geopt:
            indices_schm2_geopt = sample(indices, n_struct_schm2_geopt)
        else:
            log_write(f' >>> WARNING: not enough structures with {str(a_key)} atoms for optimization with scheme 2 <<<'+'\n')
            sys.exit()

        for rem in indices_schm2_geopt:
            indices.remove(rem)

        for i, indx in enumerate(indices_schm1_geopt):
            a_structure = Structure.from_dict(random_bulk_structures_dict[a_key][indx])
            s1strct_node = StructureData(pymatgen=a_structure).store()
            s1strct_node.label = 'scheme1'
            s1strct_node.base.extras.set('job', 'scheme1-'+str(i+1)+'_'+str(a_key)+'-atoms')
            pg_step3_group.add_nodes(s1strct_node)
            if a_key in inputs['cluster_number_of_atoms'] and inputs['cluster_calculation'] :
                cluster_list.append(a_structure)
        for i, indx in enumerate(indices_schm2_geopt):
            a_structure = Structure.from_dict(random_bulk_structures_dict[a_key][indx])
            s2strct_node = StructureData(pymatgen=a_structure).store()
            s2strct_node.label = 'scheme2'
            s2strct_node.base.extras.set('job', 'scheme2-'+str(i+1)+'_'+str(a_key)+'-atoms')
            pg_step3_group.add_nodes(s2strct_node)
            if a_key in inputs['cluster_number_of_atoms'] and inputs['cluster_calculation'] :
                cluster_list.append(a_structure)
        if cluster_list:
            boxed_molecule = []
            for a_struct in cluster_list:
                cart_coords = a_struct.cart_coords
                maxx = max(cart_coords[:,0:1])[0]
                minx = min(cart_coords[:,0:1])[0]
                maxy = max(cart_coords[:,1:2])[0]
                miny = min(cart_coords[:,1:2])[0]
                maxz = max(cart_coords[:,2:3])[0]
                minz = min(cart_coords[:,2:3])[0]
                a_cluster = maxx-minx+inputs['vacuum_length']
                b_cluster = maxy-miny+inputs['vacuum_length']
                c_cluster = maxz-minz+inputs['vacuum_length']
                molecule = Molecule(a_struct.species, cart_coords)
                b_m = molecule.get_boxed_structure(a_cluster,b_cluster,c_cluster)
                if not is_structure_valid(b_m, False, False, False, False, True)[0]:
                    continue
                boxed_molecule.append(b_m)
            if len(boxed_molecule) > len(indices_schm1_geopt) :
                selected_boxed_molecule = sample(boxed_molecule, len(indices_schm1_geopt))
            else:
                selected_boxed_molecule = boxed_molecule
            for i, a_boxed_molecule in enumerate(selected_boxed_molecule):
                bmstrct_node = StructureData(pymatgen=a_boxed_molecule).store()
                bmstrct_node.label = 'cluster'
                bmstrct_node.base.extras.set('job', 'cluster-'+str(i)+'_'+str(a_key)+'-atoms')
                pg_step3_group.add_nodes(bmstrct_node)

    for i, ref_strct in enumerate(bulk_structures):
        a_structure = Structure.from_dict(ref_strct)
        nat = len(a_structure.sites)
        rfstrct_node = StructureData(pymatgen=a_structure).store()
        rfstrct_node.label = 'scheme3'
        rfstrct_node.base.extras.set('job', 'scheme3-'+str(i+1)+'_'+str(nat)+'-atoms')
        pg_step3_group.add_nodes(rfstrct_node)

def step_3():
    """ Step 3
    """
    log_write("---------------------------------------------------------------------------------------------------"+'\n')
    log_write('STEP 3'+'\n')
    log_write(f'start time: {get_time()}'+'\n')
    # check
    previous_run_exist_check()
    group_is_empty_check('wf_step3')
    # clear groups
    for a_group_label in ['pg_singlepoint', 'pg_step3', 'results_step3']:
        a_group, _ = Group.collection.get_or_create(a_group_label)
        a_group.clear()
    # add structures
    if 'scratch' in inputs['calculation_type']:
        if inputs['Chemical_formula']:
            add_structures_to_parent_group()
    elif 'singlepoint' in inputs['calculation_type']:
        add_structures_to_parent_group_singlepoint()
    elif 'finetuning' in inputs['calculation_type']:
        add_structures_to_parent_group_finetuning()
    else:
        log_write('>>> ERROR: calculation type is not provided  <<<'+'\n')
        sys.exit()
    # submit jobs
    if 'SIRIUS' in inputs['ab_initio_code'] or 'QS' in inputs['ab_initio_code']:
        from aiida_gw.codes.cp2k.cp2k_launch_calculations import CP2KSubmissionController, CP2KSPSubmissionController
        log_write(f'Ab-initio calculations with {inputs["ab_initio_code"]}'+'\n')
        if 'singlepoint' in inputs['calculation_type']:
            controller = CP2KSPSubmissionController(
                parent_group_label='pg_singlepoint',
                group_label='wf_step3',
                max_concurrent=job_script['geopt']['number_of_jobs'],
                QSorSIRIUS=inputs['ab_initio_code'])
        else:
            controller = CP2KSubmissionController(
                parent_group_label='pg_step3',
                group_label='wf_step3',
               max_concurrent=job_script['geopt']['number_of_jobs'],
               QSorSIRIUS=inputs['ab_initio_code'])
    else:
        log_write('>>> ERROR: no ab_initio code is provided <<<'+'\n')
        sys.exit()
    # wait until all jobs are done
    while controller.num_to_run > 0 or controller.num_active_slots > 0:
        if controller.num_to_run > 0:
            controller.submit_new_batch(dry_run=False)
        sleep(60)
    # store
    total_computing_time, submitted_jobs, finished_job = report('wf_step3')
    store_step3_results()
    log_write(f'submitted jobs: {submitted_jobs}, succesful jobs: {finished_job}'+'\n')
    log_write(f'total computing time: {round(total_computing_time, 2)} core-hours'+'\n')
    log_write('STEP 3 ended'+'\n')
    log_write(f'end time: {get_time()}'+'\n')
    log_write('Exporting data'+'\n')
    if 'single' in inputs['calculation_type']:
        dir_path = os.path.join(run_dir, 'singlepoint')
    elif inputs['Chemical_formula']:
        dir_path = os.path.join(run_dir, inputs['Chemical_formula'][0])
    elif inputs['Chemical_system']:
        dir_path = os.path.join(run_dir, inputs['Chemical_system'][0])
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    export_data(dir_path)
    log_write('Extracting data'+'\n')
    extract_data(dir_path)
    store_calculation_nodes()
    return steps_status[3]
