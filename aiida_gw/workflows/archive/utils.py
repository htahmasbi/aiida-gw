import os
import json
from random import uniform
from datetime import datetime
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester
from aiida.orm import Group, Dict
from aiida_gw.workflows.settings import inputs, output_dir, api_key

def get_pertured_failed_structures(cycle_number):
    """ Perturb fialed crystal structures.
    """
    failed_b_structures = []
    failed_c_structures = []

    fpath = os.path.join(output_dir,cycle_number,'minimahopping','failed_bulk.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            failed_b_structures.extend(json.loads(fhandle.read()))
    fpath = os.path.join(output_dir,cycle_number,'minimahopping','failed_structures.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            failed_b_structures.extend(json.loads(fhandle.read()))
    fpath = os.path.join(output_dir,cycle_number,'minimahopping','failed_cluster.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            failed_c_structures.extend(json.loads(fhandle.read()))

    perturbed_b_structures = []
    perturbed_c_structures = []

    for a_b_struct in failed_b_structures:
        a_b_structure = Structure.from_dict(a_b_struct)
        a_b_s = a_b_structure.copy()
        a_b_s.perturb(uniform(0.02, 0.06))
        if is_structure_valid(a_b_s, False, 0.80, False, False, False)[0]:
            perturbed_b_structures.append(a_b_s)
    for a_c_struct in failed_c_structures:
        a_c_structure = Structure.from_dict(a_c_struct)
        a_c_s = a_c_structure.copy()
        a_c_s.perturb(uniform(0.02, 0.06))
        if is_structure_valid(a_c_s, False, 0.80, False, False, True)[0]:
            perturbed_c_structures.append(a_c_s)
    return perturbed_b_structures, perturbed_c_structures

def get_rejected_structures(cycle_number):
    b_structures = []
    c_structures = []
    rejected_bulk_structures = []
    rejected_cluster_structures = []

    fpath = os.path.join(output_dir,cycle_number,'minimahopping','rejected_bulks.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            b_structures.extend(json.loads(fhandle.read()))

    fpath = os.path.join(output_dir,cycle_number,'minimahopping','rejected_structuress.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            b_structures.extend(json.loads(fhandle.read()))

    fpath = os.path.join(output_dir,cycle_number,'minimahopping','rejected_clusters.json')
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as fhandle:
            c_structures = json.loads(fhandle.read())

    for a_structure in b_structures:
        rejected_bulk_structures.append(Structure.from_dict(a_structure))
    for a_structure in c_structures:
        rejected_cluster_structures.append(Structure.from_dict(a_structure))
    return rejected_bulk_structures, rejected_cluster_structures 

def get_element_list():
    """ Retruns the list of elements.
    """
    element_list = []
    composition_list = inputs['Chemical_formula']
    for a_composition in composition_list:
        for elmnt in Composition(a_composition).elements:
            if str(elmnt) not in element_list:
                element_list.append(str(elmnt))
    return element_list

def get_structures_from_mpdb():
    """ Store structures available in the MP databases
    """
    l = 0
    known_structures_group = Group.collection.get(label='known_structures')
    composition_list = inputs['Chemical_formula']
    mpr= MPRester(api_key)
    for a_composition in composition_list:
        results = mpr.materials.summary.search(formula=a_composition, fields=["structure", "energy_above_hull"])
        l += len(results)
        structures_eah = []
        for a_result in results:
            sga = SpacegroupAnalyzer(a_result.structure)
            conventional_structure = sga.get_conventional_standard_structure()
            structures_eah.append([conventional_structure.as_dict(), a_result.energy_above_hull])
        structures_eah.sort(key=lambda x: x[1])
        a_node = Dict({a_composition: structures_eah}).store()
        a_node.label = 'structure_from_mpdb'
        known_structures_group.add_nodes(a_node)
    return l

def get_reference_structures(EAH):
    """ EAH True: return up to 10 structures with EAH < 0.30 eV/atom and nat < 80
        EAE False: retrun structures with allowed nat for ab initio calculations
    """
    composition_list = inputs['Chemical_formula']
    reference_structures = []
    vpas = []
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'structure_from_mpdb' in a_node.label:
            tmp_dict = a_node.get_dict()
            for a_key in tmp_dict.keys():
                if EAH:
                    for a_structure_eah in tmp_dict[a_key]:
                        if a_structure_eah[1] > 0.30:
                            continue
                        pymatgen_structure = Structure.from_dict(a_structure_eah[0])
                        if len(pymatgen_structure.sites) < 80:
                            if is_structure_valid(pymatgen_structure, False, False, True, False, False)[0]:
                                reference_structures.append(a_structure_eah[0])
                                vpas.append(pymatgen_structure.volume/len(pymatgen_structure.sites))
                        else:
                            primitive_structure = pymatgen_structure.get_primitive_structure()
                            if len(primitive_structure.sites) != len(pymatgen_structure.sites) and\
                               len(primitive_structure.sites) < 80 and\
                               is_structure_valid(primitive_structure, False, False, True, False, False)[0]:
                                reference_structures.append(primitive_structure.as_dict())
                                vpas.append(primitive_structure.volume/len(primitive_structure.sites))
                    return reference_structures[:10], vpas

                a_n_a = get_allowed_n_atom_for_compositions(composition_list)
                for a_structure_eah in tmp_dict[a_key]:
                    pymatgen_structure = Structure.from_dict(a_structure_eah[0])
                    if len(pymatgen_structure.sites) in a_n_a and\
                       is_structure_valid(pymatgen_structure, False, False, True, False, False)[0]:
                        reference_structures.append(a_structure_eah[0])
                        continue
                    primitive_structure = pymatgen_structure.get_primitive_structure()
                    if len(primitive_structure.sites) in a_n_a and\
                       len(primitive_structure.sites) != len(pymatgen_structure.sites) and\
                       is_structure_valid(primitive_structure, False, False, True, False, False)[0]:
                        reference_structures.append(primitive_structure.as_dict())
                        continue
                return reference_structures, vpas

def get_allowed_n_atom_for_compositions(composition_list):
    """ Returns allowed number of atoms for given crystal structures.
    """
    allowed_n_atom_bulk = []
    for a_n_a in inputs['bulk_number_of_atoms']:
        for a_comp in composition_list:
            n_elmnt = []
            for n in Composition(a_comp).values():
                n_elmnt.append(int(n))
            n_element = n_elmnt
            if a_n_a % sum(n_element) == 0  and a_n_a not in allowed_n_atom_bulk:
                allowed_n_atom_bulk.append(a_n_a)
    return allowed_n_atom_bulk

def is_structure_valid(structure, ref_structure, min_d_prefactor, check_angles, check_vpa, is_cluster):
    """ Check if a crystal structure is a valid
    """
    max_d = None
    vpas = []
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        if 'vpas' in a_node.label:
            vpas = a_node.get_list()
        if 'dimers' in a_node.label:
            dimers = a_node.get_dict()
            max_d = max(list(dimers.values()))*2
    if not max_d:
        max_d = 5
    if check_angles and not is_cluster:
        angles = structure.lattice.angles
        if angles[0] > 145 or angles[0] < 45 or\
           angles[1] > 145 or angles[1] < 45 or\
           angles[2] > 145 or angles[2] < 45:
            return [False, 'angle']
    if check_vpa and not is_cluster:
        vpa = structure.volume/len(structure.sites)
        if vpa < vpas[0] * check_vpa[0] or vpa > vpas[1] * check_vpa[1]:
            return [False, 'vpa']
    if ref_structure:
        ref_d_matrix = ref_structure.distance_matrix
        ref_min_d = min(ref_d_matrix[ref_d_matrix != 0])
    try:
        d_matrix = structure.distance_matrix
    except:
        return [False, 'd_matrix']
    for isites in range(len(structure.sites)):
        if len(structure.get_neighbors(structure[isites], max_d)) < 2:
            return [False, 'far']
        if min_d_prefactor:
            for jsites in range(len(structure.sites)):
                if jsites == isites:
                    break
                min_d = dimers[structure[isites].species_string+'-'+structure[jsites].species_string] * min_d_prefactor
                if ref_structure:
                    min_d = min(min_d, ref_min_d * min_d_prefactor)
                if d_matrix[isites,jsites] < min_d:
                    return [False, 'close']
    if is_cluster:
        cart_coords = structure.cart_coords
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
            return [False, 'small box']
    return [True, None]

def get_time():
    return f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def store_calculation_nodes():
    calculation_nodes_group = Group.collection.get(label='calculation_nodes')
    calculation_nodes =[]
    for a_node in calculation_nodes_group.nodes:
        calculation_nodes.extend(a_node.get_list())

    known_structures_group = Group.collection.get(label='known_structures')
    known_structures_nodes = []
    for a_node in known_structures_group.nodes:
        known_structures_nodes.append(a_node.pk)
#    with open(os.path.join(output_dir, 'calculation_nodes.dat'), 'w', encoding='utf-8') as fhandle:
#        for a_node in known_structures_nodes+calculation_nodes:
#            fhandle.write(f'{a_node}'+'\n')
