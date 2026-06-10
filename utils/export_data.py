import os
import yaml
import argparse
from aiida.orm import Dict, Group, load_node
from aiida_gw.workflows.settings import run_dir, datagen_directory
from aiida.manage.configuration import load_profile

load_profile()

def add_input_node():
    pks = []
    with open(os.path.join(run_dir,'input.yaml'), 'r', encoding='utf8') as fhandle:
        inputs_data = yaml.safe_load(fhandle)
    a_node = Dict(inputs_data)
    a_node.label = 'inputs'
    store = a_node.store()
    pks.append(store.pk)
    return pks, inputs_data

def add_author_data():
    with open(os.path.join(run_dir,'author_data.yaml'), 'r', encoding='utf8') as fhandle:
        author_data = yaml.safe_load(fhandle)
    a_node = Dict(author_data)
    a_node.label = 'author_data'
    store = a_node.store()
    return store.pk

def add_protocol(inputs):
    CP2K_input_files_path = os.path.join(run_dir,'cp2k_files')\
                            if inputs['user_specified_CP2K_files']\
                            else os.path.join(datagen_directory,'codes/cp2k','cp2k_files')
    if 'SIRIUS' in inputs['ab_initio_code']:
        with open(os.path.join(CP2K_input_files_path,'protocol_SIRIUS.yml'), 'r', encoding='utf8') as fhandle:
            cp2k_protocol = yaml.safe_load(fhandle)
        single_point_protocol = cp2k_protocol['single_point']
        single_point_protocol['FORCE_EVAL'].pop('SUBSYS')
        a_node = Dict(single_point_protocol)
        a_node.label = 'protocol'
        store = a_node.store()
        return store.pk
    if 'QS' in inputs['ab_initio_code']:
        with open(os.path.join(CP2K_input_files_path,'protocol_QS.yml'), 'r', encoding='utf8') as fhandle:
            cp2k_protocol = yaml.safe_load(fhandle)
        single_point_protocol = cp2k_protocol['single_point']    
        a_node = Dict(single_point_protocol)
        a_node.label = 'protocol'
        store = a_node.store()
        return store.pk

def add_known_structures():
    pks = []
    known_structures_group = Group.collection.get(label='known_structures')
    for a_node in known_structures_group.nodes:
        pks.append(a_node.pk)
    return pks

def add_calculation_nodes():
    pks = []
    calculation_nodes_group = Group.collection.get(label='calculation_nodes')
    for a_node in calculation_nodes_group.nodes:
        pks.extend(a_node.get_list())
    return pks

def add_nodes(pk_list):
    tmp_group, _ = Group.collection.get_or_create('tmp_group')
    tmp_group.clear()
    for a_pk in pk_list:
        try:
            a_node = load_node(a_pk)
        except:
            continue
        tmp_group.add_nodes(a_node)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='export_data.py',
                    description='Exprot calculation node to an AiiDA archive')

    parser.add_argument('filename', help='Output file name *.aiida')
    args = parser.parse_args()


    pks, inputs_data = add_input_node()
    pks.append(add_author_data())
    pks.append(add_protocol(inputs_data))
    pks.append(add_known_structures())
    pks.append(add_calculation_nodes())
    add_nodes(pks)
    os.system(f"verdi archive create --no-call-calc-backward --no-call-work-backward --no-create-backward {args.filename} --groups tmp_group")
    with open('node_pks.dat', 'w', encoding='utf-8') as fhandle:
        for a_pk in pks:
            fhandle.write(f'{a_pk}'+'\n')

    tmp_group, _ = Group.collection.get_or_create('tmp_group')
    Group.collection.delete(tmp_group.pk)


