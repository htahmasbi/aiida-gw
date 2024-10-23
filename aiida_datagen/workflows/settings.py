import os
import yaml
from aiida.orm import Group
from aiida.manage.configuration import load_profile

load_profile()

this_directory = os.path.abspath(os.path.dirname(__file__))
datagen_directory = os.path.split(this_directory)[0]

run_folder_group = Group.collection.get(label='run_folder')
run_dir = run_folder_group.nodes[0].value

additional_structures_dir = os.path.join(run_dir,'additional_structures')

output_dir = os.path.join(run_dir, 'output')
log_file = os.path.join(output_dir,'datagen.log')

with open(os.path.join(run_dir,'input.yaml'), 'r', encoding='utf8') as fhandle:
    inputs = yaml.safe_load(fhandle)

with open(os.path.join(run_dir, 'config.yaml'), 'r', encoding='utf8') as fhandle:
    configs = yaml.safe_load(fhandle)

job_script = configs['job_script']
api_key = configs['MP_api']['api_key']

with open(os.path.join(run_dir,'restart.yaml'), 'r', encoding='utf8') as fhandle:
    restart = yaml.safe_load(fhandle)

steps_status = 5 * [True]
if restart['stop_after_step'] >= 0:
    steps_status[restart['stop_after_step']] = False

with open(os.path.join(this_directory, 'groups.yaml'), 'r', encoding='utf8') as fhandle:
    groups = yaml.safe_load(fhandle)

FLAME_input_files_path = os.path.join(run_dir,'flame_files')\
                        if inputs['user_specified_FLAME_files']\
                        else os.path.join(datagen_directory,'codes/flame','flame_files')
CP2K_input_files_path = os.path.join(run_dir,'cp2k_files')\
                        if inputs['user_specified_CP2K_files']\
                        else os.path.join(datagen_directory,'codes/cp2k','cp2k_files')
VASP_input_files_path = os.path.join(run_dir,'vasp_files')\
                        if inputs['user_specified_VASP_files']\
                        else os.path.join(datagen_directory,'codes/vasp','vasp_files')
