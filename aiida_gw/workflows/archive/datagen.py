#!/usr/bin/env python
import os
import sys
from aiida.orm import Str, Group
from aiida.manage.configuration import load_profile

load_profile()

__author__ = "Hossein Mirhosseini"
__copyright__ = ""
__maintainer__ = "Hossein Mirhosseini"
__email__ = "h.mirhosseini@hzdr.de"

if __name__ == "__main__":
    run_dir = os.getcwd()
    if os.path.exists(os.path.join(run_dir,'input.yaml')) and\
       os.path.exists(os.path.join(run_dir,'restart.yaml')) and\
       os.path.exists(os.path.join(run_dir,'config.yaml')):
        group, _ = Group.collection.get_or_create('run_folder')
        group.clear()
        run_dir_node = Str(run_dir).store()
        group.add_nodes(run_dir_node)
        from workflows.main import main
        main()
    else:
        print('>>> ERROR: input.yaml, restart.yaml, and config.yaml are needed  <<<')
        sys.exit()
