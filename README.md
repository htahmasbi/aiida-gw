# aiida-gw
![Python Version](https://img.shields.io/badge/python-≥%203.8-blue)
![AiiDA Version](https://img.shields.io/badge/AiiDA-≥%202.0-orange)

An AiiDA plugin to automate and manage Green's function (GW) workflows within the CP2K electronic structure package.

## Dependencies ##
[aiida-core](https://github.com/aiidateam/aiida-core), [aiida-submission-controller](https://github.com/aiidateam/aiida-submission-controller), and [aiida-cp2k](https://github.com/aiidateam/aiida-cp2k) should be installed.


## Installation ##

To install the aiida-gw package directly from the cloned repository:

```
git clone https://github.com/htahmasbi/aiida-gw.git
   
cd aiida-gw
    
pip install -e .
```

The aiida-gw directory should be added to PYTHONPATH.

## Usage ##

The directory structure of AiiDA-gw is as follows:

aiida-datagen <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── aiida_datagen <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── codes <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── cp2k <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── cp2k_files <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── flame <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── flame_files <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── flame_functions <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── vasp <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──   vasp_files <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──  workflows <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── examples <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── run_dir <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── utils <br>


The default output directory is *run_dir* but **datagen.py** can be executed in any directory that contains the following files: <br>
&nbsp;&nbsp; config.yaml <br>
&nbsp;&nbsp; input.yaml <br>
&nbsp;&nbsp; restart.yaml <br>
It is necessary to modify these three files before running aiida-datagen.

The following values should be specified in the **config.py** file:

|Values                |Description|
|-|-|
|api_ky|Materials Project API key|
|DFT_code_string|Code label for DFT calculations. Available codes and their identifiers can be listed with ``` verdi code list ```|
|FLAME_code_string |Code label for FLAME calculations. Available codes and their identifiers can be listed with ```verdi code list```|
|The following parameters should be specified for all jobs:|
|number_of_jobs|The maximum number of jobs that will be submitted for an specifc job type.|
|nodes|Number of nodes to be allocate for the job. See [slurm manual](https://slurm.schedmd.com/sbatch.html#OPT_nodes)|
|ntasks|Number of tasks per node. See [slurm manual](https://slurm.schedmd.com/sbatch.html#OPT_ntasks-per-node)|
|ncpu|Number of processors per task. See [slurm manual](https://slurm.schedmd.com/sbatch.html#OPT_cpus-per-task)|
|time|Maximum time for a job (in seconds)|
|exclusive|Whether the job can share nodes with other running jobs. See [slurm manual](https://slurm.schedmd.com/sbatch.html#OPT_exclusive)|



The data that should be provided in **input.yaml** is as follows:

|Key|Description|
|-|-|
|chemical_formula|A list of chemical formula(s)|
|bulk_number_of_atoms|Specifies number of atoms in bulk structures|
|max_number_of_bulk_structures|Specifies number of structures sent for ab-initio calculations|
|cluster_calculation|If cluster structures should be included in the training|
|cluster_number_of_atoms|Number of atoms on cluster structures. Should be a subset of bulk_number_of_atoms|
|vacuum_length|Minimum length of vacuum for each supercell containing a cluster|
|from_local_db|If True, atomic structures will be retrieved from *run_dir/local_db/known_bulk_structures.json* (a list of dict representation pymatgen Structures)|
|energy_window|The maximum value of energy for training data (eV).|
|max_force| The maximum value of force on atoms (eV/A)|
|method|behler|
|number_of_nodes|Number of nodes in the hidden layer of the NN for each cycle of training.|
|number_of_epoch|Number of epoch for each cycle of training.|
|minimahopping_time|Minimum and maximum time for minima hopping jobs (in hours)|
|minhocao_steps|Maximum number of minhocao steps|
|bulk_minhocao|Maximum number of minhocao jobs (minima hopping for bulk structures with variable cell)|
|minhopp_steps|Maximum number of minhopp steps|
|bulk_minhopp|Maximum number of minhopp jobs for bulk structures (minima hopping for bulk structures with fixed cell)|
|cluster_minhopp|Maximum number of minhopp jobs for clusters|
|selecting_method|Either QBC (querry by committee) or FDV (flame diversity check)|
|dtol_prefactor|(only for FDV) Prefactor for structure diversity check. The larger the value is, the more structures are considered similar (removed from the list).|
|prefactor_cluster|(only for FDV) A prefactor for dtol_prefactor to be employed for clusters|
|ab_initio_code|SIRIUS_CP2K, VASP, or CP2K_QS|
|user_specified_CP2K_files|If True, then codes/cp2k/cp2k_files folder should be copied into the run directory. The user can modify CP2K keywords (protocol files and/or pseudopotentials).|
|user_specified_FLAME_file|If True, then codes/flame/flame_files folder should be copied into the run directory. The user can modify FLAME keywords (protocol file).|
|user_specified_VASP_file |If True, then codes/vasp/vasp_files folder should be copied into the run directory. The user can modify VASP keywords (protocol file and/or potential_mapping).|

<br>
The step from which aiida-datagen (re)starts is specified in restart.yaml. aiida-datagen keeps track of its steps. If a failure occurs and aiida-datagen cannot advance, the user can restart aiida-datagen from the last successfully accomplished step. It is noted that users can always restart aiida-datagen from a previous step.

The following parameters can be specified in **restart.yaml**:

|Key                |Description|
|-|-|
|re-start_from_step |Specifies the step the script will (re)start.<br>-1: run unfinished jobs <br>1: start aiida-datagen <br>2: random bulk structure generation <br>3: initial ab-initio calculations <br>4: FLAME trainin loop|
|stop_after_step|The script stops after the end of this step. -1 for non-stop run.|
|training_loop_start|Specifies the cycle number and cycle name for the training cycle. The cycle names are as follows:<br>- train (training the NN) <br>- minimahopping (minima hopping for bulk structures and clusters)<br>- divcheck (structure diversity check) <br>- SP_calculations (ab-initio single-point calculations)|
|training_loop_stop|Specifies the cycle number and cycle name to stop the training cycle.|

When the above-mentioned files are ready, the script can be executed by running **datagen.py** command.


## Citation ##

## How to contribute ##

## License ##
MIT
## Contact ##
h.tahmasb@gmail.com
