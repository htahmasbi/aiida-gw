# aiida-gw
![Python Version](https://img.shields.io/badge/python-≥%203.10-blue)
![AiiDA Version](https://img.shields.io/badge/AiiDA-≥%202.0-orange)

An AiiDA plugin to automate and manage Green's function (GW) workflows within the CP2K electronic structure package.
This project is a fork of [aiida-datagen](https://github.com/hmhoseini/aiida-datagen), refactored to focus on GW calculations using CP2K.

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

The directory structure of aiida-gw is as follows:

aiida-gw <br>
&nbsp;&nbsp;&nbsp;&nbsp;├── aiida_gw <br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── codes <br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;├── cp2k <br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── cp2k_files <br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;├── utils.py <br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;└── pxtl.py <br>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── workflows <br>
&nbsp;&nbsp;&nbsp;&nbsp;├── run_dir <br>
&nbsp;&nbsp;&nbsp;&nbsp;└── utils <br>

The default output directory is *run_dir* but **datagen.py** can be executed in any directory that contains the following files: <br>
&nbsp;&nbsp; config.yaml <br>
&nbsp;&nbsp; input.yaml <br>
&nbsp;&nbsp; restart.yaml <br>
It is necessary to modify these three files before running aiida-gw.

The following values should be specified in the **config.yaml** file:

|Values                |Description|
|-|-|
|api_key|Materials Project API key|
|DFT_code_string|Code label for DFT calculations. Available codes and their identifiers can be listed with ```verdi code list```|
|The following parameters should be specified for all jobs:|
|number_of_jobs|The maximum number of jobs that will be submitted for a specific job type.|
|nodes|Number of nodes to be allocated for the job. See [slurm manual](https://slurm.schedmd.com/sbatch.html#OPT_nodes)|
|ntasks|Number of tasks per node. See [slurm manual](https://slurm.schedmd.com/sbatch.html#OPT_ntasks-per-node)|
|ncpu|Number of processors per task. See [slurm manual](https://slurm.schedmd.com/sbatch.html#OPT_cpus-per-task)|
|time|Maximum time for a job (in seconds)|
|exclusive|Whether the job can share nodes with other running jobs. See [slurm manual](https://slurm.schedmd.com/sbatch.html#OPT_exclusive)|

The data that should be provided in **input.yaml** is as follows:

|Key|Description|
|-|-|
|Chemical_formula|A list of chemical formula(s)|
|bulk_number_of_atoms|Specifies number of atoms in bulk structures|
|number_of_bulk_structures|Specifies number of structures sent for ab-initio calculations|
|cluster_calculation|If cluster structures should be included in the training|
|cluster_number_of_atoms|Number of atoms on cluster structures. Should be a subset of bulk_number_of_atoms|
|vacuum_length|Minimum length of vacuum for each supercell containing a cluster|
|random_structure_generator|Structure generator: PyXtal (default)|
|min_distance_prefactor|Prefactor to control minimum distance between atoms in initial random structures|
|ab_initio_code|SIRIUS_CP2K or CP2K_QS|
|user_specified_CP2K_files|If True, then codes/cp2k/cp2k_files folder should be copied into the run directory. The user can modify CP2K keywords (protocol files and/or pseudopotentials).|

<br>
The step from which aiida-gw (re)starts is specified in restart.yaml. aiida-gw keeps track of its steps. If a failure occurs and aiida-gw cannot advance, the user can restart aiida-gw from the last successfully accomplished step.

The following parameters can be specified in **restart.yaml**:

|Key                |Description|
|-|-|
|re-start_from_step |Specifies the step the script will (re)start.<br>1: start aiida-gw <br>2: random bulk structure generation <br>3: ab-initio calculations|
|stop_after_step|The script stops after the end of this step. -1 for non-stop run.|

When the above-mentioned files are ready, the script can be executed by running **datagen.py** command.

## License ##
MIT

## Contact ##
h.tahmasb@gmail.com
