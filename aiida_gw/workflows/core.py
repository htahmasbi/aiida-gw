import sys
from aiida_gw.workflows.settings import groups, log_file
from aiida.orm import Group, QueryBuilder, WorkChainNode, CalcJobNode

def log_write(txt):
    """ Write to log file
    """
    with open(log_file, 'a', encoding='utf8') as fhandle:
        fhandle.write(txt)

def previous_run_exist_check():
    """ If unfinished job exist
    """
    active_groups = []
    for a_group_label in groups['workflows_group_list']:
        try:
            a_group = Group.collection.get(label=a_group_label)
        except:
            continue
        for a_node in a_group.nodes:
            if not a_node.is_terminated:
                active_groups.append(a_group)
                break
    if len(active_groups) != 0:
        for a_a_g in active_groups:
            log_write(f'>>> ERROR: unfinished workflow(s) in group {a_a_g.label} (pk: {a_a_g.pk}) <<<'+'\n')
        sys.exit()

def group_is_empty_check(group_label):
    """ If a group is empty
    """
    try:
        group = Group.get(label=group_label)
    except:
        return True
    if not group.is_empty:
        log_write(f'>>> ERROR: group {group.label} (pk: {group.pk}) is not empty <<<'+'\n')
        sys.exit()

def report(group_label):
    """ Report
    """
    builder = QueryBuilder()
    builder.append(Group, filters={'label': group_label}, tag='wf_group')
    builder.append(WorkChainNode, with_group='wf_group', tag='wch_nodes')
    if group_label in ['wf_train', 'wf_minimahopping', 'wf_qbc']:
        builder.append(CalcJobNode, with_incoming='wch_nodes', tag='calcjob_nodes', project='attributes')
    else:
        builder.append(WorkChainNode, with_incoming='wch_nodes', tag='wf_nodes')
        builder.append(CalcJobNode, with_incoming='wf_nodes', tag= 'calcjob_nodes', project='attributes')
    submitted_jobs = 0
    finished_job = 0
    total_computing_time = 0
    for a_dict in builder.iterdict():
        submitted_jobs+=1
        try:
            exit_status = a_dict['calcjob_nodes']['attributes']['exit_status']
        except:
            exit_status = 300
        if exit_status == 0:
            finished_job += 1
        try:

            total_computing_time += (a_dict['calcjob_nodes']['attributes']['last_job_info']['wallclock_time_seconds']/3600) *\
                                     a_dict['calcjob_nodes']['attributes']['resources']['num_machines'] *\
                                     a_dict['calcjob_nodes']['attributes']['resources']['num_cores_per_mpiproc'] *\
                                     a_dict['calcjob_nodes']['attributes']['resources']['num_mpiprocs_per_machine']
        except:
            pass
    return total_computing_time, submitted_jobs, finished_job
