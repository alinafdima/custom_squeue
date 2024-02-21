# --------------------------------------------------------
# File: count_gpus.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Wed Jan 04 2023
#
# Script for displaying the total number of GPUs currently 
# in use on a slurm GPU cluster, as well as job information.
# --------------------------------------------------------


import argparse
import re
import subprocess
from datetime import datetime
from functools import reduce
from pprint import pprint


display_columns_running = [
    'JobId', 'JobName', 'UserId',
    'GPU ID', 
    'GPUs', 'CPUs',
    'Runtime', 
    'Remaining Time',
    'Mem',
]

display_columns_pending = [
    'JobId', 'JobName', 'UserId',
    'GPUs', 'CPUs',
]

display_columns_other = [
    'JobId', 'JobName', 'UserId',
    'Status', 'Elapsed Time', 'Run Time',
]


def parse_jobs():
    def parse_job_attributes(job):
        attributes = [tuple(x.split('=')) for x in job.split(' ') if x != '']
        attributes = dict([(x[0], '='.join(x[1:])) for x in attributes])
        return attributes
    
    output = subprocess.check_output(['scontrol','show', 'job', '-d'])
    output = output.decode('utf-8')
    jobs_raw = [x.replace('\n', ' ') for x in output.split('\n\n')]
    jobs = [parse_job_attributes(job) for job in jobs_raw if job != '']
    return jobs


def count_gpus_in_use(job):
    if job['JobState'] != 'RUNNING':
        return 0
    matches = re.match('.*gres/gpu=([0-9]*).*', job['TRES'])
    if matches is not None:
        return int(matches.group(1))
    else:
        return 0


def get_gpus_on_node(jobs, node_name):
    jobwise_counts = [count_gpus_in_use(job) for job in jobs 
                      if job['JobState'] == 'RUNNING' 
                      and job['Nodes'] == node_name]
    if len(jobwise_counts) == 0:
        return 0
    else:
        return reduce(lambda x, y: x + y, jobwise_counts)


def parse_gres(job):
    gpus = count_gpus_in_use(job)
    node = job['Nodes']
    if gpus > 0:
        gres = job['GRES']
        matches = re.match('.*(IDX:.*)\\)', job['GRES'])
        if matches is None:
            gres = f'UNKNOWN ({node})'
        else:
            # FIXME: Won't work for more than 1 GPU
            idx = matches.group(1)[4:]
            gres = f'{node[:4]} {idx}'     
    else:
        gres = ''
    return gres


def parse_user_id(job):
    # Remove the (uid) bit in the user_id
    user_id = re.match('(.*)\\(.*\\)', job['UserId']).group(1)
    partition = job['Partition'][0].upper()
    user_id_partition = f'{partition} {user_id}'
    return user_id, user_id_partition


def is_job_recent(reference_time, minutes=60, days=0):
    """
    Check if a job is recent, 
    i.e. if it started less than `minutes` minutes ago.
    """
    reference_time = datetime.strptime(reference_time, '%Y-%m-%dT%H:%M:%S')
    delta = datetime.now() - reference_time
    if delta.days > days:
        return False
    return delta.seconds < minutes * 60


def strfdelta(tdelta, fmt):
    """
    Format a timedelta object as a string.
    """
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    return fmt.format(**d)


def format_remaining_time(time_point):
    """
    Compute the delta between a future time point and the current time
    and format it accordingly.
    """    
    try:
        time_point = datetime.strptime(time_point, '%Y-%m-%dT%H:%M:%S') 
        rt_formatted = strfdelta(time_point - datetime.now(), 
                                    "{D}d:{H:02}h:{M:02}m:{S:02}s")
    except ValueError:
        rt_formatted = 'UNKNOWN'
    return rt_formatted


def format_elapsed_time(time_point):
    """
    Compute the delta between the current time and a past time point
    and format it accordingly.
    """
    try:
        time_point = datetime.strptime(time_point, '%Y-%m-%dT%H:%M:%S') 
        et_formatted = strfdelta(datetime.now() - time_point, 
                                    "{D}d:{H:02}h:{M:02}m:{S:02}s")
    except ValueError:
        et_formatted = 'UNKNOWN'
    return et_formatted
    

def print_total_gpus(jobs, nodes):
    """
    Print the total number of GPUs in use at the moment on the specified nodes.
    """
    total = sum(nodes.values())

    gpus_dict = {node: get_gpus_on_node(jobs, node) for node in nodes}
    gpus_used = reduce(lambda x, y: x + y, list(gpus_dict.values()))

    print_strings = [f'{node}: {gpus_dict[node]}/{available}' for 
                     node, available in nodes.items()]
    details = ' | '.join(print_strings)
    print(f'GPUs used: {gpus_used}/{total} ---> {details}')


def get_running_job_display_data(jobs):
    job_display_data = []
    for job in jobs:
        if job['JobState'] != 'RUNNING':
            continue
        user_id, user_id_partition = parse_user_id(job)
        gpus = count_gpus_in_use(job)

        job_info = {
            'JobId': job['JobId'], 
            'JobName': job['JobName'],
            'UserId': user_id_partition,
            'UserId2': user_id,
            'GPU ID': parse_gres(job),
            'GPUs': str(gpus),
            'CPUs': job['NumCPUs'],
            'Runtime': job['RunTime'], 
            'Time Limit': job['TimeLimit'], 
            'Start Time': job['StartTime'], 
            'End Time': job['EndTime'],
            'Remaining Time': format_remaining_time(job['EndTime']),
            'Mem': f"{int(job['Mem']) // 1000} GB", 
            'Partition': job['Partition'][0].upper(),
        }
        job_display_data.append(job_info)

    return job_display_data


def get_pending_job_display_data(jobs):
    job_display_data = []
    for job in jobs:
        if job['JobState'] != 'PENDING':
            continue
        user_id, user_id_partition = parse_user_id(job)
        job_info = {
            'JobId': job['JobId'], 
            'JobName': job['JobName'],
            'UserId': user_id_partition,
            'UserId2': user_id,
            'GPUs': job['NumNodes'],
            'CPUs': job['NumCPUs'],
            'Runtime': job['RunTime'], 
            'Time Limit': job['TimeLimit'], 
            'Start Time': job['StartTime'], 
            'End Time': job['EndTime'],
            'Remaining Time': format_remaining_time(job['EndTime']),
            'Partition': job['Partition'][0].upper(),
        }
        job_display_data.append(job_info)

    return job_display_data


def other_job_display_data(jobs, minutes=60):
    job_display_data = []
    for job in jobs:
        if job['JobState'] in {'RUNNING', 'PENDING'}:
            continue
        if not is_job_recent(job['EndTime'], minutes=minutes):
            continue
        user_id, user_id_partition = parse_user_id(job)
        job_info = {
            'JobId': job['JobId'], 
            'JobName': job['JobName'],
            'UserId': user_id_partition,
            'UserId2': user_id,
            'Partition': job['Partition'][0].upper(),
            'Status': job['JobState'],
            'Elapsed Time': format_elapsed_time(job['EndTime']),
            'Run Time': str(job['RunTime']),
        }
        job_display_data.append(job_info)

    return job_display_data


def print_display_data(job_display_data, display_columns, print_string,
                       spacer=' | ', sort_key='UserId2'):
    
    if len(job_display_data) == 0:
        print(f'\nNo {print_string} to display.')
        return
    
    print()
    print(f'{print_string}:')

    # Determine the number of characters per column
    formatting = {key: max([len(key)] + 
                           [len(entry[key]) for entry in job_display_data]) 
                 for key in display_columns}
    
    # Print header
    print(spacer.join([key.center(formatting[key]) for key in display_columns]))
    
    # Sort by key
    job_display_data = sorted(job_display_data, key=lambda x: x[sort_key])

    # Print job info
    for job in job_display_data:
        print(spacer.join([str(job[key]).ljust(formatting[key]) 
                           for key in display_columns]))


def parse_nodes(raw_nodes):
    nodes = {}
    for node in raw_nodes:
        node_name, num_gpus = node.split(':')
        nodes[node_name] = int(num_gpus)
    return nodes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=str, nargs='+', default=[])
    args = parser.parse_args()
    nodes = parse_nodes(args.nodes)

    jobs = parse_jobs()

    print_total_gpus(jobs, nodes)

    print_display_data(get_running_job_display_data(jobs), 
                       display_columns_running, 'Running jobs')
    
    print_display_data(get_pending_job_display_data(jobs), 
                       display_columns_pending, 'Pending jobs')

    all_states = set([job['JobState'] for job in jobs])
    other_states = all_states - {'RUNNING', 'PENDING'}
    other_jobs = [job for job in jobs if job['JobState'] in other_states]

    print_display_data(other_job_display_data(other_jobs, minutes=30), 
                        display_columns_other, 'Other jobs')