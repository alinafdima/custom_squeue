# ------------------------------------------------------------------------------
# File: jobs.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Mon Aug 12 2024
#
# Parses scontrol show job -d and returns a list of jobs with their attributes.
#
# ------------------------------------------------------------------------------
# %%

import re
from datetime import datetime


qos_order_dict = {
    'phd|d': 1,
    'phd|n': 2,
    'msc|d': 3,
    'msc|n': 4,
    'other': 5,
}


def strfdelta(tdelta, fmt):
    """
    Format a timedelta object as a string.
    """
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    return fmt.format(**d)


def format_time_delta(time_point, is_future):
    """
    Compute the delta between a time point (future or past) and the current 
    time and format it accordingly.
    """    
    try:
        time_point = datetime.strptime(time_point, '%Y-%m-%dT%H:%M:%S')
        if is_future:
            delta = time_point - datetime.now()
        else:    
            delta = datetime.now() - time_point
        time_formatted = strfdelta(delta, "{D}d:{H:02}h:{M:02}m:{S:02}s")
    except ValueError:
        time_formatted = 'UNKNOWN'
    return time_formatted


class Job():
    def __init__(self, raw_dict):
        self.raw_dict = raw_dict
        self.job_id = raw_dict['JobId']
        self.name = raw_dict['JobName']
        self.user_id, self.user_id_partition = self.parse_user_id()
        self.partition = raw_dict['Partition'][0].upper()
        self.status = raw_dict['JobState']
        self.runtime = str(raw_dict['RunTime'])
        self.qos = self.parse_qos()
        self.qos_order = qos_order_dict[self.qos]
        self.priority = raw_dict['Priority']
        self.node_list = raw_dict['NodeList']

    def parse_user_id(self):
        job = self.raw_dict
        user_id = re.match('(.*)\\(.*\\)', job['UserId']).group(1)
        partition = job['Partition'][0].upper()
        user_id_partition = f'{partition} {user_id}'
        return user_id, user_id_partition

    def display_raw(self, attributes):
        return ' '.join([self.raw_dict[attribute] for attribute in attributes])
    
    def display_attr_simple(self, attributes):
        return ' '.join([getattr(self, attribute) for attribute in attributes])
    
    def display(self, attr_dict):
        return '  '.join([f'{getattr(self, attr)}'[:l].ljust(l) 
                         for attr, l in attr_dict.items()])
    
    def parse_qos(self):
        qos = self.raw_dict['QOS']
        is_deadline = 'deadline' in qos
        is_master = 'master' in qos
        is_phd = 'phd' in qos
        if is_phd and is_deadline:
            return 'phd|d'
        elif is_phd and not is_deadline:
            return 'phd|n'
        elif is_master and is_deadline:
            return 'msc|d'
        elif is_master and not is_deadline:
            return 'msc|n'
        else:
            return 'other'


class RunningJob(Job):
    def __init__(self, raw_dict):
        if raw_dict['JobState'] != 'RUNNING':
            raise ValueError(
                f'Faulty instantiation. ' + 
                f'Expected job state RUNNING, got {raw_dict["JobState"]}')
        super().__init__(raw_dict)
        self.node = raw_dict['Nodes']
        self.gpus = self.count_gpus_in_use()
        self.gres = self.parse_gres()
        self.cpus = int(raw_dict['NumCPUs'])
        self.remaining_time = format_time_delta(
            raw_dict['EndTime'], is_future=True)
        self.runtime = raw_dict['RunTime']
        self.start_time = raw_dict['StartTime']
        self.time_limit = raw_dict['TimeLimit']
        self.mem = f"{int(raw_dict['Mem']) // 1000} GB"

    def count_gpus_in_use(self):
        job = self.raw_dict
        matches = re.match('.*gres/gpu=([0-9]*).*', job['AllocTRES'])
        if matches is not None:
            return int(matches.group(1))
        else:
            return 0

    def parse_gres(self):
        job = self.raw_dict
        if self.gpus > 0:
            gres = job['GRES']
            matches = re.match('.*(IDX:.*)\\)', job['GRES'])
            if matches is None:
                gres = f'UNKNOWN ({self.node})'
            else:
                # FIXME: Won't work for more than 1 GPU
                idx = matches.group(1)[4:]
                gres = f'{self.node[:4]} {idx}'     
        else:
            gres = ''
        return gres


class PendingJob(Job):
    def __init__(self, raw_dict):
        if raw_dict['JobState'] != 'PENDING':
            raise ValueError(
                f'Faulty instantiation. ' + 
                f'Expected job state PENDING, got {raw_dict["JobState"]}')
        super().__init__(raw_dict)
        self.gpus = raw_dict['NumNodes']
        # self.gres = raw_dict['GRES']
        self.cpus = int(raw_dict['NumCPUs'])
        self.remaining_time = format_time_delta(
            raw_dict['EndTime'], is_future=True)
        self.runtime = raw_dict['RunTime']
        self.start_time = raw_dict['StartTime']
        self.time_limit = raw_dict['TimeLimit']


class OtherJob(Job):
    def __init__(self, raw_dict):
        if raw_dict['JobState'] in ['PENDING', 'RUNNING']:
            raise ValueError(
                f'Faulty instantiation. ' + 
                f'Expected job state other than PENDING or RUNNING, '+\
                    f'got {raw_dict["JobState"]}')
        super().__init__(raw_dict)
        self.elapsed_time = format_time_delta(
            raw_dict['EndTime'], is_future=False)
        self.end_time = raw_dict['EndTime']

    def is_recent(self, minutes=60, days=0):
        reference_time = datetime.strptime(self.end_time, '%Y-%m-%dT%H:%M:%S')
        delta = datetime.now() - reference_time
        if delta.days > days:
            return False
        return delta.seconds < minutes * 60

