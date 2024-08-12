# ------------------------------------------------------------------------------
# File: job_master.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Mon Aug 12 2024
#
# Handles sorting and overview of jobs.
#
# ------------------------------------------------------------------------------
# %%
import os
import subprocess
from functools import reduce

from jobs import RunningJob, PendingJob, OtherJob
from nodes import NodeMaster


class JobMaster:
    def __init__(self):
        self.user = os.environ['LOGNAME']
        self.raw_jobs = self.parse_jobs_dict()
        self.running_jobs = [RunningJob(job) for job in self.raw_jobs 
                             if job['JobState'] == 'RUNNING']
        self.pending_jobs = [PendingJob(job) for job in self.raw_jobs 
                             if job['JobState'] == 'PENDING']
        self.other_jobs = [OtherJob(job) for job in self.raw_jobs
                            if job['JobState'] not in ['RUNNING', 'PENDING']]
        self.running_jobs = self.sort_jobs_qos(self.running_jobs)
        self.pending_jobs = sorted(self.pending_jobs, key=lambda x: x.priority)
        self.other_jobs = self.sort_jobs_qos(self.other_jobs)

        self.user_running_jobs = [job for job in self.running_jobs
                                    if job.user_id == self.user]
        self.user_pending_jobs = [job for job in self.pending_jobs
                                    if job.user_id == self.user]
        self.user_other_jobs = [job for job in self.other_jobs
                                if job.user_id == self.user]
        self.user_running_jobs = sorted(self.user_running_jobs, 
                                        key=lambda x: x.job_id)
        self.user_pending_jobs = sorted(self.user_pending_jobs,
                                        key=lambda x: x.job_id)
        self.user_other_jobs = sorted(self.user_other_jobs,
                                        key=lambda x: x.job_id)

        self.node_master = NodeMaster()

    def parse_jobs_dict(self):
        def parse_job_attributes(job):
            attributes = [tuple(x.split('=')) for x in job.split(' ') if x != '']
            attributes = dict([(x[0], '='.join(x[1:])) for x in attributes])
            return attributes
        
        output = subprocess.check_output(['scontrol','show', 'job', '-d'])
        output = output.decode('utf-8')
        jobs_raw = [x.replace('\n', ' ') for x in output.split('\n\n')]
        jobs = [parse_job_attributes(job) for job in jobs_raw if job != '']
        return jobs

    def sort_jobs_qos(self, jobs_list):
        return sorted(jobs_list, key=lambda x: (x.qos_order, x.name))
    
    def display_jobs(self, jobs_list, display_dict):
        for elem, l in display_dict.items():
            print(elem[:l].ljust(l), end='  ')
        print()
        for job in jobs_list:
            print(job.display(display_dict))

    def get_gpus_on_node(self, node_name):
        jobwise_counts = [job.gpus for job in self.running_jobs 
                        if job.node == node_name]
        if len(jobwise_counts) == 0:
            return 0
        else:
            return reduce(lambda x, y: x + y, jobwise_counts)

