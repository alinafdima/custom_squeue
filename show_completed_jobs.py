# %%
import os
from pprint import pprint
from job_master import JobMaster
jobs = JobMaster()

other_jobs = jobs.other_jobs

for job in other_jobs:
    if job.status != 'COMPLETED':
        continue
    job_id = job.job_id
    runtime = job.runtime
    requested_time = job.raw_dict['TimeLimit']
    job_state = job.status
    print(f'{job.user_id}: {runtime} / {requested_time} ({job_state})')