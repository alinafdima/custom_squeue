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
import subprocess
from datetime import datetime
from nodes import NodeMaster

public_node_names = [
    node.name for node in NodeMaster().nodes if not node.is_private
]

qos_order_dict = {
    "phd|d": 1,
    "phd|n": 2,
    "msc|d": 3,
    "msc|n": 4,
    "other": 5,
}


def get_full_name(username):
    command = f"getent passwd {username}"
    ret = subprocess.run(
        command, shell=True, check=True, stdout=subprocess.PIPE
    )
    output_raw = ret.stdout
    uname, _, uid, gid, fullname, home, shell = output_raw.decode().split(":")
    return fullname


def expand_gres(input):
    gpus = []
    for gpu_range in input.split(","):
        if "-" in gpu_range:
            start, end = [int(x) for x in gpu_range.split("-")]
            gpus.extend(range(start, end + 1))
        else:
            gpus.append(int(gpu_range))
    return gpus


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
        time_point = datetime.strptime(time_point, "%Y-%m-%dT%H:%M:%S")
        if is_future:
            delta = time_point - datetime.now()
        else:
            delta = datetime.now() - time_point
        time_formatted = strfdelta(delta, "{D}d:{H:02}h:{M:02}m:{S:02}s")
    except ValueError:
        time_formatted = "UNKNOWN"
    return time_formatted


class Job:
    def __init__(self, raw_dict):
        self.raw_dict = raw_dict
        self.job_id = raw_dict["JobId"]
        self.name = raw_dict["JobName"]
        self.user_id, self.user_id_partition = self.parse_user_id()
        self.user_name = get_full_name(self.user_id)
        self.partition = raw_dict["Partition"][0].upper()
        self.status = raw_dict["JobState"]
        self.runtime = str(raw_dict["RunTime"])
        self.qos = self.parse_qos()
        self.qos_order = qos_order_dict[self.qos]
        self.priority = raw_dict["Priority"]
        self.node_list = raw_dict["NodeList"]

    def parse_user_id(self):
        job = self.raw_dict
        user_id = re.match("(.*)\\(.*\\)", job["UserId"]).group(1)
        partition = job["Partition"][0].upper()
        user_id_partition = f"{partition} {user_id}"
        return user_id, user_id_partition

    def display_raw(self, attributes):
        return " ".join([self.raw_dict[attribute] for attribute in attributes])

    def display_attr_simple(self, attributes):
        return " ".join([getattr(self, attribute) for attribute in attributes])

    def display(self, attr_dict):
        return "  ".join(
            [
                f"{getattr(self, attr)}"[:l].ljust(l)
                for attr, l in attr_dict.items()
            ]
        )

    def parse_qos(self):
        qos = self.raw_dict["QOS"]
        is_deadline = "deadline" in qos
        is_master = "master" in qos
        is_phd = "phd" in qos
        if is_phd and is_deadline:
            return "phd|d"
        elif is_phd and not is_deadline:
            return "phd|n"
        elif is_master and is_deadline:
            return "msc|d"
        elif is_master and not is_deadline:
            return "msc|n"
        else:
            return "other"

    def to_dict(self):
        return {
            attr: getattr(self, attr)
            for attr in self.__dict__
            if attr != "raw_dict"
        }


class RunningJob(Job):
    def __init__(self, raw_dict):
        if raw_dict["JobState"] != "RUNNING":
            raise ValueError(
                f"Faulty instantiation. "
                + f'Expected job state RUNNING, got {raw_dict["JobState"]}'
            )
        super().__init__(raw_dict)
        self.node = raw_dict["Nodes"]
        self.gpus = self.count_gpus_in_use()
        self.gres, new_gpu_count = self.parse_gres()
        if new_gpu_count != self.gpus:
            print(f"WARNING: GPU count mismatch for job {self.job_id}")
            self.gpus = new_gpu_count
        self.cpus = int(raw_dict["NumCPUs"])
        self.remaining_time = format_time_delta(
            raw_dict["EndTime"], is_future=True
        )
        self.runtime = raw_dict["RunTime"]
        self.start_time = raw_dict["StartTime"]
        self.time_limit = raw_dict["TimeLimit"]
        self.mem = f"{int(raw_dict['Mem']) // 1000} GB"
        self.is_private_gpu = self.node not in public_node_names

    def count_gpus_in_use(self):
        job = self.raw_dict
        matches = re.match(".*gres/gpu=([0-9]*).*", job["AllocTRES"])
        if matches is not None:
            return int(matches.group(1))
        else:
            return 0

    def parse_gres(self):
        job = self.raw_dict
        if self.gpus > 0:
            gres = job["GRES"]
            matches = re.match(".*(IDX:.*)\\)", job["GRES"])
            if matches is None:
                gres = f"UNKNOWN ({self.node})"
                gpus_count = self.gpus
            else:
                try:
                    # Sometimes the matching goes wrong. This is a workaround.
                    indices_raw = matches.group(1)[4:]
                    expanded_gres = expand_gres(indices_raw)
                    gres = f"{self.node[:4]} {str(expanded_gres)}"
                    gpus_count = len(expanded_gres)
                    # gres = f'{self.node[:4]} {idx}'
                except ValueError:
                    gres = f"UNKNOWN ({self.node})"
                    gpus_count = self.gpus
        else:
            gres = ""
            gpus_count = 0
        return gres, gpus_count


class PendingJob(Job):
    def __init__(self, raw_dict):
        if raw_dict["JobState"] != "PENDING":
            raise ValueError(
                f"Faulty instantiation. "
                + f'Expected job state PENDING, got {raw_dict["JobState"]}'
            )
        super().__init__(raw_dict)
        self.gpus = raw_dict["NumNodes"]
        self.cpus = int(raw_dict["NumCPUs"])
        self.remaining_time = format_time_delta(
            raw_dict["EndTime"], is_future=True
        )
        self.runtime = raw_dict["RunTime"]
        self.start_time = raw_dict["StartTime"]
        self.time_limit = raw_dict["TimeLimit"]
        self.parse_gpus_from_tres()

    def parse_gpus_from_tres(self):
        tres = self.raw_dict["ReqTRES"]
        requests = tres.split(",")
        regex1 = r"gres/gpu=(\d+)"
        regex2 = r"gres/gpu:(.*)=(\d+)"
        matches1 = [re.match(regex1, x) for x in requests]
        matches2 = [re.match(regex2, x) for x in requests]
        gres = [m.group(1) for m in matches1 if m]
        if len(gres) > 0:
            gpus = int(gres[0])
        else:
            gpus = 0
        constraints = [m.group(1) for m in matches2 if m]
        if len(constraints) > 0:
            gpu_type = constraints[0].upper()
        else:
            gpu_type = "<ANY>"
        self.gpu_type = gpu_type
        self.gpus = gpus
        if gpus == 0:
            self.gres = "0"
        else:
            self.gres = f"{gpus}x{gpu_type}"


class OtherJob(Job):
    def __init__(self, raw_dict):
        if raw_dict["JobState"] in ["PENDING", "RUNNING"]:
            raise ValueError(
                f"Faulty instantiation. "
                + f"Expected job state other than PENDING or RUNNING, "
                + f'got {raw_dict["JobState"]}'
            )
        super().__init__(raw_dict)
        self.elapsed_time = format_time_delta(
            raw_dict["EndTime"], is_future=False
        )
        self.end_time = raw_dict["EndTime"]

    def is_recent(self, minutes=60, days=0):
        reference_time = datetime.strptime(self.end_time, "%Y-%m-%dT%H:%M:%S")
        delta = datetime.now() - reference_time
        if delta.days > days:
            return False
        return delta.seconds < minutes * 60
