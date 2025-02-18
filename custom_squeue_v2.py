# ------------------------------------------------------------------------------
# File: custom_squeue_v2.py
# Author: Alina Dima <alina.dima@tum.de>
# Created on Mon Aug 12 2024
#
# Custom squeue, but even better.
#
# ------------------------------------------------------------------------------
# %%

import argparse
from job_master import JobMaster


display_dict_running = {
    "job_id": 6,
    "node": 12,
    # 'user_id': 8,
    # 'qos': 5,
    "gpus": 4,
    "cpus": 4,
    "mem": 6,
    "priority": 7,
    "runtime": 10,
    "remaining_time": 14,
    "name": 27,
}

display_dict_pending = {
    "job_id": 6,
    "user_id": 8,
    "qos": 5,
    "gpus": 4,
    "cpus": 4,
    "priority": 10,
    "time_limit": 10,
    "name": 27,
}

display_dict_other = {
    "job_id": 6,
    "user_id": 8,
    "qos": 5,
    "status": 15,
    "runtime": 10,
    "elapsed_time": 15,
}


def print_free_gpus(job_master: JobMaster):
    # For a list of nodes compute: used GPUs, total GPUs, free GPUs
    data = {}
    for node in job_master.node_master.nodes:
        if node.is_unavailable:
            continue
        free_gpus = job_master.get_gpus_on_node(node.name)
        used_gpus = node.gpu_count - free_gpus
        data[node.name] = (free_gpus, node.gpu_count, used_gpus)

    names_non_aster = [
        n.name
        for n in job_master.node_master.nodes
        if not n.is_asteroid and not n.is_unavailable
    ]

    total_gpus = sum([x[1] for x in data.values()])
    used_gpus = sum([x[0] for x in data.values()])
    free_gpus = sum([x[2] for x in data.values()])
    free_non_aster = sum(
        [x[2] for k, x in data.items() if k in names_non_aster]
    )

    print(
        f"Free GPUs: {free_gpus} / {total_gpus}",
        f"({free_non_aster} are non-asteroid GPUs)",
    )


def print_unavailable_nodes(job_master: JobMaster):
    unavailable_nodes = [
        (node.name, node.state)
        for node in job_master.node_master.nodes
        if node.is_unavailable and not node.is_asteroid
    ]
    if len(unavailable_nodes) > 0:
        unav_string = ", ".join(
            [f"{node} ({state})" for node, state in unavailable_nodes]
        )
        print(f"Unavailable nodes: {unav_string}")
    else:
        print("All the nodes in the universe are up and running.")


def print_available_nodes(job_master: JobMaster):
    available_nodes = []
    for node in job_master.node_master.nodes:
        if not node.is_unavailable:
            gpus = job_master.get_gpus_on_node(node.name)
            free_gpus = node.gpu_count - gpus
            total_gpus = node.gpu_count
            available_nodes.append((node.name, free_gpus, total_gpus))
    avail_string = ", ".join(
        [f"{node} ({gpus}/{total})" for node, gpus, total in available_nodes]
    )
    print(f"Available nodes: {avail_string}")


def print_overall_gpu_info(job_master, available=True, unavailable=True):
    print_free_gpus(job_master)
    if available:
        print_available_nodes(job_master)
    if unavailable:
        print_unavailable_nodes(job_master)


def print_my_running_jobs(jobs: JobMaster):
    running_jobs = jobs.user_running_jobs
    if len(running_jobs) > 0:
        print(
            f"My running jobs: ({len(running_jobs)}).",
            "[Only the first 10 are displayed]",
        )
        jobs.display_jobs(running_jobs, display_dict_running)
    else:
        print("No running jobs of the current user.")


def print_my_pending_jobs(jobs: JobMaster):
    pending_jobs = jobs.user_pending_jobs
    if len(pending_jobs) > 0:
        print()
        print(
            f"My pending jobs: ({len(pending_jobs)}).",
            "[Only the first 10 are displayed]",
        )
        jobs.display_jobs(pending_jobs[:10], display_dict_pending)


def print_my_other_jobs(jobs: JobMaster):
    other_jobs = jobs.user_other_jobs
    recent_jobs = [job for job in other_jobs if job.is_recent()]
    if len(other_jobs) > 0:
        print()
        print("My past jobs:")
        jobs.display_jobs(other_jobs, display_dict_other)


def print_usage_breakdown(jobs: JobMaster):
    print("GPU usage per user (running jobs):")
    running_users = set([job.user_id for job in jobs.running_jobs])
    gpus_per_user = {
        user: sum(
            [job.gpus for job in jobs.running_jobs if job.user_id == user]
        )
        for user in running_users
    }
    gpus_per_user = sorted(
        [(user, gpus) for user, gpus in gpus_per_user.items()],
        key=lambda x: (-x[1], x[0]),
    )
    for user, gpus in gpus_per_user:
        user_jobs = [job for job in jobs.running_jobs if job.user_id == user]
        user_qos = user_jobs[0].qos
        user_name = user_jobs[0].user_name
        if gpus > 0:
            print(
                f"{user}".ljust(10)
                + f" {user_name} ".ljust(30)
                + f" ({user_qos}) ".ljust(10)
                + f"{gpus} GPUs"
            )
    master_student_jobs = [job for job in jobs.running_jobs if "msc" in job.qos]
    phd_student_jobs = [job for job in jobs.running_jobs if "phd" in job.qos]

    print()
    phd_gpus = sum([job.gpus for job in phd_student_jobs])
    msc_gpus = sum([job.gpus for job in master_student_jobs])
    print(f"Total phd student jobs: {phd_gpus}")
    print(f"Total msc student jobs: {msc_gpus}")


def print_pending_usage_breakdown(jobs: JobMaster):
    print("Breakdown of GPU requests per user (pending jobs):")
    pending_users = set([job.user_id for job in jobs.pending_jobs])
    gpus_per_user_p = {
        user: [job.gres for job in jobs.pending_jobs if job.user_id == user]
        for user in pending_users
    }
    jobs_per_user_p = {
        user: len(gpus) for user, gpus in gpus_per_user_p.items()
    }
    gpus_per_user_p = sorted(
        [(user, gpus) for user, gpus in gpus_per_user_p.items()],
        key=lambda x: x[0],
    )
    for user, gpus in gpus_per_user_p:
        if len(gpus) > 0:
            user_jobs = [
                job for job in jobs.pending_jobs if job.user_id == user
            ]
            user_qos = user_jobs[0].qos
            user_name = user_jobs[0].user_name
            print(
                f"{user}".ljust(10)
                + f" {user_name} ".ljust(30)
                + f" ({user_qos}) ".ljust(5)
                + f" {jobs_per_user_p[user]} jobs: "
                + f' [{", ".join([str(x) for x in gpus])}]'
            )


def print_running_jobs_all(jobs: JobMaster, exclude_user):
    print("Running jobs all:")
    if exclude_user:
        running_jobs_all = [
            job for job in jobs.running_jobs if job.user_id != jobs.user
        ]
    else:
        running_jobs_all = jobs.running_jobs
    jobs.display_jobs(running_jobs_all, display_dict_running)


def print_pending_jobs_all(jobs: JobMaster, exclude_user):
    print("Pending jobs all:")
    if exclude_user:
        pending_jobs_all = [
            job for job in jobs.pending_jobs if job.user_id != jobs.user
        ]
    else:
        pending_jobs_all = jobs.pending_jobs
    jobs.display_jobs(pending_jobs_all, display_dict_pending)


parser = argparse.ArgumentParser(description="Custom squeue.")
# Profiles
parser.add_argument("--default", action="store_true", help="Show default info.")
parser.add_argument(
    "--more", action="store_true", help="Show info. + all running jobs."
)
parser.add_argument("--all", action="store_true", help="Show all info.")
parser.add_argument("--jobs", action="store_true", help="Show only job info.")

parser.add_argument(
    "--overall", action="store_true", help="Show overall GPU info."
)
parser.add_argument("--user", action="store_true", help="Show only user jobs.")
parser.add_argument(
    "--all_users", action="store_true", help="Show jobs from all users."
)
parser.add_argument(
    "--usage", action="store_true", help="Show GPU usage breakdown."
)
parser.add_argument(
    "--all_users_pending",
    action="store_true",
    help="Show pending jobs from all users.",
)
# Default values
parser.set_defaults(
    overall=False,
    user=False,
    usage=False,
    all_users=False,
    all_users_pending=False,
)


if __name__ == "__main__":
    args = parser.parse_args()

    overall = args.overall
    user = args.user
    all_users = args.all_users
    usage = args.usage
    all_users_pending = args.all_users_pending

    # Profiles
    if args.default:
        overall = True
        user = True
        usage = True
        all_users = False
        all_users_pending = False
    if args.more:
        overall = True
        user = True
        usage = True
        all_users = True
        all_users_pending = False
    if args.jobs:
        overall = False
        user = False
        usage = False
        all_users = True
        all_users_pending = True
    if args.all:
        overall = True
        user = True
        usage = True
        all_users = True
        all_users_pending = True

    jobs = JobMaster()
    if overall:
        print_overall_gpu_info(jobs, unavailable=False)
        print()
    if user:
        print_my_running_jobs(jobs)
        print_my_pending_jobs(jobs)
        # print_my_other_jobs(jobs)
        print()
    if usage:
        print_usage_breakdown(jobs)
        print()
        print_pending_usage_breakdown(jobs)
        print()
    if all_users:
        print_running_jobs_all(jobs, exclude_user=user)
        print()
    if all_users_pending:
        print_pending_jobs_all(jobs, exclude_user=user)
        print()
