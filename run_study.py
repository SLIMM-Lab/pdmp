#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import argparse
import subprocess as sp


def get_n_cups():
    # Command to sum ppn values for jobs that specify Resource_List.nodes
    cmd_primary = (
        "qstat -f -u lriccius | awk '/Resource_List.nodes/ { "
        "if (match($0, /:ppn=([0-9]+)/, arr)) { sum += arr[1] } else { sum += 1 } "
        "} END { print sum }'"
    )
    # "qstat -f -u lriccius | grep 'Resource_List.nodes' | sed -E 's/.*:ppn=([0-9]+).*/\1/' | awk '{sum += $1} END {print sum}'"
    primary_process = sp.run(cmd_primary, shell=True, stderr=sp.DEVNULL,
                             stdout=sp.PIPE, text=True)
    primary_sum_str = primary_process.stdout.strip()
    primary_sum = int(primary_sum_str) if primary_sum_str else 0

    # Count how many jobs have Resource_List.nodes specified
    cmd_count_with_nodes = "qstat -f -u lriccius | grep -c 'Resource_List.nodes'"
    with_nodes_process = sp.run(cmd_count_with_nodes, shell=True, stderr=sp.DEVNULL,
                                stdout=sp.PIPE, text=True)
    count_with_nodes_str = with_nodes_process.stdout.strip()
    count_with_nodes = int(count_with_nodes_str) if count_with_nodes_str else 0

    # Count total number of jobs (assuming each job prints a line starting with a digit)
    cmd_total_jobs = "qstat -u lriccius | grep -E '^[0-9]+' | wc -l"
    total_jobs_process = sp.run(cmd_total_jobs, shell=True, stderr=sp.DEVNULL,
                                stdout=sp.PIPE, text=True)
    total_jobs_str = total_jobs_process.stdout.strip()
    total_jobs = int(total_jobs_str) if total_jobs_str else 0

    # For jobs that don't specify any node/ppn, assume 1 CPU per job.
    fallback_count = total_jobs - count_with_nodes

    # Total CPUs in use: sum from jobs with resource specification + fallback (1 per job)
    return primary_sum + fallback_count


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run all listed in the --job-file in the --study-dir.')
    parser.add_argument(
        '--job-file',
        type=str,
        default='jobs.txt',
        help='The name of the jobfile.',
    )
    parser.add_argument(
        '--cpu-limit',
        type=int,
        default=100,
        help='The maximum number of cpus occupied simultaniously.',
    )
    parser.add_argument(
        '--sleep-time',
        type=float,
        default=10.,
        help='The time between subsequent queue checks.',
    )

    # read all arguments
    args = parser.parse_args()
    job_file = args.job_file
    cpu_limit = args.cpu_limit
    sleep_time = args.sleep_time

    # get working directory
    study_dir = os.getcwd()

    # read jobs file
    with open(os.path.join(study_dir, job_file), 'r') as file:
        jobs = file.read().splitlines()

    # convert all relative paths to absolute paths
    for i, job in enumerate(jobs):
        jobs[i] = os.path.join(study_dir, job)

    # loop over all jobs and submit
    for job in jobs:

        cpus_used = get_n_cups()

        # wait until there are available CPUs
        if cpus_used >= cpu_limit:
            print(f'Waiting for CPU vacancy... Currently using {cpus_used} CPUs out of {cpu_limit}')

            while cpus_used >= cpu_limit:
                time.sleep(sleep_time)
                cpus_used = get_n_cups()

        print(f'Found vacancy! Running job in {job}')

        # run the job
        sp.run(f'bash -c "cd {job};qsub job"', shell=True)

        print(f'Currently using {cpus_used} CPUs')
        time.sleep(1.)

    print(f'Done running all {len(jobs)} jobs!')
