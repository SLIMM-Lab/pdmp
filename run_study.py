#!/usr/bin/env python3

import time
import os
import argparse
import subprocess as sp


if __name__ == '__main__':

    # get pdmp dir from environment
    pdmp_dir = os.environ['PDMP']

    parser = argparse.ArgumentParser(description='Run all listed in the --job-file in the --study-dir.')
    parser.add_argument(
        '--study-dir',
        type=str,
        required=True,
        help='The relative path to the study directory.',
    )
    parser.add_argument(
        '--job-file',
        type=str,
        default='jobs.txt',
        help='The name of the jobfile.',
    )
    parser.add_argument(
        '--job-limit',
        type=int,
        default=100,
        help='The maximum number of jobs in the cue.',
    )
    parser.add_argument(
        '--sleep-time',
        type=float,
        default=10.,
        help='The time between subsequent queue checks.',
    )

    # read all arguments
    args = parser.parse_args()
    study_dir = args.study_dir
    job_file = args.job_file
    job_limit = args.job_limit
    sleep_time = args.sleep_time

    # read jobs file
    with open(os.path.join(pdmp_dir, study_dir, 'jobs.txt'), 'r') as file:
        jobs = file.read().splitlines()

    # convert all relative paths to absolute paths
    for i, job in enumerate(jobs):
        jobs[i] = os.path.join(pdmp_dir, job)

    # loop over all jobs and submit
    for job in jobs:

        # get current number of jobs
        nJobs = int(str(sp.run('qstat | grep lriccius | wc', shell=True,
                               stderr=sp.DEVNULL, stdout=sp.PIPE).stdout).split()[1])

        # wait until there is a vacancy
        if nJobs > (job_limit - 1):
            print(f'Waiting for queue vacancy... Currently running {nJobs} jobs')

            while nJobs > (job_limit - 1):
                time.sleep(sleep_time)
                nJobs = int(str(sp.run('qstat | grep lriccius | wc', shell=True,
                                       stderr=sp.DEVNULL, stdout=sp.PIPE).stdout).split()[1])

        print(f'Found vacancy! Running job in {job}')

        # run the job
        sp.run(f'bash -c "cd {job};qsub job"', shell=True)

        # # wait a bit before checking again
        # nJobs = int(str(sp.run('qstat | grep lriccius | wc', shell=True,
        #                        stderr=sp.DEVNULL, stdout=sp.PIPE).stdout).split()[1])

        print(f'Currently running {nJobs} jobs')
        time.sleep(1.)

    print(f'Done running all {len(jobs)} jobs!')
