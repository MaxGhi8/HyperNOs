# Slurm Template Usage Guide

This guide explains how to use the `template.slurm` file to submit Ray-based jobs to the cluster.

## Overview

The `template.slurm` file is a [Slurm](https://slurm.schedmd.com/) batch script designed to launch a Ray cluster across multiple nodes. It handles:

- Allocating resources (nodes, CPUs, GPUs).
- Setting up the environment.
- Starting the Ray head node.
- Starting Ray worker nodes on remaining allocated nodes.
- Running your Python script.

## Comparison to Standard Slurm Scripts

Unlike a standard single-node script, this template:

1.  **Dynamically detects node IPs**: It finds the head node's IP to allow workers to connect.
2.  **Manages Ray processes**: It explicitly starts `ray start --head` and `ray start --address` commands.
3.  **Splits IPv6 addresses**: It includes logic to handle specific network configurations found in some high-performance computing (HPC) environments.

## Step-by-Step Instructions

1.  **Copy the Template**
    Create a copy of the template for your specific job to avoid modifying the original:

    ```bash
    cp template.slurm my_job.slurm
    ```

2.  **Fill in the Placeholders**
    Open `my_job.slurm` and replace the following placeholders:

    | Placeholder         | Description               | Example            |
    | :------------------ | :------------------------ | :----------------- |
    | `<YOUR_ACCOUNT>`    | Your cluster account name | `my_account`       |
    | `<PARTITION_NAME>`  | Partition to submit to    | `boost_usr_prod`   |
    | `<TIME_LIMIT>`      | Max runtime (HH:MM:SS)    | `04:00:00`         |
    | `<NUM_NODES>`       | Total number of nodes     | `4`                |
    | `<TASKS_PER_NODE>`  | Slurm tasks per node      | `1`                |
    | `<CPUS_PER_TASK>`   | CPUs per Ray instance     | `32`               |
    | `<GPUS_PER_NODE>`   | GPUs per node to request  | `4`                |
    | `<GPUS_PER_TASK>`   | GPUs per Ray instance     | `4`                |
    | `<MEMORY>`          | Memory per node           | `494000` (in MB)   |
    | `<OUTPUT_FILENAME>` | Prefix for log files      | `ray_logs`         |
    | `<PATH_TO_VENV>`    | Path to your Python venv  | `/home/user/myenv` |
    | `<SCRIPT_NAME>`     | Python script to run      | `train_model`      |

    **Example Configuration:**

    ```bash
    #SBATCH -A my_account
    #SBATCH -p boost_usr_prod
    #SBATCH --time 04:00:00
    #SBATCH -N 4
    ...
    source /home/user/myenv/bin/activate
    ...
    python3 train_model.py
    ```

3.  **Submit the Job**
    Submit your configured script to the scheduler:

    ```bash
    sbatch my_job.slurm
    ```

4.  **Monitor Progress**
    Check the status of your job:
    ```bash
    squeue -u $USER
    ```
    Output logs will be written to the file specified in `--output` (e.g., `ray_logs_<node_name>_<job_id>`).
