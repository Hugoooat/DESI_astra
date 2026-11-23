#!/bin/bash
# Usage: sbatch -N <nodes> -t <time> slurm/run_list.sbatch <list files...>
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J ads_list
#SBATCH -o /global/homes/h/hugon/home/astra-ds/logs/%x_%j.out
bash /global/homes/h/hugon/home/astra-ds/slurm/run_list.sh "$@"
