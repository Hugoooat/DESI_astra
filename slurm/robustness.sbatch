#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --cpus-per-task=256
#SBATCH -t 00:30:00
#SBATCH -J ads_robustness
#SBATCH -o /global/homes/h/hugon/home/astra-ds/logs/%x_%j.out
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
srun python scripts/robustness.py --nbox 8 --nthreads 128
