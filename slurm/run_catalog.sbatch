#!/bin/bash
# Usage: sbatch slurm/run_catalog.sbatch <cosmo> <hod> [space] [extra args...]
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 06:00:00
#SBATCH -J astra_ds
#SBATCH -o /global/homes/h/hugon/home/astra-ds/logs/%x_%j.out

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

COSMO=$1; HOD=$2; SPACE=${3:-z}; shift 3 || shift $#
cd /global/homes/h/hugon/home/astra-ds
srun python scripts/run_catalog.py --cosmo "$COSMO" --hod "$HOD" --space "$SPACE" \
    --nthreads "$SLURM_CPUS_PER_TASK" "$@"
