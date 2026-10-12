#!/bin/bash
# Coverage test of the emulator inference on the held-out test cosmologies (subbox covariance, full-box volume).
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J ads_mcmccov
#SBATCH -o /global/homes/h/hugon/home/astra-ds/logs/%x_%j.out
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
export OMP_NUM_THREADS=8
cat config/mcmc_coverage_tasks.txt | srun -N 1 -n 1 -c 256 xargs -P 30 -L 1 bash -c \
  'python scripts/mcmc.py --data $0 --vector $1 --cov subbox --volume-factor 64 --nsteps 8000 --nwalkers 64 > logs/mcmc_$(echo $0 | tr ":" "_")_$1.log 2>&1'
echo "coverage done $(date)"
