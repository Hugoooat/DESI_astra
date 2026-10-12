source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
export OMP_NUM_THREADS=8
mkdir -p logs/mcmc_v1 logs/quijote_subsets
cat config/mcmc_tasks_v1.txt | srun -N 1 -n 1 -c 256 --exact xargs -P 30 -L 1 bash -c \
  'python scripts/mcmc.py --data $0 --vector $1 --cov covbox --volume-factor 1 --nsteps 8000 --nwalkers 64 > logs/mcmc_v1/$(echo $0 | tr ":" "_")_$1.log 2>&1' &
cat config/quijote_tasks.txt | srun -N 1 -n 1 -c 256 --exact xargs -P 9 -L 1 bash -c \
  'python -u scripts/quijote_fisher.py "$@" > logs/quijote_subsets/$(echo "$@" | grep -o "_p[0-9]c[0-9]").log 2>&1' _ &
wait
echo "morning done $(date)"
