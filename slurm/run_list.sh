#!/bin/bash
# Run every line "cosmo hod space los" of the given list files as one full-node srun
# step inside the current allocation; outputs that already exist are skipped, so an
# interrupted list can be resumed by another allocation. Use from salloc or sbatch:
#   salloc -N 4 -C cpu -q interactive -t 02:30:00 -A desi bash slurm/run_list.sh config/runs_*.txt
#   sbatch -N 8 -t 06:00:00 slurm/run_list.sbatch config/runs_emulator.txt
# NB: salloc runs this script on the login node: everything heavy goes through srun.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
mkdir -p logs

while read c h space los; do
    [ -z "$c" ] && continue
    tag=c$(printf %03d $c)_hod$(printf %03d $h)_${space}_los${los}
    [ -f "data/${tag}.npz" ] && continue
    srun -N 1 -n 1 -c 256 --exclusive \
        python scripts/run_catalog.py --cosmo "$c" --hod "$h" --space "$space" --los "$los" --nthreads 128 \
        < /dev/null > "logs/${tag}.log" 2>&1 &
    sleep 1
done < <(cat "$@")
wait
echo "all done: $(date)"
