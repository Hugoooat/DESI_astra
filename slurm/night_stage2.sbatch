#!/bin/bash
# Stage 2: remaining Quijote fiducial realisations, only if the evaluation asked for it.
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 8
#SBATCH -t 03:00:00
#SBATCH -J ads_night2
#SBATCH -o /global/homes/h/hugon/home/astra-ds/logs/%x_%j.out
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
if [ ! -f data/quijote/CONTINUE ]; then echo "evaluation: converged, stage 2 not needed"; exit 0; fi
cat data/quijote/CONTINUE
for k in 0 1 2 3 4 5 6 7; do
    srun -N 1 -n 1 -c 256 --exact python -u scripts/run_quijote.py --set fiducial --start 4000 --stop 15517 \
        --chunk $k --nchunks 8 --nthreads 128 < /dev/null > logs/night2_quijote_fid_$k.log 2>&1 &
    sleep 1
done
wait
python -u scripts/quijote_fisher.py
echo "stage 2 done $(date)"
