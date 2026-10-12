#!/bin/bash
# Stage 1: official AbacusHOD mocks (covariance) + Quijote derivatives and first fiducial realisations.
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 8
#SBATCH -t 05:00:00
#SBATCH -J ads_night1
#SBATCH -o /global/homes/h/hugon/home/astra-ds/logs/%x_%j.out
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
step() { srun -N 1 -n 1 -c 256 --exact "$@" --nthreads 128 < /dev/null & sleep 1; }
for k in 0 1 2 3; do step python -u scripts/run_official.py --kind covbox --chunk $k --nchunks 4 > logs/night_covbox_$k.log 2>&1; done
for k in 0 1 2 3 4; do step python -u scripts/run_official.py --kind cubicbox --chunk $k --nchunks 5 > logs/night_cubicbox_$k.log 2>&1; done
for s in Om_p Om_m Ob2_p Ob2_m h_p h_m ns_p ns_m s8_p s8_m Mnu_p Mnu_pp Mnu_ppp fiducial_ZA; do
    step python -u scripts/run_quijote.py --set $s --start 0 --stop 500 > logs/night_quijote_$s.log 2>&1
done
for k in 0 1 2 3; do step python -u scripts/run_quijote.py --set fiducial --start 0 --stop 4000 --chunk $k --nchunks 4 > logs/night_quijote_fid_$k.log 2>&1; done
wait
echo "stage 1 done $(date)"
