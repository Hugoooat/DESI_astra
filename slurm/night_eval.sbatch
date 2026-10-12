#!/bin/bash
# Evaluation: Quijote Fisher with convergence tests -> decides whether stage 2 is useful.
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J ads_eval
#SBATCH -o /global/homes/h/hugon/home/astra-ds/logs/%x_%j.out
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
echo "official: covbox $(ls data/official/covbox_*.npz | wc -l), cubicbox $(ls data/official/cubicbox_*.npz | wc -l)"
for s in data/quijote/*/; do echo "$s $(ls $s | grep -c npz)"; done
python -u scripts/quijote_fisher.py --report
