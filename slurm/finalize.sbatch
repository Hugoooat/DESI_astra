#!/bin/bash
# Rebuild the emulator dataset, retrain the 10-network ensembles and redo all the emulator forecasts.
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:30:00
#SBATCH -J ads_finalize
#SBATCH -o /global/homes/h/hugon/home/astra-ds/logs/%x_%j.out
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
set -x
python -u scripts/build_emulator_dataset.py
srun -N 1 -n 1 -c 256 python -u scripts/train_emulator.py --nseeds 10 --models-dir models10 --tag _10seeds
srun -N 1 -n 1 -c 256 python -u scripts/train_emulator.py --nseeds 10 --models-dir models10_plots --tag ''  # figures + default validation file
cd scripts
python -u fisher_emulator.py --models-dir ../models10 --validation-tag _10seeds
python -u compare_methods.py --models-dir ../models10 --validation-tag _10seeds
python -u compare_methods.py --models-dir ../models10 --validation-tag _10seeds --compress
echo "finalize done: $(date)"
