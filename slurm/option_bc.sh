#!/bin/bash
# Options b (pair-difference loss) and c (small-scale weight) + reduced pair regression.
#   salloc -N 4 -C cpu -q interactive -t 03:30:00 -A desi bash slurm/option_bc.sh
# Every heavy command is an srun step (salloc runs this script on the login node).
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
mkdir -p logs
run() { srun -N 1 -n 1 -c ${NCPU:-256} --exact "$@"; }
T="python -u scripts/train_emulator.py"

echo "== 1. pair datasets and reduced regressions $(date)"
run python -u scripts/build_pair_dataset.py --holdout 4 --out pair_dataset.npz > logs/bc_pairs_val.log 2>&1 &
run python -u scripts/build_pair_dataset.py --holdout 0 --out pair_dataset_full.npz > logs/bc_pairs_full.log 2>&1 &
(cd scripts && for k in 2 3 4 6; do
     srun -N 1 -n 1 -c 256 --exact python -u pair_regression.py --ndir $k; done) > logs/bc_regression.log 2>&1 &
wait

echo "== 2. trainings $(date)"
NCPU=128 run $T --nseeds 5 --pairs data/pair_dataset.npz --lam 0 --models-dir models_val_base --tag _val_base > logs/bc_val_base.log 2>&1 &
NCPU=128 run $T --nseeds 5 --pairs data/pair_dataset.npz --lam 1 --models-dir models_val_pairs --tag _val_pairs > logs/bc_val_pairs.log 2>&1 &
NCPU=128 run $T --nseeds 5 --pairs data/pair_dataset.npz --lam 1 --small-scale-weight 1 --models-dir models_val_pairs_ssw --tag _val_pairs_ssw > logs/bc_val_pairs_ssw.log 2>&1 &
NCPU=128 run $T --nseeds 10 --pairs data/pair_dataset_full.npz --lam 1 --models-dir models_pairs --tag _pairs > logs/bc_final_pairs.log 2>&1 &
run $T --nseeds 10 --pairs data/pair_dataset_full.npz --lam 1 --small-scale-weight 1 --models-dir models_pairs_ssw --tag _pairs_ssw > logs/bc_final_pairs_ssw.log 2>&1 &
wait

echo "== 3. validation and forecasts $(date)"
cd scripts
run python -u validate_local.py ../models_val_base ../models_val_pairs ../models_val_pairs_ssw > ../logs/bc_validate_local.log 2>&1
for m in pairs pairs_ssw; do
    run python -u fisher_emulator.py --models-dir ../models_$m --validation-tag _$m > ../logs/bc_fisher_$m.log 2>&1
    run python -u compare_methods.py --models-dir ../models_$m --validation-tag _$m --compress > ../logs/bc_compare_$m.log 2>&1
done
for k in 3 4; do
    run python -u compare_methods.py --models-dir ../models_pairs --validation-tag _pairs --compress \
        --regression ../data/derivatives_regression_ndir$k.npz > ../logs/bc_compare_regression_ndir$k.log 2>&1
done
echo "== done $(date)"
