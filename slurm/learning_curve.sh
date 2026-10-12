source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
for f in 0.25 0.5 0.75 1.0; do
  srun -N 1 -n 1 -c 64 --exact python -u scripts/train_emulator.py --nseeds 3 --pairs data/pair_dataset_full.npz --lam 1 \
     --train-fraction $f --stats gg astra_rand_cross_q1 astra_rand_cross_q4 ds_rand_cross_q1 ds_rand_cross_q4 \
     --models-dir /pscratch/sd/h/hugon/astra-ds-store/models_lc$f --tag _lc$f > logs/learning_curve_$f.log 2>&1 &
  sleep 2
done
wait
echo "learning curve done $(date)"
