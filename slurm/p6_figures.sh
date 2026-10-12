source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
P="omega_b omega_cdm n_s sigma8 w0 wa"
step() { srun -N 1 -n 1 -c 48 --exact "$@" < /dev/null & sleep 1; }
step python -u scripts/paper_figures.py --only fig_derivatives fig_triangle fig_quartiles --params $P > logs/p6_paper_figures.log 2>&1
step python -u scripts/paper_figures_v1style.py --only derivatives quartiles autocross smin constraints --params $P > logs/p6_v1style.log 2>&1
cd scripts
for f in hod239 0_212 0_795; do
  step python -u compare_methods.py --models-dir ../models_pairs --validation-tag _pairs --compress --cov covbox --fiducial $f --params $P > ../logs/compare_p6_fid$f.log 2>&1
done
wait
echo "p6 done $(date)"
