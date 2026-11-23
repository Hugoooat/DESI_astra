source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
step() { srun -N 1 -n 1 -c 24 --exact "$@" --nthreads 24 < /dev/null & sleep 1; }
for k in $(seq 0 19); do step python -u scripts/run_quijote.py --classes --set fiducial --start 0 --stop 4000 --chunk $k --nchunks 20 > logs/qclass_fidfull_$k.log 2>&1; done
wait
echo "quijote classes full done $(date)"
