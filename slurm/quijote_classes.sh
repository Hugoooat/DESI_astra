source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
step() { srun -N 1 -n 1 -c 24 --exact "$@" --nthreads 24 < /dev/null & sleep 1; }
for k in 0 1 2 3 4 5 6 7; do step python -u scripts/run_quijote.py --classes --set fiducial --start 1000 --stop 2000 --chunk $k --nchunks 8 > logs/qclass_fid_$k.log 2>&1; done
for s in Om_p Om_m ns_p ns_m s8_p s8_m; do for k in 0 1; do
    step python -u scripts/run_quijote.py --classes --set $s --start 0 --stop 500 --chunk $k --nchunks 2 > logs/qclass_${s}_$k.log 2>&1
done; done
wait
echo "quijote classes done $(date)"
