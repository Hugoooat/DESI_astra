#!/bin/bash
# Run the fiducial + 8 derivative catalogues, each as one full-node srun step:
#   salloc -N 4 -C cpu -q interactive -t 01:30:00 -A desi bash slurm/run_all_interactive.sh [los ...]
# NB: salloc runs this script on the login node: everything heavy must go through srun.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/h/hugon/home/astra-ds
mkdir -p logs
LOS_LIST=${@:-z x y}

# drop any queued batch duplicates
squeue -h -u "$USER" -t PENDING -o "%i %j" | awk '$2 ~ /^ads_/ {print $1}' | xargs -r scancel

python - <<'EOF' > logs/catalog_list.txt
import json
s = json.load(open('config/hod_selection.json'))
print(s['fiducial']['cosmo'], s['fiducial']['hod'])
for p in s['pairs'].values():
    for k in ('plus', 'minus'):
        print(p[k]['cosmo'], p[k]['hod'])
EOF

# priority: all catalogues along z, then the fiducial along x, y (covariance), then the pairs
for los in $LOS_LIST; do
    while read c h; do
        out=data/c$(printf %03d $c)_hod$(printf %03d $h)_z_los${los}.npz
        [ -f "$out" ] && continue
        srun -N 1 -n 1 -c 256 --exclusive \
            python scripts/run_catalog.py --cosmo "$c" --hod "$h" --space z --los "$los" --nthreads 128 \
            < /dev/null > "logs/run_c${c}_hod${h}_los${los}.log" 2>&1 &
        sleep 1
    done < logs/catalog_list.txt
done
wait
echo "all done"
