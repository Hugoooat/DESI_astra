#!/usr/bin/env python
"""
ASTRA split and density split on Quijote halos at z = 0.5 (Rockstar host
halos, out_3_pid.list), for Fisher forecasts with clean derivatives.

The tracers are all host halos (PID = -1; the catalogues contain halos with
at least 100 particles, n ~ 5e-5 (h/Mpc)^3, ten times sparser than the LRGs
of the AbacusSummit analysis). Redshift-space positions along z:
    z_s = z + v_z / (a H(a)),   H = 100 E(a) h km/s/Mpc, flat LCDM with the
Omega_m of each simulation (read from the catalogue header).
The whole 1 Gpc/h periodic box is one volume: periodic images for the
tessellation and the natural estimator of a periodic box for the clustering.

    python scripts/run_quijote.py --set fiducial --start 0 --stop 1000 [--chunk 0 --nchunks 8]
Outputs: data/quijote/{set}/{i}.npz
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import astra_ds

QUIJOTE = Path('/pscratch/sd/h/hugon/quijote')
BOX = 1000.


def read_halos(path):
    """Host-halo positions in redshift space (z axis) and the number of halos."""
    header = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            for item in line[1:].split(';'):
                if '=' in item:
                    k, v = item.split('=')
                    header[k.strip()] = float(v)
    d = pd.read_csv(path, sep=r'\s+', comment='#', header=None, usecols=[8, 9, 10, 13, 41],
                    names=['X', 'Y', 'Z', 'VZ', 'PID'])
    d = d[d.PID == -1]
    a, om = header['a'], header['Om']
    E = np.sqrt(om / a**3 + 1. - om)
    z_s = d.Z.values + d.VZ.values / (a * 100. * E)
    return np.column_stack([d.X.values, d.Y.values, z_s]) % BOX


def complete(path):
    return path.exists() and path.stat().st_size > 1_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=500)
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--nchunks', type=int, default=1)
    parser.add_argument('--nthreads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    outdir = ROOT / 'data' / 'quijote' / args.set
    outdir.mkdir(parents=True, exist_ok=True)
    boxsize = np.full(3, BOX)
    clustering = None
    for i in list(range(args.start, args.stop))[args.chunk::args.nchunks]:
        path = QUIJOTE / args.set / str(i) / 'out_3_pid.list'
        out = outdir / f'{i}.npz'
        if out.exists() or not complete(path):
            continue
        pos = read_halos(path)
        result, clustering = astra_ds.measure_positions(pos, boxsize, f'{args.set}/{i}', nside=1,
                                                        periodic_clustering=True, nthreads=args.nthreads,
                                                        keep_sample=0, clustering=clustering)
        result.update(set=args.set, realisation=i)
        tmp = out.with_suffix('.tmp.npz')
        np.savez(tmp, **result)
        os.replace(tmp, out)
        logging.info(f'saved {out}')


if __name__ == '__main__':
    main()
