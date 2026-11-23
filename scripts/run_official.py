#!/usr/bin/env python
"""
ASTRA split and density split on the official DESI AbacusHOD LRG mocks at
z = 0.5 (one best-fit HOD, `velbias` model), for an independent covariance:

* covbox : the 1786 AbacusSummit small boxes (500 Mpc/h, c000, phases 3000+),
           each treated as one open subvolume, exactly like the subboxes of the
           main analysis (Landy-Szalay with geometry randoms, periodic padding
           for the tessellation);
* cubicbox : the 25 base boxes of c000 (phases 0-24), cut into 64 subboxes.

Positions x, y, z are used as provided (z already includes the RSD along z).

    python scripts/run_official.py --kind covbox [--chunk 0 --nchunks 8]
Outputs: data/official/{kind}_{phase}.npz
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import fitsio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import astra_ds

MOCKS = Path('/global/cfs/cdirs/desi/cosmosim/AbacusHOD_mocks/v1')
NAME = 'catalog_rsd_xi2d_lrg_main_z0.5_velbias_mockcov'


def files(kind):
    if kind == 'covbox':
        out = sorted((MOCKS / 'CovBox/LRG/z0.500').glob(f'{NAME}_ph*.fits'))
        return [(f, f.stem.split('_ph')[-1]) for f in out]
    out = sorted((MOCKS / 'CubicBox/LRG/z0.500').glob('AbacusSummit_base_c000_ph*'))
    return [(d / f'{NAME}.fits', d.name.split('_ph')[-1]) for d in out if (d / f'{NAME}.fits').exists()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['covbox', 'cubicbox'], required=True)
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--nchunks', type=int, default=1)
    parser.add_argument('--max', type=int, default=None)
    parser.add_argument('--nthreads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    outdir = ROOT / 'data' / 'official'
    outdir.mkdir(parents=True, exist_ok=True)
    todo = files(args.kind)[args.chunk::args.nchunks][:args.max]
    boxsize = np.full(3, 500. if args.kind == 'covbox' else 2000.)
    nside = 1 if args.kind == 'covbox' else 4
    clustering = None
    for path, phase in todo:
        out = outdir / f'{args.kind}_ph{phase}.npz'
        if out.exists():
            continue
        d = fitsio.read(str(path), columns=['x', 'y', 'z'])
        pos = np.column_stack([d['x'], d['y'], d['z']]).astype('f8') + boxsize / 2.   # [-L/2, L/2) -> [0, L)
        result, clustering = astra_ds.measure_positions(pos, boxsize, path.name, nside=nside,
                                                        nthreads=args.nthreads, keep_sample=0,
                                                        clustering=clustering)
        result.update(kind=args.kind, phase=phase, source_file=str(path))
        tmp = out.with_suffix('.tmp.npz')
        np.savez(tmp, **result)
        os.replace(tmp, out)
        logging.info(f'saved {out}')


if __name__ == '__main__':
    main()
