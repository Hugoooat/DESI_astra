#!/usr/bin/env python
"""
Measure ASTRA-split and density-split clustering in the 64 subboxes of one
HOD catalogue.

    python scripts/run_catalog.py --cosmo 0 --hod 484 [--space z] [--subboxes 0 1]

Output: data/c{cosmo}_hod{hod}_{space}.npz (see astra_ds.measure_catalog).
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import astra_ds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cosmo', type=int, required=True)
    parser.add_argument('--hod', type=int, required=True)
    parser.add_argument('--space', choices=['z', 'r'], default='z')
    parser.add_argument('--subboxes', type=int, nargs='*', default=None, help='default: all 64')
    parser.add_argument('--nthreads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    parser.add_argument('--outdir', type=Path, default=ROOT / 'data')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    path = astra_ds.hod_path(args.cosmo, args.hod)
    result = astra_ds.measure_catalog(path, space=args.space, subboxes=args.subboxes,
                                      nthreads=args.nthreads)
    result.update(cosmo=args.cosmo, hod=args.hod, space=args.space)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f'c{args.cosmo:03d}_hod{args.hod:03d}_{args.space}{args.tag}.npz'
    np.savez(out, **result)
    logging.info(f'saved {out}')


if __name__ == '__main__':
    main()
