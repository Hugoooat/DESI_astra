#!/usr/bin/env python
"""
Densify the emulator training set around the fiducial point where the Fisher
derivatives are taken: for c000 and every cosmology of the AbacusSummit
derivative grid (c100-c113), the --nlocal HOD draws closest to the fiducial
HOD (hod239), in HOD parameters normalised to the prior range.

    python scripts/select_local_runs.py [--nlocal 16]   ->  config/runs_local_{0,1}.txt
"""
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import astra_ds
import analysis as A
from select_hods import read_all

parser = argparse.ArgumentParser()
parser.add_argument('--nlocal', type=int, default=16)
parser.add_argument('--nchunks', type=int, default=2)
args = parser.parse_args()
sel = A.selection()
lo, hi = np.array(sel['prior_lo']), np.array(sel['prior_hi'])
h0 = A.hod_vector(sel['fiducial']['cosmo'], sel['fiducial']['hod'])
done = {tuple(map(int, f.stem.split('_')[0][1:].split('c')[-1:] + [f.stem.split('_')[1][3:]]))
        for f in A.DATA.glob('c*_hod*_z_losz.npz')}
cosmos = [0] + [c for c in range(100, 114) if (astra_ds.HOD_DIR / f'c{c:03d}_ph000').exists()]
lines = []
for c in cosmos:
    hods, params = read_all(c)
    d = np.linalg.norm((params - lo) / (hi - lo) - h0, axis=1)
    new = [int(h) for h in hods[np.argsort(d)] if (c, int(h)) not in done][:args.nlocal]
    lines += [f'{c} {h} z z' for h in new]
for k in range(args.nchunks):
    (ROOT / 'config' / f'runs_local_{k}.txt').write_text('\n'.join(lines[k::args.nchunks]) + '\n')
print(f'{len(cosmos)} cosmologies, {len(lines)} new runs -> config/runs_local_*.txt')
