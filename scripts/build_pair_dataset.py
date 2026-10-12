#!/usr/bin/env python
"""
Precise differences between catalogues, to train the emulator on (option b).

All catalogues share the initial conditions (ph000) and the random points,
so the difference of two full-box measurements is much more precise than
each of them: its error, estimated from the scatter of the 64 paired subbox
differences / 8, is a small fraction of the statistical error of one box.
We collect, for every statistic:

* cross-cosmology pairs of the AbacusSummit derivative grid: every catalogue
  of c+ with every catalogue of c- whose HOD is within --rmax of the fiducial
  HOD (7 pairs: omega_b, omega_cdm, n_s, alpha_s, w0, w0-wa, sigma8);
* within-cosmology pairs for c000 and the grid cosmologies: every catalogue
  with the one closest to the fiducial HOD of that cosmology (HOD response at
  fixed cosmology).

--holdout N keeps N local catalogues per grid cosmology out of every pair (and
lists them, so that train_emulator.py also removes them from the absolute
training set): they are used by validate_local.py to test the emulator on pair
differences it has never seen.

    python scripts/build_pair_dataset.py [--rmax 1.0] [--holdout 4]   ->  data/pair_dataset.npz
"""

import sys
import argparse
from pathlib import Path
from itertools import product

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A
import emulator as E

GRID = [(100, 101), (102, 103), (104, 105), (106, 107), (108, 109), (110, 111), (112, 113)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rmax', type=float, default=1.0)
    parser.add_argument('--holdout', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', default='pair_dataset.npz')
    args = parser.parse_args()
    sel = A.selection()
    h0 = A.hod_vector(sel['fiducial']['cosmo'], sel['fiducial']['hod'])
    d = np.load(A.DATA / 'emulator_dataset.npz')
    row = {(int(c), int(h)): i for i, (c, h) in enumerate(d['ids'])}
    stats = [str(s) for s in d['stat_names']]
    rng = np.random.default_rng(args.seed)

    # catalogues of each cosmology near the fiducial HOD
    cosmos = [0] + [c for p in GRID for c in p]
    near, holdout = {}, []
    for c in cosmos:
        cats = []
        for (cc, h), i in row.items():
            if cc == c:
                dist = np.linalg.norm(A.hod_vector(c, h) - h0)
                if dist <= args.rmax or c == 0:
                    cats.append((dist, h))
        cats.sort()
        hods = [h for _, h in cats]
        if c != 0 and args.holdout:
            out = list(rng.choice(hods[1:], size=min(args.holdout, len(hods) - 2), replace=False))
            holdout += [(c, int(h)) for h in out]
            hods = [h for h in hods if h not in out]
        near[c] = hods
    pairs = []
    for cp, cm in GRID:
        pairs += [((cp, a), (cm, b)) for a, b in product(near[cp], near[cm])]
    for c in cosmos:
        anchor = near[c][0]
        pairs += [((c, h), (c, anchor)) for h in near[c][1:]]
    print(f'{len(pairs)} pairs ({sum(1 for p in pairs if p[0][0] != p[1][0])} cross-cosmology), '
          f'{len(holdout)} held-out catalogues')

    # per-subbox data vectors of all the catalogues involved (line of sight z)
    needed = sorted({k for p in pairs for k in p} | set(holdout))
    sub = {}
    for k in needed:
        meta, st, ids = A.load(*k, los=['z'])
        sub[k] = {name: A.data_vector(meta, st, [name])[1] for name in stats}
    out = {'i': np.array([row[a] for a, _ in pairs]), 'j': np.array([row[b] for _, b in pairs]),
           'holdout': np.array([row[k] for k in holdout]), 'holdout_ids': np.array(holdout)}
    for name in stats:
        diff = np.array([sub[a][name] - sub[b][name] for a, b in pairs])        # (npairs, nsub, ndim)
        out[f'delta/{name}'] = diff.mean(1)
        out[f'sigma/{name}'] = diff.std(1, ddof=1) / np.sqrt(diff.shape[1])
    np.savez(A.DATA / args.out, **out)
    r = out['sigma/gg'] / E.statistical_error('gg')
    print(f'pair-difference error / error of one box (gg): median {np.median(r):.2f}')
    print(f'saved data/{args.out}')


if __name__ == '__main__':
    main()
