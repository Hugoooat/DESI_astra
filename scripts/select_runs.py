#!/usr/bin/env python
"""
Lists of catalogues to measure (one line per run: cosmo hod space los).

* config/runs_hod_calibration.txt : extra c000 HOD draws around the HODs used
  for the Fisher analysis, to calibrate the response of the statistics to the
  HOD parameters (linear model). They are the --ncalib c000 draws closest to
  the centroid of the 9 Fisher HODs (fiducial + 4 pairs), in HOD parameters
  normalised to the prior range, so that they bracket the pair differences.
* config/runs_real_space.txt      : the 9 Fisher catalogues in real space, 3 lines of sight.
* config/runs_emulator.txt        : emulator training set, --nemu HOD draws per
  cosmology for every available cosmology (the draw closest to the prior
  centre, then random draws; the files are already a Latin hypercube).

    python scripts/select_runs.py [--ncalib 72] [--nemu 8] [--nchunks 3]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncalib', type=int, default=72)
    parser.add_argument('--nemu', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nchunks', type=int, default=3)
    args = parser.parse_args()
    sel = A.selection()
    lo, hi = np.array(sel['prior_lo']), np.array(sel['prior_hi'])

    fisher_runs = [(sel['fiducial']['cosmo'], sel['fiducial']['hod'])]
    fisher_runs += [(p[k]['cosmo'], p[k]['hod']) for p in sel['pairs'].values() for k in ('plus', 'minus')]

    # --- HOD calibration around the Fisher HODs
    hods0, p0 = read_all(0)
    p0 = (p0 - lo) / (hi - lo)
    fisher_p = []
    for c, h in fisher_runs:
        hc, pc = read_all(c) if c != 0 else (hods0, None)
        pc = p0 if c == 0 else (pc - lo) / (hi - lo)
        fisher_p.append(pc[list(hc).index(h)])
    fisher_p = np.array(fisher_p)
    centre = fisher_p.mean(0)
    d = np.linalg.norm(p0 - centre, axis=1)
    order = [i for i in np.argsort(d) if hods0[i] != sel['fiducial']['hod']][:args.ncalib]
    calib = hods0[order]
    span_calib = p0[order].max(0) - p0[order].min(0)
    span_fisher = fisher_p.max(0) - fisher_p.min(0)
    print(f'calibration: {len(calib)} c000 draws, distance to centroid {d[order].min():.2f}-{d[order].max():.2f}')
    print('  per-parameter span calib / Fisher HODs:', np.round(span_calib / span_fisher, 2))
    with open(ROOT / 'config' / 'runs_hod_calibration.txt', 'w') as f:
        for h in calib:
            f.write(f'0 {h} z z\n')

    # --- real space
    with open(ROOT / 'config' / 'runs_real_space.txt', 'w') as f:
        for los in 'zxy':
            for c, h in fisher_runs:
                f.write(f'{c} {h} r {los}\n')

    # --- emulator training set
    rng = np.random.default_rng(args.seed)
    cosmos = sorted(int(p.name[1:4]) for p in astra_ds.HOD_DIR.glob('c*_ph000'))
    n = 0
    with open(ROOT / 'config' / 'runs_emulator.txt', 'w') as f:
        for c in cosmos:
            hods, params = read_all(c)
            pn = (params - lo) / (hi - lo)
            first = int(np.argmin(np.linalg.norm(pn - 0.5, axis=1)))
            others = rng.choice([i for i in range(len(hods)) if i != first], size=args.nemu - 1, replace=False)
            for i in [first, *others]:
                if (c, int(hods[i])) in fisher_runs:
                    continue
                f.write(f'{c} {hods[i]} z z\n')
                n += 1
    print(f'emulator: {len(cosmos)} cosmologies, {n} runs')
    lines = (ROOT / 'config' / 'runs_emulator.txt').read_text().splitlines()
    for k in range(args.nchunks):   # independent lists, one per batch job
        (ROOT / 'config' / f'runs_emulator_{k}.txt').write_text('\n'.join(lines[k::args.nchunks]) + '\n')
    print('saved config/runs_{hod_calibration,real_space,emulator}.txt')


if __name__ == '__main__':
    main()
