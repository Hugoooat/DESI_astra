#!/usr/bin/env python
"""
Robustness of the ASTRA split to the numerical choices, on the fiducial
catalogue (c000, line of sight z, first `--nbox` subboxes).

1. Measurement tests: the quartile multipoles are re-measured with
     pad = 60 Mpc/h        (vs 30: periodic padding around each subbox)
     nrand_astra = 2       (vs 1: randoms per galaxy in the tessellation)
     thin = 2, 5           (galaxies randomly subsampled by 2 and 5)
   and compared to the baseline in units of the statistical error of one
   subbox (fiducial covariance), bin by bin.
2. Estimator stability under thinning: r of the *same* galaxies computed with
   the full sample and with the sample thinned by 2, 5, 10 (randoms thinned
   accordingly): Spearman correlation and quartile retention.

    python scripts/robustness.py [--nbox 8] [--nthreads 128]
Outputs: data/robustness.npz, plots/robustness_*.png
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import astra_ds
import analysis as A

TESTS = {'pad 60': dict(pad=60.), 'nrand 2': dict(nrand_astra=2), 'thin 2': dict(thin=2), 'thin 5': dict(thin=5)}
STATS = [f'{m}_rand_cross_q{q}' for m in ('astra', 'ds') for q in (1, 4)] + \
        [f'{m}_data_cross_q{q}' for m in ('astra', 'ds') for q in (1, 4)]


def measurement_tests(nbox, nthreads):
    path = astra_ds.hod_path(A.selection()['fiducial']['cosmo'], A.selection()['fiducial']['hod'])
    boxes = list(range(nbox))
    out = {'baseline': astra_ds.measure_catalog(path, subboxes=boxes, nthreads=nthreads, keep_sample=0)}
    for name, kw in TESTS.items():
        out[name] = astra_ds.measure_catalog(path, subboxes=boxes, nthreads=nthreads, keep_sample=0, **kw)
    return out


def thinning_test(nbox, factors=(2, 5, 10), seed=42):
    """Spearman(r_full, r_thin) and fraction of galaxies keeping their quartile."""
    sel = A.selection()['fiducial']
    galaxies, boxsize = astra_ds.load_catalog(astra_ds.hod_path(sel['cosmo'], sel['hod']))
    randoms = astra_ds.uniform_randoms(len(galaxies), boxsize, seed)
    corners, size = astra_ds.subbox_corners(boxsize)
    rng = np.random.default_rng(seed)
    res = {f: {'spearman': [], 'retention': []} for f in factors}
    for ibox in range(nbox):
        _, gal, gcore = astra_ds.extract(galaxies, corners[ibox], size, boxsize, 30.)
        _, ran, _ = astra_ds.extract(randoms, corners[ibox], size, boxsize, 30.)
        r_full = astra_ds.astra_r(gal, ran)[0]
        for f in factors:
            kg = rng.random(len(gal)) < 1. / f
            kr = rng.random(len(ran)) < 1. / f
            r_thin = astra_ds.astra_r(gal[kg], ran[kr])[0]
            core = gcore[kg]
            a, b = r_full[kg][core], r_thin[core]
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            res[f]['spearman'].append(np.corrcoef(ra, rb)[0, 1])
            qa, qb = astra_ds.quantile_labels(a, rng), astra_ds.quantile_labels(b, rng)
            res[f]['retention'].append([np.mean(qb[qa == q] == q) for q in range(4)])
    return {f: {k: np.array(v) for k, v in d.items()} for f, d in res.items()}


def plot_measurements(meas, sigma, s, out):
    fig, axes = plt.subplots(2, len(STATS) // 2, figsize=(3.2 * len(STATS) // 2, 6), sharex=True, sharey=True)
    base = meas['baseline']
    for ax, stat in zip(axes.flat, STATS):
        for name, color in zip(TESTS, ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']):
            diff = (meas[name][stat][:, 0] - base[stat][:, 0]).mean(0) / sigma[stat]
            ax.plot(s, diff, color=color, lw=1.4, label=name)
        ax.axhspan(-0.2, 0.2, color='0.9')
        ax.axhline(0, color='0.5', lw=0.6)
        ax.set_title(stat.replace('_', ' '), fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\Delta\xi_0\,/\,\sigma_{\rm subbox}$')
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('Change of the monopoles w.r.t. the baseline (pad 30, nrand 1, full sample), '
                 'in units of the 1-subbox error; grey band: $\\pm0.2\\sigma$', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'saved {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbox', type=int, default=8)
    parser.add_argument('--nthreads', type=int, default=8)
    args = parser.parse_args()
    (ROOT / 'plots').mkdir(exist_ok=True)

    thin = thinning_test(args.nbox)
    print('\nASTRA r of the same galaxies, full vs thinned sample (mean over subboxes)')
    for f, d in thin.items():
        print(f'  thin {f:2d}: Spearman {d["spearman"].mean():.3f}   '
              f'quartile retention Q1..Q4 {np.round(d["retention"].mean(0), 2)}')

    meas = measurement_tests(args.nbox, args.nthreads)
    meta, fstats, _ = A.load(A.selection()['fiducial']['cosmo'], A.selection()['fiducial']['hod'], los=['z'])
    sigma = {stat: fstats[stat][:, 0].std(0, ddof=1) for stat in STATS}
    print('\nmax |Δξ0| / σ_subbox over s > 20 Mpc/h (mean of the tested subboxes)')
    keep = meta['s'] > 20
    for name in TESTS:
        line = '  '.join(f'{st.replace("_cross", "")}: '
                         f'{np.abs((meas[name][st][:, 0] - meas["baseline"][st][:, 0]).mean(0) / sigma[st])[keep].max():.2f}'
                         for st in STATS)
        print(f'  {name:8s} {line}')
    plot_measurements(meas, sigma, meta['s'], ROOT / 'plots' / 'robustness_measurements.png')

    save = {f'thin{f}_{k}': v for f, d in thin.items() for k, v in d.items()}
    for name, m in meas.items():
        for st in STATS + ['gg']:
            save[f'{name}/{st}'] = m[st]
    save['s'] = meta['s']
    np.savez(A.DATA / 'robustness.npz', **save)
    print(f'saved {A.DATA / "robustness.npz"}')


if __name__ == '__main__':
    main()
