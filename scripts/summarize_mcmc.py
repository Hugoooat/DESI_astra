#!/usr/bin/env python
"""
Summary of the emulator inference runs (scripts/mcmc.py), one subbox
(0.125 (Gpc/h)^3) of data with the official small-box covariance:

* coverage: fraction of runs whose posterior mean lies within 1 sigma of the
  truth (expected ~68% for calibrated posteriors), per parameter and vector;
* constraining power: median posterior standard deviation, compared with the
  standard deviation of the (uniform) prior;
* figure: posteriors of 2PCF, ASTRA, DS and ASTRA + DS for one subbox of the
  official DESI mock (true cosmology c000 and the official best-fit HOD).

    python scripts/summarize_mcmc.py [--suffix _covbox_v1]
Outputs: data/mcmc_summary.json, plots/mcmc_coverage.png, plots/mcmc_official.png
"""

import sys
import json
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ['2PCF', 'ASTRA', 'DS', 'ASTRA+DS', '2PCF+ASTRA+DS']
COLORS = {'2PCF': 'k', 'ASTRA': '#d95f02', 'DS': '#1b9e77', 'ASTRA+DS': '#7570b3', '2PCF+ASTRA+DS': '#e7298a'}
PARAMS = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
TEX = {'omega_b': r'\omega_b', 'omega_cdm': r'\omega_{\rm cdm}', 'sigma8_m': r'\sigma_8', 'n_s': 'n_s',
       'alpha_s': r'\alpha_s', 'N_ur': r'N_{\rm ur}', 'w0_fld': 'w_0', 'wa_fld': 'w_a'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='_covbox_v1')
    args = parser.parse_args()
    chains = sorted(glob(str(ROOT / 'data' / 'chains' / f'*{args.suffix}.npz')))
    stats = {kind: {v: defaultdict(list) for v in VECTORS} for kind in ('cosmo', 'official')}
    prior_sd = None
    for f in chains:
        name = Path(f).stem.replace(args.suffix, '')
        vec = next(v for v in sorted(VECTORS, key=len, reverse=True) if name.endswith('_' + v))
        kind = name.split('_')[0]
        d = np.load(f)
        prior_sd = (d['hi'] - d['lo']) / np.sqrt(12)
        for k, p in enumerate(PARAMS):
            m, s = d['chain'][:, k].mean(), d['chain'][:, k].std()
            stats[kind][vec][p].append(((m - d['truth'][k]) / s, s))
    summary = {}
    for kind in stats:
        print(f'\n=== {kind}: fraction within 1 sigma (expected 68%) | median posterior sigma / prior sigma')
        print(f'{"vector":15s} N   ' + ' '.join(f'{p:>16s}' for p in PARAMS))
        for v in VECTORS:
            st = stats[kind][v]
            if not st[PARAMS[0]]:
                continue
            cells, summary[f'{kind}/{v}'] = [], {}
            for k, p in enumerate(PARAMS):
                b = np.array([x[0] for x in st[p]])
                s = np.median([x[1] for x in st[p]])
                summary[f'{kind}/{v}'][p] = {'coverage': float(np.mean(np.abs(b) < 1)), 'sigma': float(s),
                                             'sigma_over_prior': float(s / prior_sd[k]),
                                             'mean_bias_sigma': float(np.mean(b))}
                cells.append(f'{np.mean(np.abs(b) < 1):4.0%} {s / prior_sd[k]:5.2f}')
            print(f'{v:15s} {len(st[PARAMS[0]]):2d}  ' + ' '.join(f'{c:>16s}' for c in cells))
    (ROOT / 'data' / 'mcmc_summary.json').write_text(json.dumps(summary, indent=1))

    # figure 1: posterior width relative to the prior, per parameter and vector (test cosmologies)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.16
    for i, v in enumerate(VECTORS):
        key = f'cosmo/{v}'
        if key in summary:
            ax.bar(np.arange(len(PARAMS)) + (i - 2) * w, [summary[key][p]['sigma_over_prior'] for p in PARAMS], w,
                   color=COLORS[v], label=v)
    ax.axhline(1, color='0.5', ls='--', lw=0.8)
    ax.set_xticks(range(len(PARAMS)), [f'${TEX[p]}$' for p in PARAMS])
    ax.set_ylabel(r'$\sigma_{\rm posterior}/\sigma_{\rm prior}$')
    ax.legend(fontsize=8, frameon=False, ncol=5)
    ax.set_title('Emulator inference on held-out cosmologies, one $(500\\,h^{-1}$Mpc$)^3$ subbox: '
                 'posterior width relative to the prior (median of 16 runs)', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / 'mcmc_widths.png', dpi=140)

    # figure 2: posteriors for one subbox of the official mock
    try:
        from getdist import MCSamples, plots
        samples = []
        for v in ['2PCF', 'ASTRA', 'DS', 'ASTRA+DS']:
            fs = sorted(glob(str(ROOT / 'data' / 'chains' / f'official_000_0_{v}{args.suffix}.npz')))
            if fs:
                d = np.load(fs[0])
                samples.append(MCSamples(samples=d['chain'][:, :4], names=PARAMS[:4],
                                         labels=[TEX[p] for p in PARAMS[:4]], label=v,
                                         ranges={p: (d['lo'][k], d['hi'][k]) for k, p in enumerate(PARAMS[:4])}))
                truth = d['truth'][:4]
        if samples:
            g = plots.get_subplot_plotter(width_inch=8)
            g.triangle_plot(samples, filled=[False, True, True, False],
                            contour_colors=[COLORS[s.label] for s in samples],
                            markers={p: t for p, t in zip(PARAMS[:4], truth)})
            g.export(str(ROOT / 'plots' / 'mcmc_official.png'))
    except Exception as exc:
        print(f'triangle plot skipped: {exc}')
    print('\nsaved data/mcmc_summary.json, plots/mcmc_widths.png, plots/mcmc_official.png')


if __name__ == '__main__':
    main()
