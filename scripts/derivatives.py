#!/usr/bin/env python
"""
Paired central-difference derivatives of every statistic.

For each parameter, with the HOD pair chosen by select_hods.py:

    d xi / d theta = [ xi(c+, subbox i) - xi(c-, subbox i) ] / (theta+ - theta-)

computed subbox by subbox *before* averaging: the +/- simulations share the
same initial phases (ph000) and the same random points, so cosmic variance
and random noise cancel in the difference. Mean over the 64 subboxes is the
derivative, scatter / sqrt(64) its error.

    python scripts/derivatives.py [--space z]   ->  data/derivatives_{space}.npz
                                                    plots/derivative_{param}_{space}.png
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
COLORS = ['#3b4cc0', '#7aa6f7', '#f4a582', '#b2182b']
PARAM_COLUMNS = {'omega_b': 'omega_b', 'omega_cdm': 'omega_cdm', 'n_s': 'n_s', 'sigma8': 'sigma8_m'}
PARAM_LABELS = {'omega_b': r'\omega_b', 'omega_cdm': r'\omega_{\rm cdm}', 'n_s': r'n_s', 'sigma8': r'\sigma_8'}


def cosmology_table():
    df = pd.read_csv(ROOT / 'config' / 'cosmologies.csv', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df.index = df['root'].str.extract(r'cosm(\d+)')[0].astype(int)
    return df


def load(cosmo, hod, space):
    return dict(np.load(DATA / f'c{cosmo:03d}_hod{hod:03d}_{space}.npz'))


def plot(param, s, ells, der, names, space, title):
    families = [('gg', None)] + [(f'{m}_{p}_{k}', m) for m in ('astra', 'ds') for p in ('rand', 'data')
                                 for k in ('cross',)]
    families += [(f'{m}_rand_auto', m) for m in ('astra', 'ds')]
    fig, axes = plt.subplots(2, len(families), figsize=(3.6 * len(families), 6.5), sharex=True)
    for col, (fam, method) in enumerate(families):
        stems = ['gg'] if fam == 'gg' else [f'{fam}_q{q + 1}' for q in range(4)]
        for row, ell in enumerate((0, 2)):
            ax = axes[row, col]
            il = list(ells).index(ell)
            for iq, stem in enumerate(stems):
                d = der[stem][:, il]
                mean, err = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
                color = 'k' if stem == 'gg' else COLORS[iq]
                ax.plot(s, s**2 * mean, color=color, lw=1.5, label=None if stem == 'gg' else f'Q{iq + 1}')
                ax.fill_between(s, s**2 * (mean - err), s**2 * (mean + err), color=color, alpha=0.25)
            ax.axhline(0, color='0.5', lw=0.6)
            if row == 0:
                ax.set_title('galaxy 2PCF' if fam == 'gg' else fam.replace('_', ' '), fontsize=10)
            else:
                ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
            if col == 0:
                ax.set_ylabel(rf'$s^2\,\partial\xi_{ell}/\partial {PARAM_LABELS[param]}$')
    axes[0, 1].legend(fontsize=8, frameon=False)
    fig.suptitle(title + r' — mean $\pm$ error of the mean over 64 paired subboxes', fontsize=11)
    fig.tight_layout()
    out = ROOT / 'plots' / f'derivative_{param}_{space}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'saved {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', default='z')
    args = parser.parse_args()
    (ROOT / 'plots').mkdir(exist_ok=True)
    selection = json.loads((ROOT / 'config' / 'hod_selection.json').read_text())
    table = cosmology_table()
    out = {}
    for param, pair in selection['pairs'].items():
        plus, minus = pair['plus'], pair['minus']
        try:
            dp, dm = load(plus['cosmo'], plus['hod'], args.space), load(minus['cosmo'], minus['hod'], args.space)
        except FileNotFoundError as exc:
            print(f'skip {param}: {exc}')
            continue
        col = PARAM_COLUMNS[param]
        step = table.loc[plus['cosmo'], col] - table.loc[minus['cosmo'], col]
        names = list(dp['stat_names'])
        der = {name: (dp[name] - dm[name]) / step for name in names}   # (nbox, nell, ns)
        for name in names:
            out[f'{param}/{name}'] = der[name]
        out[f'{param}/step'] = step
        title = (f'$\\partial/\\partial {PARAM_LABELS[param]}$: c{plus["cosmo"]}_hod{plus["hod"]:03d} '
                 f'$-$ c{minus["cosmo"]}_hod{minus["hod"]:03d}, $\\Delta$={step:.4g}')
        plot(param, dp['s'], dp['ells'], der, names, args.space, title)
    if out:
        out['s'], out['ells'] = dp['s'], dp['ells']
        np.savez(DATA / f'derivatives_{args.space}.npz', **out)
        print(f'saved {DATA / f"derivatives_{args.space}.npz"}')


if __name__ == '__main__':
    main()
