#!/usr/bin/env python
"""
ASTRA split vs density split: multipoles of the quartile statistics for one
catalogue, mean +/- 1 sigma (scatter over subboxes = error of one 500 Mpc/h box).

    python scripts/plot_multipoles.py data/c000_hod239_z.npz [--pop rand]

Figures (in plots/):
    multipoles_{tag}_{pop}.png   rows: ell = 0, 2 ; columns: ASTRA cross,
                                 DS cross, ASTRA auto, DS auto
    environment_{tag}.png        transition matrix ASTRA -> DS quartiles and
                                 r vs delta for a sample of galaxies
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#3b4cc0', '#7aa6f7', '#f4a582', '#b2182b']   # Q1 (underdense) -> Q4 (overdense)
METHOD_LABELS = {'astra': 'ASTRA split', 'ds': 'Density split'}
POP_LABELS = {'data': 'galaxy quartiles', 'rand': 'random (query-point) quartiles'}
KIND_LABELS = {'cross': r'Q$_i$ $\times$ galaxies', 'auto': r'Q$_i$ $\times$ Q$_i$'}


def plot_multipoles(d, pop, tag, outdir):
    s = d['s']
    ells = list(d['ells'])
    panels = [(m, k) for k in ('cross', 'auto') for m in ('astra', 'ds')]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for col, (method, kind) in enumerate(panels):
        for row, ell in enumerate((0, 2)):
            ax = axes[row, col]
            il = ells.index(ell)
            for q in range(4):
                x = d[f'{method}_{pop}_{kind}_q{q + 1}'][:, il]   # (nbox, ns)
                mean, std = x.mean(0), x.std(0, ddof=1)
                ax.plot(s, s**2 * mean, color=COLORS[q], lw=1.6, label=f'Q{q + 1}')
                ax.fill_between(s, s**2 * (mean - std), s**2 * (mean + std), color=COLORS[q], alpha=0.2)
            g = d['gg'][:, il]
            ax.plot(s, s**2 * g.mean(0), color='k', lw=1, ls='--', label='galaxies' if col == 0 else None)
            ax.axhline(0, color='0.5', lw=0.6)
            if row == 0:
                ax.set_title(f'{METHOD_LABELS[method]} — {KIND_LABELS[kind]}', fontsize=11)
            else:
                ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
            if col == 0:
                ax.set_ylabel(rf'$s^2\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
    axes[0, 0].legend(fontsize=9, frameon=False)
    # same y-range for ASTRA and DS so that they can be compared by eye
    for row in range(2):
        for c in (0, 2):
            lims = [axes[row, c].get_ylim(), axes[row, c + 1].get_ylim()]
            lim = (min(l[0] for l in lims), max(l[1] for l in lims))
            axes[row, c].set_ylim(lim)
            axes[row, c + 1].set_ylim(lim)
    nbox = d['gg'].shape[0]
    fig.suptitle(f'{tag} — {POP_LABELS[pop]} — mean $\\pm1\\sigma$ over {nbox} subboxes of 500 $h^{{-1}}$Mpc',
                 fontsize=12)
    fig.tight_layout()
    out = outdir / f'multipoles_{tag}_{pop}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'saved {out}')


def plot_environment(d, tag, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    for ax, pop in zip(axes[:2], ('data', 'rand')):
        t = d[f'transition_{pop}'].mean(0)
        im = ax.imshow(t, vmin=0, vmax=1, cmap='Blues')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{t[i, j]:.2f}', ha='center', va='center',
                        color='w' if t[i, j] > 0.5 else 'k')
        ax.set_xticks(range(4), [f'DS Q{q + 1}' for q in range(4)])
        ax.set_yticks(range(4), [f'ASTRA Q{q + 1}' for q in range(4)])
        rho = d[f'spearman_{pop}'].mean()
        ax.set_title(f'{POP_LABELS[pop]}\nP(DS quartile | ASTRA quartile), Spearman $\\rho$={rho:.2f}', fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.046, label='fraction')
    ax = axes[2]
    ax.scatter(d['sample_delta'], d['sample_r'] + np.random.default_rng(0).normal(0, 0.01, len(d['sample_r'])),
               s=2, alpha=0.4, color='k')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlabel(r'density-split contrast $\delta_{R=10}$')
    ax.set_ylabel('ASTRA $r$')
    ax.set_title('galaxies of one subbox', fontsize=10)
    out = outdir / f'environment_{tag}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=Path, nargs='+')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'plots')
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True)
    for fn in args.files:
        d = dict(np.load(fn))
        tag = fn.stem
        for pop in ('rand', 'data'):
            plot_multipoles(d, pop, tag, args.outdir)
        plot_environment(d, tag, args.outdir)


if __name__ == '__main__':
    main()
