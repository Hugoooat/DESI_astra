#!/usr/bin/env python
"""
Readable versions of three paper figures (same data as before):

  emulator    emulator accuracy on held-out cosmologies: one panel per statistic,
              median and 68% range, both multipoles  -> plots/readable_emulator_validation.png
  mcmc        1D marginal posteriors of 4 cosmological parameters for one official
              subvolume, against the uniform prior    -> plots/readable_mcmc_official.png
  robustness  change of the Q1/Q4 cross monopoles under the numerical tests,
              one row per method                      -> plots/readable_robustness.png

    python scripts/readable_figures.py [--only emulator mcmc robustness]
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
import analysis as A

PLOTS = ROOT / 'plots'
DATA = ROOT / 'data'
COLV = {'2PCF': 'k', 'ASTRA': '#d95f02', 'DS': '#1b9e77', 'ASTRA+DS': '#7570b3'}


def fig_emulator():
    d = np.load(DATA / 'emulator_validation_pairs.npz')
    s = np.arange(25., 150., 10.)
    ns = len(s)
    stats = [('gg', '2PCF'), ('astra_rand_cross_q1', 'ASTRA Q1 x galaxies'), ('astra_rand_cross_q4', 'ASTRA Q4 x galaxies'),
             ('ds_rand_cross_q1', 'DS Q1 x galaxies'), ('ds_rand_cross_q4', 'DS Q4 x galaxies')]
    fig, axes = plt.subplots(1, len(stats), figsize=(16, 3.6), sharey=True)
    for ax, (st, title) in zip(axes, stats):
        x = np.abs(d[f'{st}/res'] / d[f'{st}/sigma'])                  # (n_test, 2 * ns)
        for ell, col in ((0, '#1f77b4'), (1, '#d62728')):
            y = x[:, ell * ns:(ell + 1) * ns]
            ax.plot(s, np.median(y, 0), color=col, lw=2, label=rf'$\ell={2 * ell}$ median')
            ax.fill_between(s, np.percentile(y, 16, 0), np.percentile(y, 84, 0), color=col, alpha=0.15,
                            label=rf'$\ell={2 * ell}$ 68% range')
        ax.axhline(1, color='0.4', ls=':', lw=1)
        ax.axhline(8, color='k', ls='--', lw=1)
        ax.text(148, 8.6, r'error of one $(500\,h^{-1}$Mpc$)^3$ volume', ha='right', fontsize=7)
        ax.set_yscale('log')
        ax.set_ylim(0.05, 15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r'$s$ [$h^{-1}$Mpc]')
    axes[0].set_ylabel(r'|emulator $-$ simulation| / $\sigma_{\rm box}$')
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle(r'Emulator error on 8 cosmologies never seen in training, in units of the error of a full $2\,h^{-1}$Gpc box',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / 'readable_emulator_validation.png', dpi=150)
    plt.close(fig)


def fig_mcmc(suffix='_covbox_v1'):
    names = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s']
    tex = [r'$\omega_b$', r'$\omega_{\rm cdm}$', r'$\sigma_8$', r'$n_s$']
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for v, col in COLV.items():
        d = np.load(DATA / 'chains' / f'official_000_0_{v}{suffix}.npz')
        idx = [list(d['labels']).index(n) if 'labels' in d.files and n in list(d['labels']) else k
               for k, n in enumerate(names)]
        for ax, k, i in zip(axes, range(4), idx):
            lo, hi = d['lo'][i], d['hi'][i]
            ax.hist(d['chain'][:, i], bins=np.linspace(lo, hi, 16), density=True, histtype='step', lw=2,
                    color=col, label=v)
            ax.axvline(d['truth'][i], color='0.3', ls=':', lw=1.2)
            if v == '2PCF':
                ax.axhspan(0, 1 / (hi - lo), color='0.85', zorder=0, label='uniform prior')
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            ax.locator_params(axis='x', nbins=4)
            ax.set_xlabel(tex[k], fontsize=12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=9, frameon=False)
    fig.suptitle(r'Marginal posteriors for one $(500\,h^{-1}$Mpc$)^3$ subvolume of the official mock (12 free HOD parameters); '
                 'dotted: true value, grey: uniform prior', fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PLOTS / 'readable_mcmc_official.png', dpi=150)
    plt.close(fig)


def fig_robustness():
    r = np.load(DATA / 'robustness.npz')
    sel = A.selection()['fiducial']
    meta, d, _ = A.load(sel['cosmo'], sel['hod'], space='z')
    s = meta['s']
    keep = s >= 20
    tests = [('pad 60', 'padding 30 $\\to$ 60 $h^{-1}$Mpc', '#1b9e77'), ('nrand 2', 'twice more ASTRA randoms', '#d95f02'),
             ('thin 2', 'galaxies thinned by 2', '#7570b3'), ('thin 5', 'galaxies thinned by 5', '#e7298a')]
    panels = [('rand_cross_q1', 'random Q1 (voids) x galaxies'), ('rand_cross_q4', 'random Q4 (clusters) x galaxies'),
              ('data_cross_q1', 'galaxy Q1 x galaxies'), ('data_cross_q4', 'galaxy Q4 x galaxies')]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True, sharey=True)
    for row, (m, mname) in enumerate((('astra', 'ASTRA split'), ('ds', 'Density split'))):
        for col, (key, title) in enumerate(panels):
            ax = axes[row, col]
            st = f'{m}_{key}'
            sigma = d[st][:, 0].std(0, ddof=1)
            base = r[f'baseline/{st}'][:, 0]
            for t, lab, c in tests:
                diff = (r[f'{t}/{st}'][:, 0] - base).mean(0) / sigma
                ax.plot(s[keep], diff[keep], color=c, lw=1.8, label=lab)
            ax.axhspan(-1, 1, color='0.85', zorder=0)
            ax.axhline(0, color='0.5', lw=0.6)
            ax.set_yscale('symlog', linthresh=1)
            ax.set_ylim(-60, 60)
            ax.set_yticks([-50, -10, -1, 0, 1, 10, 50])
            ax.set_yticklabels(['-50', '-10', '-1', '0', '1', '10', '50'])
            ax.grid(axis='y', color='0.9', lw=0.5)
            ax.set_title(f'{mname}: {title}', fontsize=9)
            if row == 1:
                ax.set_xlabel(r'$s$ [$h^{-1}$Mpc]')
        axes[row, 0].set_ylabel(r'$\Delta\xi_0/\sigma$')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    fig.suptitle(r'Change of the cross-correlation monopoles under numerical tests, in units of the error of one '
                 r'$(500\,h^{-1}$Mpc$)^3$ volume; grey: $\pm1\sigma$; axis linear within $\pm1\sigma$, logarithmic beyond', fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / 'readable_robustness.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', default=['emulator', 'mcmc', 'robustness'])
    for name in parser.parse_args().only:
        globals()[f'fig_{name}']()
        print(f'{name} done')


if __name__ == '__main__':
    main()
