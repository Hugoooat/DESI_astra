#!/usr/bin/env python
"""
Figures following the layout of the first version of the paper (itself modelled
on the density-split Fisher analysis of Paillas et al. 2023), redone with the
baseline products: official-box covariance, emulator derivatives at the
official fiducial point, HOD marginalised. One figure per method where the
first version had density split only.

  multipoles   quartile multipoles Q1..Q4, cross/auto x monopole/quadrupole (one figure per method) + 2PCF
  derivatives  normalised derivatives of the Q1..Q4 cross monopoles (one figure per method)
  quartiles    Fisher contours of individual quartiles, all quartiles and the 2PCF (one per method)
  autocross    Fisher contours, cross-correlations only vs cross + auto (both methods)
  smin         log10 sigma as a function of the minimum scale
  constraints  Fisher contours 2PCF, Q1, Q4, all quartiles, all + 2PCF (one per method)

    python scripts/paper_figures_v1style.py [--only multipoles smin ...]
Outputs: plots/v1style_*.png
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
sys.path.insert(0, str(ROOT / 'scripts'))
import analysis as A
import emulator as E

PLOTS = ROOT / 'plots'
QCOL = ['#1f77b4', '#ff4f6d', '#2ca02c', '#c9a800']            # Q1..Q4, as in the first version
NAME = {'astra': 'ASTRA split', 'ds': 'Density split'}
PTEX = {'omega_b': r'\omega_b', 'omega_cdm': r'\omega_{\rm cdm}', 'n_s': 'n_s', 'sigma8': r'\sigma_8', 'w0': 'w_0', 'wa': 'w_a'}
PARAMS = ['omega_b', 'omega_cdm', 'n_s', 'sigma8']


def st(m, kind, i, pop='rand'):
    return f'{m}_{pop}_{kind}_q{i}'


def fig_multipoles():
    sel = A.selection()['fiducial']
    meta, d, _ = A.load(sel['cosmo'], sel['hod'], space='z')
    s, ells = meta['s'], list(meta['ells'])
    for m in ('astra', 'ds'):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for col, kind in enumerate(('cross', 'auto')):
            for row, ell in enumerate((0, 2)):
                ax = axes[row, col]
                for i in range(1, 5):
                    x = d[st(m, kind, i)][:, ells.index(ell)]
                    mu, sd = x.mean(0), x.std(0, ddof=1)
                    ax.plot(s, s**2 * mu, color=QCOL[i - 1], label=f'Q{i}')
                    ax.fill_between(s, s**2 * (mu - sd), s**2 * (mu + sd), color=QCOL[i - 1], alpha=0.2)
                if kind == 'cross':
                    g = d['gg'][:, ells.index(ell)].mean(0)
                    ax.plot(s, s**2 * g, 'k--', lw=1, label='2PCF')
                ax.axhline(0, color='0.6', lw=0.5)
                ax.set_title(f'{kind.capitalize()} {"monopole" if ell == 0 else "quadrupole"}')
                ax.set_xlabel(r'$s$ [$h^{-1}$ Mpc]')
                ax.set_ylabel(r'$s^2\xi_%d$' % ell)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(f'{NAME[m]} multipoles — redshift space (random quartiles; mean and scatter of 192 subvolumes)')
        fig.tight_layout()
        fig.savefig(PLOTS / f'v1style_multipoles_{m}.png', dpi=140)
        plt.close(fig)


def fig_derivatives(fc):
    from mcmc import covariance_samples
    from fisher_emulator import INPUT_OF
    s = E.Emulator.load('gg', fc.models_dir).s
    ns = len(s)
    for m in ('astra', 'ds'):
        nr = (len(PARAMS) + 1) // 2
        fig, axes = plt.subplots(nr, 2, figsize=(12, 5 * nr))
        for ax, p in zip(axes.flat, PARAMS):
            ip = E.COSMO_INPUTS.index(INPUT_OF[p])
            for i in range(1, 5):
                name = st(m, 'cross', i)
                sig = covariance_samples('covbox', [name]).std(0, ddof=1)[:ns]
                der = fc.jac(name).mean(0)[:ns, ip] * fc.x0[ip] / sig      # d xi_0 / d ln theta, in sigma
                ax.plot(s, der, color=QCOL[i - 1], label=f'Q{i}')
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_title(rf'$\partial\xi_0/\partial\ln {PTEX[p]}$')
            ax.set_xlabel(r'$s$ [$h^{-1}$ Mpc]')
            ax.set_ylabel(r'derivative / $\sigma$')
            ax.legend(fontsize=8)
        fig.suptitle(f'Normalised derivatives of the quartile-galaxy cross monopoles — {NAME[m]}\n'
                     r'(emulator at the official fiducial point, in units of the error of one $(500\,h^{-1}$Mpc$)^3$ volume)')
        fig.tight_layout()
        fig.savefig(PLOTS / f'v1style_derivatives_{m}.png', dpi=140)
        plt.close(fig)


def triangle(fc, vectors, colors, styles, title, out, limits_from='2PCF'):
    from getdist.gaussian_mixtures import GaussianND
    from getdist import plots
    from fisher_emulator import INPUT_OF
    fid = [fc.x0[E.COSMO_INPUTS.index(INPUT_OF[p])] for p in PARAMS]
    covs = {k: fc(v)['cov_marg'] for k, v in vectors.items()}
    gs = [GaussianND(fid, c, names=PARAMS, labels=[PTEX[p] for p in PARAMS], label=k) for k, c in covs.items()]
    sref = np.sqrt(np.diag(covs[limits_from])) if limits_from else None
    limits = {p: (f - 3 * s, f + 3 * s) for p, f, s in zip(PARAMS, fid, sref)} if limits_from else None
    g = plots.get_subplot_plotter(width_inch=9)
    g.settings.legend_fontsize = 12
    g.settings.axes_labelsize = 16
    g.triangle_plot(gs, filled=False, contour_colors=colors, contour_ls=styles, contour_lws=[2.5] * len(gs),
                    markers=dict(zip(PARAMS, fid)), param_limits=limits,
                    legend_loc='upper right')
    g.fig.suptitle(title, fontsize=14, y=1.02)
    g.export(str(PLOTS / out))


def fig_quartiles(fc):
    for m in ('astra', 'ds'):
        vec = {f'Q{i}': [st(m, 'cross', i)] for i in range(1, 5)}
        vec['Q1-Q4'] = [st(m, 'cross', i) for i in range(1, 5)]
        vec['2PCF'] = ['gg']
        triangle(fc, vec, QCOL + ['#c051b8', '0.55'], ['-'] * 5 + ['--'],
                 f'Constraints from individual and combined {NAME[m]} quartiles (cross-correlations)',
                 f'v1style_quartiles_{m}.png')


def fig_autocross(fc):
    vec = {'ASTRA cross': [st('astra', 'cross', i) for i in (1, 4)],
           'ASTRA cross + auto': [st('astra', k, i) for k in ('cross', 'auto') for i in (1, 4)],
           'DS cross': [st('ds', 'cross', i) for i in (1, 4)],
           'DS cross + auto': [st('ds', k, i) for k in ('cross', 'auto') for i in (1, 4)]}
    triangle(fc, vec, ['#d95f02', '#d95f02', '#1b9e77', '#1b9e77'], ['--', '-', '--', '-'],
             'Cross-correlations only vs cross + auto-correlations (Q1+Q4)', 'v1style_autocross.png',
             limits_from='DS cross')


def fig_smin(fc):
    vec = {'2PCF': ['gg'], 'ASTRA': [st('astra', 'cross', i) for i in (1, 4)],
           'DS': [st('ds', 'cross', i) for i in (1, 4)],
           'ASTRA+DS': [st(m, 'cross', i) for m in ('astra', 'ds') for i in (1, 4)],
           '2PCF+ASTRA+DS': ['gg'] + [st(m, 'cross', i) for m in ('astra', 'ds') for i in (1, 4)]}
    col = {'2PCF': '#4c72b0', 'ASTRA': '#d95f02', 'DS': '#1b9e77', 'ASTRA+DS': '#7570b3', '2PCF+ASTRA+DS': 'k'}
    smins = [20., 30., 40., 50., 60., 70.]   # beyond, the 2PCF has fewer bins than parameters
    res = {k: np.array([fc(v, smin)['marg'][0] for smin in smins]) for k, v in vec.items()}
    nr = (len(PARAMS) + 1) // 2
    fig, axes = plt.subplots(nr, 2, figsize=(10, 4 * nr))
    for k, (ax, p) in enumerate(zip(axes.flat, PARAMS)):
        for lab, r in res.items():
            ax.plot(smins, np.log10(r[:, k]), 'o-', color=col[lab], lw=2, label=lab)
        ax.set_title(f'${PTEX[p]}$')
        ax.set_xlabel(r'$s_{\rm min}$ [$h^{-1}$Mpc]')
        ax.set_ylabel(r'$\log_{10}(\sigma)$')
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Marginalised errors as a function of the minimum scale (Q1+Q4 cross-correlations, baseline)')
    fig.tight_layout()
    fig.savefig(PLOTS / 'v1style_smin.png', dpi=140)
    plt.close(fig)


def fig_constraints(fc):
    for m in ('astra', 'ds'):
        allq = [st(m, 'cross', i) for i in range(1, 5)]
        vec = {'2PCF': ['gg'], 'Q1': [st(m, 'cross', 1)], 'Q4': [st(m, 'cross', 4)], 'Q1-Q4': allq,
               'Q1-Q4 + 2PCF': ['gg'] + allq}
        triangle(fc, vec, ['#4c72b0', '#55a868', '#c44e52', '#ccb974', '0.2'], ['-'] * 5,
                 f'Fisher constraints, {NAME[m]} ($s_{{\\rm min}} = 20\\,h^{{-1}}$Mpc, HOD marginalised)',
                 f'v1style_constraints_{m}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', default=None)
    parser.add_argument('--params', nargs='+', default=None)
    args = parser.parse_args()
    global PARAMS
    if args.params:
        import compare_methods
        PARAMS = compare_methods.PARAMS = list(args.params)
    todo = args.only or ['multipoles', 'derivatives', 'quartiles', 'autocross', 'smin', 'constraints']
    fc = None
    if set(todo) - {'multipoles'}:
        from compare_methods import Forecaster
        fc = Forecaster(ROOT / 'models_pairs', '_pairs', compress=True, cov='covbox', fiducial='official')
    for name in todo:
        f = globals()[f'fig_{name}']
        f() if name == 'multipoles' else f(fc)
        print(f'{name} done', flush=True)


if __name__ == '__main__':
    main()
