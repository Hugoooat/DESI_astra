#!/usr/bin/env python
"""
Figures of the paper that follow the structure of the density-split Fisher
analysis of Paillas et al. (2023), with the baseline products of this work.

  fig_map          quartile maps (Q1, Q4) in a slab, ASTRA vs density split
  fig_env_hist     distributions of r (ASTRA) and delta (DS) per quartile
  fig_corr         correlation matrix of the 2PCF + ASTRA + DS data vector (official covariance)
  fig_derivatives  emulator derivatives normalised by the statistical error
  fig_triangle     Fisher contours: 2PCF, ASTRA, DS and combinations (baseline)
  fig_quartiles    information per quartile, cross vs cross + auto (baseline)
  fig_response     response of the mean environment of each quartile to cosmology (Quijote)

    python scripts/paper_figures.py [--only fig_map fig_corr ...]
Outputs: plots/paper_*.png
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
import astra_ds
import emulator as E

PLOTS = ROOT / 'plots'
CQ = ['#3b4cc0', '#7aa6f7', '#f4a582', '#b2182b']
COLV = {'2PCF': 'k', 'ASTRA': '#d95f02', 'DS': '#1b9e77', 'ASTRA+DS': '#7570b3', '2PCF+ASTRA+DS': '#e7298a'}
PARAMS = ['omega_b', 'omega_cdm', 'n_s', 'sigma8']
PTEX = {'omega_b': r'\omega_b', 'omega_cdm': r'\omega_{\rm cdm}', 'n_s': 'n_s', 'sigma8': r'\sigma_8', 'w0': 'w_0', 'wa': 'w_a'}


def q(method, pop='rand', kinds=('cross',), qs=(1, 4)):
    return [f'{method}_{pop}_{k}_q{i}' for k in kinds for i in qs]


def slab_environments(nthreads=8, seed=42, slab=20.):
    """ASTRA and DS quartiles of the random points and galaxies of one subbox of the fiducial catalogue."""
    gal, box = astra_ds.load_catalog(astra_ds.hod_path(0, 239))
    ran = astra_ds.uniform_randoms(len(gal), box, seed)
    mesh, cell = astra_ds.smoothed_density(gal, box, 5., 10., nthreads)
    corners, size = astra_ds.subbox_corners(box, 4)
    ig, g, gc = astra_ds.extract(gal, corners[0], size, box, 30.)
    ir, r, rc = astra_ds.extract(ran, corners[0], size, box, 30.)
    rg, rr = astra_ds.astra_r(g, r)
    dg = astra_ds.read_cic(mesh, cell, gal[ig][gc])
    dr = astra_ds.read_cic(mesh, cell, ran[ir][rc])
    rng = np.random.default_rng(seed)
    out = {'gal': g[gc], 'ran': r[rc], 'r_gal': rg[gc], 'r_ran': rr[rc], 'd_gal': dg, 'd_ran': dr, 'size': size}
    for key, v in [('qa_ran', rr[rc]), ('qd_ran', dr), ('qa_gal', rg[gc]), ('qd_gal', dg)]:
        out[key] = astra_ds.quantile_labels(v, rng)
    return out


def fig_map(env):
    z0 = env['size'][2] / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, key, title in zip(axes, ['qa_ran', 'qd_ran'], ['ASTRA split', 'Density split']):
        sg = np.abs(env['gal'][:, 2] - z0) < 10.
        sr = np.abs(env['ran'][:, 2] - z0) < 10.
        ax.scatter(*env['gal'][sg, :2].T, s=0.6, c='0.55', lw=0, label='galaxies')
        for qi, lab in [(0, 'Q1 (underdense)'), (3, 'Q4 (overdense)')]:
            m = sr & (env[key] == qi)
            ax.scatter(*env['ran'][m, :2].T, s=2.5, c=CQ[qi], lw=0, label=lab)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r'$x\ [h^{-1}{\rm Mpc}]$')
        ax.set_aspect('equal')
    axes[0].set_ylabel(r'$y\ [h^{-1}{\rm Mpc}]$')
    axes[0].legend(markerscale=4, fontsize=8, loc='upper right')
    fig.suptitle(r'Random points of the extreme quartiles in a $20\,h^{-1}$Mpc slab of one $(500\,h^{-1}$Mpc$)^3$ subvolume '
                 '(fiducial catalogue, redshift space)', fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / 'paper_map.png', dpi=150)
    plt.close(fig)


def fig_env_hist(env):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (v, lab, xl) in zip(axes, [(('r_ran', 'qa_ran'), 'ASTRA', '$r$'),
                                       (('d_ran', 'qd_ran'), 'Density split', r'$\delta_{R=10}$')]):
        x, qq = env[v[0]], env[v[1]]
        bins = np.linspace(-1, 1, 41) if lab == 'ASTRA' else np.linspace(-1, 3, 61)
        for qi in range(4):
            ax.hist(x[qq == qi], bins=bins, color=CQ[qi], alpha=0.7, label=f'Q{qi + 1}', histtype='stepfilled')
        ax.set_xlabel(xl)
        ax.set_title(f'{lab}: environment of the random points by quartile', fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / 'paper_env_hist.png', dpi=150)
    plt.close(fig)


def fig_corr():
    from mcmc import covariance_samples
    stats = ['gg'] + q('astra') + q('ds')
    x = covariance_samples('covbox', stats)
    c = np.corrcoef(x, rowvar=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1)
    n = x.shape[1] // len(stats)
    labels = ['2PCF', 'ASTRA Q1', 'ASTRA Q4', 'DS Q1', 'DS Q4']
    for k in range(1, len(stats)):
        ax.axhline(k * n - 0.5, color='k', lw=0.6)
        ax.axvline(k * n - 0.5, color='k', lw=0.6)
    ax.set_xticks([(k + 0.5) * n for k in range(len(stats))], labels, rotation=45, fontsize=8)
    ax.set_yticks([(k + 0.5) * n for k in range(len(stats))], labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Correlation matrix (1786 official small boxes); each block: $\\ell=0$ then $\\ell=2$, '
                 '$20\\leq s\\leq150\\,h^{-1}$Mpc', fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / 'paper_corr.png', dpi=150)
    plt.close(fig)


def fig_derivatives(fc):
    stats = ['gg'] + q('astra') + q('ds')
    from mcmc import covariance_samples
    sig = covariance_samples('covbox', stats).std(0, ddof=1)
    J = np.concatenate([fc.jac(st).mean(0) for st in stats])          # (ndim, 20)
    s = E.Emulator.load('gg', fc.models_dir).s
    ns = len(s)
    params = PARAMS
    from fisher_emulator import INPUT_OF
    fig, axes = plt.subplots(2, len(params), figsize=(3.8 * len(params), 6), sharex=True)
    styles = {'gg': ('k', '-'), 'astra_rand_cross_q1': ('#d95f02', '--'), 'astra_rand_cross_q4': ('#d95f02', '-'),
              'ds_rand_cross_q1': ('#1b9e77', '--'), 'ds_rand_cross_q4': ('#1b9e77', '-')}
    names = {'gg': '2PCF', 'astra_rand_cross_q1': 'ASTRA Q1', 'astra_rand_cross_q4': 'ASTRA Q4',
             'ds_rand_cross_q1': 'DS Q1', 'ds_rand_cross_q4': 'DS Q4'}
    for k, p in enumerate(params):
        ip = E.COSMO_INPUTS.index(INPUT_OF[p])
        width = {'omega_b': 0.001, 'omega_cdm': 0.005, 'n_s': 0.02, 'sigma8': 0.02, 'w0': 0.1, 'wa': 0.2}[p]
        for j, st in enumerate(stats):
            d = J[2 * ns * j:2 * ns * (j + 1), ip] * width / sig[2 * ns * j:2 * ns * (j + 1)]
            for ell in range(2):
                c, ls = styles[st]
                axes[ell, k].plot(s, d[ell * ns:(ell + 1) * ns], color=c, ls=ls, lw=1.4, label=names[st])
        for ell in range(2):
            axes[ell, k].axhline(0, color='0.6', lw=0.5)
        axes[0, k].set_title(rf'$\Delta {PTEX[p]} = {width}$', fontsize=10)
        axes[1, k].set_xlabel(r'$s\ [h^{-1}{\rm Mpc}]$')
    axes[0, 0].set_ylabel(r'$\Delta\xi_0/\sigma$')
    axes[1, 0].set_ylabel(r'$\Delta\xi_2/\sigma$')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('Change of the statistics for a small change of each parameter (emulator, official fiducial HOD), '
                 'in units of the error of one $(500\\,h^{-1}$Mpc$)^3$ volume', fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / 'paper_derivatives.png', dpi=150)
    plt.close(fig)


def fig_triangle(fc):
    from getdist.gaussian_mixtures import GaussianND
    from getdist import plots
    from fisher_emulator import INPUT_OF
    vectors = {'2PCF': ['gg'], 'ASTRA': q('astra'), 'DS': q('ds'), 'ASTRA+DS': q('astra') + q('ds'),
               '2PCF+ASTRA+DS': ['gg'] + q('astra') + q('ds')}
    fid = [fc.x0[E.COSMO_INPUTS.index(INPUT_OF[p])] for p in PARAMS]
    gs = [GaussianND(fid, fc(v)['cov_marg'], names=PARAMS, labels=[PTEX[p] for p in PARAMS], label=name)
          for name, v in vectors.items()]
    g = plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(gs, filled=[False, True, True, False, False], contour_colors=[COLV[k] for k in vectors])
    g.export(str(PLOTS / 'paper_triangle.png'))


def fig_quartiles(fc):
    params = PARAMS
    ref = fc(['gg'])['marg'][0]
    variants = [('Q1 cross', dict(qs=(1,))), ('Q4 cross', dict(qs=(4,))), ('Q1+Q4 cross', {}),
                ('Q1+Q4 cross+auto', dict(kinds=('cross', 'auto')))]
    fig, axes = plt.subplots(1, len(params), figsize=(3.8 * len(params), 3.6), sharey=True)
    w = 0.38
    for k, (ax, p) in enumerate(zip(axes, params)):
        for j, (m, col) in enumerate([('astra', '#d95f02'), ('ds', '#1b9e77')]):
            vals = [ref[k] / fc(q(m, **kw))['marg'][0][k] for _, kw in variants]
            ax.bar(np.arange(len(variants)) + (j - 0.5) * w, vals, w, color=col,
                   label='ASTRA' if m == 'astra' else 'DS')
        ax.axhline(1, color='k', lw=0.6, ls='--')
        ax.set_xticks(range(len(variants)), [v for v, _ in variants], rotation=30, fontsize=8)
        ax.set_title(f'${PTEX[p]}$')
    axes[0].set_ylabel(r'$\sigma_{\rm 2PCF}/\sigma$')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Information per environment and statistic (baseline, HOD marginalised): gain over the 2PCF', fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / 'paper_quartiles.png', dpi=150)
    plt.close(fig)


def fig_response():
    """Shift of the mean environment of each quartile, Quijote halos, in units of the fiducial scatter."""
    qd = A.DATA / 'quijote'

    def mean_env(name, n=500):
        files = sorted((qd / name).glob('[0-9]*.npz'))[:n]
        return np.array([np.load(f)['mean_env'][0] for f in files])       # (n, method, pop, q)
    fid = mean_env('fiducial', 2000)
    mu, sd = fid.mean(0), fid.std(0, ddof=1)
    sets = [('Om', 'Om_p', 'Om_m'), ('Ob', 'Ob2_p', 'Ob2_m'), ('h', 'h_p', 'h_m'), ('ns', 'ns_p', 'ns_m'),
            ('s8', 's8_p', 's8_m')]
    labels = {'Om': r'$\Omega_m$', 'Ob': r'$\Omega_b$', 'h': '$h$', 'ns': '$n_s$', 's8': r'$\sigma_8$'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (im, title) in zip(axes, [(0, 'ASTRA: mean $r$ of the halos per quartile'), (1, r'Density split: mean $\delta$ of the halos per quartile')]):
        for k, (p, sp, sm) in enumerate(sets):
            for qi in range(4):
                for name, mk, col in [(sp, '^', '#1f78b4'), (sm, 'v', '#e31a1c')]:
                    y = (mean_env(name).mean(0)[im, 0, qi] - mu[im, 0, qi]) / sd[im, 0, qi]
                    ax.scatter(k + 0.15 * (qi - 1.5), y, marker=mk, color=col, s=25,
                               label=('+' if mk == '^' else '$-$') if (k == 0 and qi == 0) else None)
        ax.axhline(0, color='0.5', lw=0.6)
        ax.set_xticks(range(len(sets)), [labels[p] for p, _, _ in sets])
        ax.set_title(title + ' (Q1 to Q4 left to right)', fontsize=10)
    axes[0].set_ylabel(r'shift / fiducial $1\sigma$ (one realisation)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Response of the halo-quartile environments to the cosmological parameters, Quijote halos '
                 '(500 realisations per variation)', fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / 'paper_response.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', default=None)
    parser.add_argument('--params', nargs='+', default=None)
    args = parser.parse_args()
    global PARAMS
    if args.params:
        import compare_methods
        PARAMS = compare_methods.PARAMS = list(args.params)
    todo = args.only or ['fig_map', 'fig_env_hist', 'fig_corr', 'fig_derivatives', 'fig_triangle',
                         'fig_quartiles', 'fig_response']
    env = slab_environments() if {'fig_map', 'fig_env_hist'} & set(todo) else None
    fc = None
    if {'fig_derivatives', 'fig_triangle', 'fig_quartiles'} & set(todo):
        from compare_methods import Forecaster
        fc = Forecaster(ROOT / 'models_pairs', '_pairs', compress=True, cov='covbox', fiducial='official')
    for name in todo:
        f = globals()[name]
        if name in ('fig_map', 'fig_env_hist'):
            f(env)
        elif name in ('fig_derivatives', 'fig_triangle', 'fig_quartiles'):
            f(fc)
        else:
            f()
        print(f'{name} done', flush=True)


if __name__ == '__main__':
    main()
