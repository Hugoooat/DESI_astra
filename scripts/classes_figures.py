#!/usr/bin/env python
"""
Figures of the ASTRA cosmic-web class analysis (void, sheet, filament, knot).

  abacus     classes on the fiducial AbacusSummit catalogue (64 subvolumes):
             slab map and multipoles of the class statistics
             -> plots/classes_abacus_map.png, plots/classes_abacus_multipoles.png
  response   response of the class fractions to the cosmological parameters (Quijote)
             -> plots/classes_response.png
  fisher     Quijote Fisher errors of classes vs quartiles vs density split
             -> plots/classes_fisher.png

    python scripts/classes_figures.py [--only abacus response fisher]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A
import astra_ds as D

PLOTS = ROOT / 'plots'
CCOL = {'void': '#3b4cc0', 'sheet': '#7aa6f7', 'filament': '#f4a582', 'knot': '#b2182b'}


def fig_abacus(nthreads=128):
    out = A.DATA / 'classes_abacus_fiducial.npz'
    gal, box = D.load_catalog(D.hod_path(0, 239))
    if not out.exists():
        res, _ = D.measure_classes(gal, box, 'c000_hod239', nside=4, nthreads=nthreads)
        np.savez(out, **res)
    res = np.load(out)
    # map: one slab of the first subvolume
    ran = D.uniform_randoms(len(gal), box, 42)
    corners, size = D.subbox_corners(box, 4)
    _, g, gc = D.extract(gal, corners[0], size, box, 30.)
    _, r, rc = D.extract(ran, corners[0], size, box, 30.)
    rg, rr = D.astra_r(g, r)
    lab = {'gal': D.astra_class_labels(rg[gc]), 'ran': D.astra_class_labels(rr[rc])}
    pos = {'gal': g[gc], 'ran': r[rc]}
    z0 = size[2] / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, pop, title in zip(axes, ('ran', 'gal'), ('random points', 'galaxies')):
        sl = np.abs(pos[pop][:, 2] - z0) < 10.
        for ic, c in enumerate(D.CLASSES):
            m = sl & (lab[pop] == ic)
            ax.scatter(*pos[pop][m, :2].T, s=4 if c in ('void', 'knot') else 0.8, c=CCOL[c], lw=0,
                       label=f'{c} ({np.mean(lab[pop] == ic):.1%})')
        ax.set_title(f'ASTRA classes of the {title}', fontsize=11)
        ax.set_xlabel(r'$x\ [h^{-1}{\rm Mpc}]$')
        ax.set_aspect('equal')
        ax.legend(markerscale=4, fontsize=8, loc='upper right')
    axes[0].set_ylabel(r'$y\ [h^{-1}{\rm Mpc}]$')
    fig.suptitle(r'ASTRA cosmic-web classes in a $20\,h^{-1}$Mpc slab of one subvolume (fiducial catalogue); '
                 'fractions over the subvolume in the legend', fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / 'classes_abacus_map.png', dpi=150)
    plt.close(fig)
    # multipoles of the class statistics
    s, ells = res['s'], list(res['ells'])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, kind in enumerate(('cross', 'auto')):
        for row, ell in enumerate((0, 2)):
            ax = axes[row, col]
            for c in D.CLASSES:
                x = res[f'astra_rand_{kind}_{c}'][:, ells.index(ell)]
                x = x[np.all(np.isfinite(x), axis=1)]
                if len(x) < 5:
                    continue
                mu, sd = x.mean(0), x.std(0, ddof=1)
                ax.plot(s, s**2 * mu, color=CCOL[c], label=c)
                ax.fill_between(s, s**2 * (mu - sd), s**2 * (mu + sd), color=CCOL[c], alpha=0.2)
            if kind == 'cross':
                ax.plot(s, s**2 * res['gg'][:, ells.index(ell)].mean(0), 'k--', lw=1, label='2PCF')
            ax.axhline(0, color='0.6', lw=0.5)
            ax.set_title(f'{kind.capitalize()} {"monopole" if ell == 0 else "quadrupole"}')
            ax.set_xlabel(r'$s$ [$h^{-1}$ Mpc]')
            ax.set_ylabel(r'$s^2\xi_%d$' % ell)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('ASTRA classes of the random points: multipoles (mean and scatter of 64 subvolumes, fiducial catalogue)')
    fig.tight_layout()
    fig.savefig(PLOTS / 'classes_abacus_multipoles.png', dpi=140)
    plt.close(fig)
    print('abacus fractions rand', res['fraction_rand'].mean(0).round(4), 'data', res['fraction_data'].mean(0).round(4))


def fig_response():
    qd = A.DATA / 'quijote_classes'

    def frac(name, pop):
        return np.array([np.load(f)[f'fraction_{pop}'][0] for f in sorted((qd / name).glob('[0-9]*.npz'))])
    sets = [('Om', r'$\Omega_m$', 'Om_p', 'Om_m'), ('ns', '$n_s$', 'ns_p', 'ns_m'), ('s8', r'$\sigma_8$', 's8_p', 's8_m')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, pop, title in zip(axes, ('rand', 'data'), ('random points (volume)', 'halos')):
        fid = frac('fiducial', pop)
        sd = fid.std(0, ddof=1)
        for k, (p, lab, sp, sm) in enumerate(sets):
            for ic, c in enumerate(D.CLASSES):
                if sd[ic] == 0:
                    continue
                dp = (frac(sp, pop)[:, ic] - frac(sm, pop)[:, ic])
                y, e = dp.mean() / 2 / sd[ic], dp.std(ddof=1) / np.sqrt(len(dp)) / 2 / sd[ic]
                ax.errorbar(k + 0.15 * (ic - 1.5), y, e, fmt='o', color=CCOL[c], label=c if k == 0 else None)
        ax.axhline(0, color='0.5', lw=0.6)
        ax.set_xticks(range(len(sets)), [s[1] for s in sets])
        ax.set_title(f'Class fractions of the {title}', fontsize=10)
    axes[0].set_ylabel(r'$(f_+ - f_-)/2$ / fiducial $1\sigma$')
    axes[0].legend(fontsize=8)
    fig.suptitle('Response of the ASTRA class fractions to the $\\pm$ variations of the parameters (Quijote halos, '
                 '500 pairs), in units of the scatter of one realisation', fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / 'classes_response.png', dpi=150)
    plt.close(fig)


def fig_fisher():
    vectors = [('2PCF', '2PCF', 'k'), ('quartiles DS rand Q1+Q4', 'DS quartiles, cross', '#1b9e77'),
               ('quartiles DS rand + auto', 'DS quartiles, cross+auto', '#66c2a5'),
               ('quartiles ASTRA rand Q1+Q4', 'ASTRA quartiles, cross', '#d95f02'),
               ('quartiles ASTRA rand + auto', 'ASTRA quartiles, cross+auto', '#fc8d62'),
               ('classes rand: sheet, filament', 'ASTRA classes, cross', '#3b4cc0'),
               ('classes rand + auto: sheet, filament', 'ASTRA classes, cross+auto', '#7aa6f7')]
    labels = {'Om': r'$\Omega_m$', 'ns': '$n_s$', 's8': r'$\sigma_8$'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={'width_ratios': [2, 3]})
    for ax, tag in zip(axes, ('_p2', '_p3')):
        d = json.loads((A.DATA / f'quijote_classes_fisher{tag}.json').read_text())
        params = d['params']
        w = 0.8 / len(vectors)
        for j, (key, lab, col) in enumerate(vectors):
            r = d['results'][key]
            rel = np.array(r['sigma_compressed']) / np.array(r['sigma_standard'])
            for k in range(len(params)):
                ok = rel[k] < 1.5
                ax.bar(k + (j - len(vectors) / 2 + 0.5) * w, r['sigma'][k], w, color=col if ok else 'none',
                       edgecolor=col, hatch=None if ok else '//', label=lab if k == 0 else None)
        ax.set_yscale('log')
        ax.set_xticks(range(len(params)), [labels[p] for p in params])
        ax.set_title(f'Parameters {", ".join(labels[p] for p in params)}', fontsize=10)
    axes[0].set_ylabel(r'marginalised $\sigma$, $V = 1\,(h^{-1}$Gpc$)^3$')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, fontsize=8, frameon=False)
    fig.suptitle('Quijote halos: ASTRA classes vs ASTRA and density-split quartiles (random points); '
                 'hatched: not reliable (compressed/standard Fisher > 1.5)', fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(PLOTS / 'classes_fisher.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', default=['abacus', 'response', 'fisher'])
    for name in parser.parse_args().only:
        globals()[f'fig_{name}']()
        print(f'{name} done', flush=True)


if __name__ == '__main__':
    main()
