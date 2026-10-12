#!/usr/bin/env python
"""
Fisher forecasts on (omega_b, omega_cdm, n_s, sigma8): ASTRA split vs density split.

* covariance : all (line of sight, subbox) samples of the fiducial c000
               catalogue, Hartlap-corrected;
* derivatives: paired central differences of the HOD-matched +/- catalogues
               (same subbox, same line of sight), averaged over samples, with
               the derivative-noise bias subtracted from the Fisher matrix
               (see analysis.fisher).

Constraints are quoted for the volume of one 500 Mpc/h subbox
(0.125 (Gpc/h)^3); --volume rescales them (e.g. --volume 8 for the full box,
assuming the subboxes are independent). They are at fixed HOD parameters, so
they measure the information content of each statistic, not a final forecast.

    python scripts/fisher.py [--smin 20] [--smax 150] [--rebin 2] [--ells 0 2]
                             [--los z x y] [--vectors '2PCF' 'ASTRA rand' ...]

Outputs: plots/fisher_*.png, data/fisher_{tag}.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A

COLORS = {'2PCF': 'k', 'ASTRA rand': '#d95f02', 'DS rand': '#1b9e77', 'ASTRA gal': '#e6ab02',
          'DS gal': '#66a61e', '2PCF + ASTRA rand': '#7570b3', '2PCF + DS rand': '#e7298a',
          'ASTRA rand + auto': '#a6761d', 'DS rand + auto': '#386cb0'}


def compute(vectors, los, smin, smax, rebin, ells, hod_correct=False, hod_marg=None, hod_model='gp'):
    sel = A.selection()
    table = A.cosmology_table()
    fid = sel['fiducial']
    meta, fstats, _ = A.load(fid['cosmo'], fid['hod'], los=los)
    runs = {}
    for param, pair in sel['pairs'].items():
        p, m = pair['plus'], pair['minus']
        try:
            runs[param] = (A.load(p['cosmo'], p['hod'], los=los), A.load(m['cosmo'], m['hod'], los=los),
                           table.loc[p['cosmo'], A.PARAM_COLUMNS[param]] - table.loc[m['cosmo'], A.PARAM_COLUMNS[param]],
                           (A.hod_vector(p['cosmo'], p['hod']), A.hod_vector(m['cosmo'], m['hod'])))
        except FileNotFoundError as exc:
            print(f'skip {param}: {exc}')
    params = [p for p in A.PARAMS if p in runs]

    results = {}
    for name in vectors:
        stats = A.DATA_VECTORS[name]
        _, cov_samples = A.data_vector(meta, fstats, stats, ells, smin, smax, rebin)
        try:
            prec, cov, hartlap = A.precision(cov_samples)
        except ValueError as exc:
            print(f'{name}: {exc}')
            continue
        dv = dict(ells=ells, smin=smin, smax=smax, rebin=rebin)
        Response = A.GPHODResponse if hod_model == 'gp' else A.HODResponse
        response = Response(stats, **dv) if (hod_correct or hod_marg) else None
        derivs = {}
        for param in params:
            (mp, sp, ip), (mm, sm, im), step, (hp, hm) = runs[param]
            xp = A.data_vector(mp, sp, stats, **dv)[1]
            xm = A.data_vector(mm, sm, stats, **dv)[1]
            derivs[param] = A.paired_derivative(xp, xm, ip, im, step)
            if response is not None:   # remove the response to the HOD mismatch of the pair
                derivs[param] = derivs[param] - response.correction(hp, hm) / step
        F, noise_frac = A.fisher(derivs, prec)
        extra = {}
        if response is not None:
            extra['hod_loo_rel'] = float(np.sqrt((response.loo_residual**2).mean() / (response.dy**2).mean()))
            extra['ncalib'] = len(response.hods)
        if hod_marg:
            # joint Fisher (cosmology + 12 HOD parameters), Gaussian prior of width hod_marg on h
            M = np.array([derivs[p].mean(0) for p in params] + list(response.G.T))
            Fj = M @ prec @ M.T
            Fj[:len(params), :len(params)] = F            # cosmology block with derivative-noise debiasing
            Fj[len(params):, len(params):] += np.eye(response.G.shape[1]) / hod_marg**2
            extra['F_joint'] = Fj.tolist()
            F = np.linalg.inv(np.linalg.inv(Fj)[:len(params), :len(params)])   # marginalised over the HOD
        Finv = np.linalg.inv(F)
        if np.any(np.linalg.eigvalsh(F) <= 0):
            print(f'{name}: Fisher matrix not positive definite (derivatives degenerate / noise-dominated)')
        results[name] = dict(F=F.tolist(), cov=Finv.tolist(), sigma=np.sqrt(np.diag(Finv)).tolist(),
                             sigma_unmarg=(1 / np.sqrt(np.diag(F))).tolist(),
                             noise_frac=noise_frac.tolist(), ndim=cov_samples.shape[1],
                             nsamples=cov_samples.shape[0], hartlap=hartlap,
                             nderiv=min(len(d) for d in derivs.values()), **extra)
    return params, results


def report(params, results, volume):
    scale = 1 / np.sqrt(volume / 0.125)
    head = f'{"data vector":22s} {"ndim":>4s} {"Nsamp":>5s} {"Hartlap":>7s} ' + \
           ' '.join(f'{p:>16s}' for p in params)
    print(f'\nmarginalised 1-sigma for V = {volume} (Gpc/h)^3   [derivative-noise fraction of F_ii]')
    print(head)
    for name, r in results.items():
        cells = ' '.join(f'{s * scale:9.2e} [{nf:4.0%}]' for s, nf in zip(r['sigma'], r['noise_frac']))
        hod = f'  [HOD model: {r["ncalib"]} calib., LOO error {r["hod_loo_rel"]:.0%}]' if 'ncalib' in r else ''
        print(f'{name:22s} {r["ndim"]:4d} {r["nsamples"]:5d} {r["hartlap"]:7.2f} {cells}{hod}')


def ellipse(ax, mean, cov, nsig, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    w, h = 2 * nsig * np.sqrt(vals[::-1])
    ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False, **kwargs))


def plot_triangle(params, results, names, fiducial, volume, out):
    scale2 = 0.125 / volume
    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    for name in names:
        r = results[name]
        C = np.array(r['cov']) * scale2
        color = COLORS.get(name, None)
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if j > i:
                    ax.axis('off')
                    continue
                if i == j:
                    sig = np.sqrt(C[i, i])
                    x = fiducial[i] + np.linspace(-4, 4, 200) * sig
                    ax.plot(x, np.exp(-0.5 * ((x - fiducial[i]) / sig)**2), color=color, lw=1.4)
                    ax.set_yticks([])
                else:
                    sub = C[np.ix_([j, i], [j, i])]
                    for nsig, alpha in ((1.52, 1.), (2.48, 0.5)):   # 68% / 95% 2D contours
                        ellipse(ax, (fiducial[j], fiducial[i]), sub, nsig, color=color, lw=1.4, alpha=alpha)
    for i in range(n):
        lim = max(np.sqrt(np.array(results[nm]['cov'])[i, i] * scale2) for nm in names) * 3.5
        for j in range(n):
            if j <= i:
                axes[i, j].set_xlim(fiducial[j] - max(np.sqrt(np.array(results[nm]['cov'])[j, j] * scale2)
                                                      for nm in names) * 3.5,
                                    fiducial[j] + max(np.sqrt(np.array(results[nm]['cov'])[j, j] * scale2)
                                                      for nm in names) * 3.5)
            if j < i:
                axes[i, j].set_ylim(fiducial[i] - lim, fiducial[i] + lim)
        axes[-1, i].set_xlabel(f'${A.PARAM_LABELS[params[i]]}$')
        if i > 0:
            axes[i, 0].set_ylabel(f'${A.PARAM_LABELS[params[i]]}$')
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], color=COLORS.get(nm), lw=2, label=nm) for nm in names]
    fig.legend(handles=handles, loc='upper right', fontsize=10, frameon=False)
    fig.suptitle(f'Fisher forecast, V = {volume} $(h^{{-1}}{{\\rm Gpc}})^3$, fixed HOD', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'saved {out}')


def plot_bars(params, results, names, volume, out):
    scale = 1 / np.sqrt(volume / 0.125)
    ref = np.array(results['2PCF']['sigma']) if '2PCF' in results else None
    fig, axes = plt.subplots(1, len(params), figsize=(3.4 * len(params), 3.6), sharey=True)
    for k, (ax, p) in enumerate(zip(np.atleast_1d(axes), params)):
        vals = [results[nm]['sigma'][k] * scale for nm in names]
        if ref is not None:
            vals = [ref[k] * scale / v for v in vals]
        ax.barh(range(len(names)), vals, color=[COLORS.get(nm) for nm in names])
        ax.set_title(f'${A.PARAM_LABELS[p]}$')
        ax.axvline(1 if ref is not None else 0, color='0.5', lw=0.8, ls='--')
        ax.set_xlabel(r'$\sigma_{\rm 2PCF} / \sigma$' if ref is not None else r'$\sigma$')
    np.atleast_1d(axes)[0].set_yticks(range(len(names)), names)
    fig.suptitle('Gain in marginalised precision with respect to the galaxy 2PCF', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'saved {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smin', type=float, default=20.)
    parser.add_argument('--smax', type=float, default=150.)
    parser.add_argument('--rebin', type=int, default=2)
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--los', nargs='+', default=['x', 'y', 'z'])
    parser.add_argument('--volume', type=float, default=0.125, help='(Gpc/h)^3')
    parser.add_argument('--vectors', nargs='+', default=list(A.DATA_VECTORS))
    parser.add_argument('--hod-correct', action='store_true',
                        help='subtract the calibrated HOD response from the pair differences')
    parser.add_argument('--hod-marg', type=float, default=None, metavar='SIGMA',
                        help='joint cosmology + HOD Fisher, marginalised over the HOD with a Gaussian prior '
                             'of width SIGMA in prior-range units (0.29 = std of the uniform prior)')
    parser.add_argument('--hod-model', choices=['gp', 'linear'], default='gp',
                        help='model of the HOD response fitted on the c000 calibration catalogues')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    params, results = compute(args.vectors, args.los, args.smin, args.smax, args.rebin, tuple(args.ells),
                              args.hod_correct or args.hod_marg is not None, args.hod_marg, args.hod_model)
    report(params, results, args.volume)
    tag = args.tag or (f'smin{args.smin:g}_los{"".join(sorted(args.los))}'
                       + ('_hodmarg' if args.hod_marg else '_hodcorr' if args.hod_correct else '')
                       + (f'_{args.hod_model}' if (args.hod_marg or args.hod_correct) else ''))
    (ROOT / 'plots').mkdir(exist_ok=True)
    (A.DATA / f'fisher_{tag}.json').write_text(json.dumps(
        dict(params=params, args=vars(args), results=results), indent=1, default=str))

    table = A.cosmology_table()
    fiducial = [table.loc[0, A.PARAM_COLUMNS[p]] for p in params]
    groups = {'methods': ['2PCF', 'ASTRA rand', 'DS rand'],
              'combined': ['2PCF', '2PCF + ASTRA rand', '2PCF + DS rand'],
              'galaxy_quartiles': ['2PCF', 'ASTRA gal', 'DS gal']}
    for gname, names in groups.items():
        names = [nm for nm in names if nm in results and np.all(np.linalg.eigvalsh(np.array(results[nm]['F'])) > 0)]
        if len(names) > 1 and len(params) > 1:
            plot_triangle(params, results, names, fiducial, args.volume, ROOT / 'plots' / f'fisher_{gname}_{tag}.png')
    plot_bars(params, results, [nm for nm in results if np.all(np.isfinite(results[nm]['sigma']))], args.volume, ROOT / 'plots' / f'fisher_bars_{tag}.png')


if __name__ == '__main__':
    main()
