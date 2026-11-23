#!/usr/bin/env python
"""
Fisher forecasts with emulator derivatives: ASTRA split vs density split vs 2PCF.

* derivatives: autodiff of the emulator at the fiducial point (c000 cosmology,
  hod239 HOD), i.e. at *exactly* fixed HOD (and fixed theta_*, sigma8 for the
  other cosmological parameters, as in the AbacusSummit grid);
* covariance : 192 (subbox x line of sight) samples of the fiducial,
  Hartlap-corrected, plus the emulator error (variance of the residuals on the
  held-out test cosmologies), for the volume of one subbox;
* HOD: fixed, or marginalised in a joint Fisher with a Gaussian prior of
  width --hod-prior (in units of the prior range; 0.29 = std of the uniform).

Also compares the emulator with the +/- simulation pairs:
  (a) measured pair difference vs emulator prediction of the same difference
      (HOD mismatch included): a direct test of the emulator;
  (b) emulator derivative at fixed HOD vs pair derivative corrected with the
      GP HOD calibration (scripts/fisher.py --hod-correct).

    python scripts/fisher_emulator.py [--hod-prior 0.29] [--params omega_b omega_cdm n_s sigma8]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A
import emulator as E
import astra_ds

INPUT_OF = {'omega_b': 'omega_b', 'omega_cdm': 'omega_cdm', 'sigma8': 'sigma8_m', 'n_s': 'n_s',
            'alpha_s': 'alpha_s', 'N_ur': 'N_ur', 'w0': 'w0_fld', 'wa': 'wa_fld'}
LABELS = dict(A.PARAM_LABELS, alpha_s=r'\alpha_s', N_ur=r'N_{\rm ur}', w0='w_0', wa='w_a')
COLORS = {'2PCF': 'k', 'ASTRA rand': '#d95f02', 'DS rand': '#1b9e77', 'ASTRA gal': '#e6ab02',
          'DS gal': '#66a61e', '2PCF + ASTRA rand': '#7570b3', '2PCF + DS rand': '#e7298a',
          'ASTRA rand + auto': '#a6761d', 'DS rand + auto': '#386cb0'}


def input_point(cosmo, hod):
    """Emulator input vector (physical units) of a catalogue."""
    table = A.cosmology_table()
    h = astra_ds.read_hod_header(astra_ds.hod_path(cosmo, hod))
    return np.array([float(table.loc[cosmo, c]) for c in E.COSMO_INPUTS] + [h[k] for k in astra_ds.HOD_PARAMS])


MODELS_DIR = E.MODELS
VALIDATION_TAG = ''


def emulators(stats):
    return {st: E.Emulator.load(st, MODELS_DIR) for st in stats}


def predict(emus, stats, x):
    with torch.no_grad():
        return np.concatenate([emus[st](x[None])[0].numpy() for st in stats])


def jacobian(emus, stats, x, members=False):
    """(ndim, 20), or (nmembers, ndim, 20): ensemble members are paired across statistics by seed."""
    return np.concatenate([emus[st].jacobian(x, members=members) for st in stats], axis=-2)


def emulator_cov(stats):
    v = np.load(A.DATA / f'emulator_validation{VALIDATION_TAG}.npz')
    return np.diag(np.concatenate([v[f'{st}/res'].var(0) for st in stats]))


def compute(vectors, params, hod_prior, x0, use_emu_cov=True):
    sel = A.selection()['fiducial']
    meta, fstats, _ = A.load(sel['cosmo'], sel['hod'])
    lo, hi = np.array(A.selection()['prior_lo']), np.array(A.selection()['prior_hi'])
    ipar = [E.COSMO_INPUTS.index(INPUT_OF[p]) for p in params]
    ihod = list(range(len(E.COSMO_INPUTS), len(x0)))
    results = {}
    for name in vectors:
        stats = A.DATA_VECTORS[name]
        _, samples = A.data_vector(meta, fstats, stats)
        n, p = samples.shape
        cov = np.cov(samples, rowvar=False)
        hartlap = (n - p - 2) / (n - 1)
        C = cov + (emulator_cov(stats) if use_emu_cov else 0.)
        prec = hartlap * np.linalg.inv(C)
        emus = emulators(stats)
        J = jacobian(emus, stats, x0)
        D = J[:, ipar + ihod]
        F = D.T @ prec @ D
        npar = len(params)
        F_fixed = F[:npar, :npar]
        Fp = F.copy()
        Fp[npar:, npar:] += np.diag(1. / (hod_prior * (hi - lo))**2)
        C_marg = np.linalg.inv(Fp)[:npar, :npar]
        # scatter of the constraints over the ensemble members (training noise of the emulator)
        sig_members = []
        for Jm in jacobian(emus, stats, x0, members=True):
            Dm = Jm[:, ipar + ihod]
            Fm = Dm.T @ prec @ Dm
            Fm[npar:, npar:] += np.diag(1. / (hod_prior * (hi - lo))**2)
            sig_members.append(np.sqrt(np.diag(np.linalg.inv(Fm)[:npar, :npar])))
        sig_members = np.array(sig_members)
        results[name] = dict(sigma_fixed_hod=np.sqrt(np.diag(np.linalg.inv(F_fixed))).tolist(),
                             sigma_marg_hod_members_mean=sig_members.mean(0).tolist(),
                             sigma_marg_hod_members_std=sig_members.std(0, ddof=1).tolist(),
                             sigma_marg_hod=np.sqrt(np.diag(C_marg)).tolist(),
                             cov_fixed_hod=np.linalg.inv(F_fixed).tolist(), cov_marg_hod=C_marg.tolist(),
                             ndim=p, hartlap=hartlap,
                             emu_to_stat_var=float(np.mean(np.diag(emulator_cov(stats)) / np.diag(cov))))
    return results


def report(params, results):
    for key, title in (('sigma_fixed_hod', 'fixed HOD'), ('sigma_marg_hod', 'HOD marginalised')):
        print(f'\nmarginalised 1-sigma, V = 0.125 (Gpc/h)^3, {title}')
        print(f'{"data vector":20s} {"ndim":>4s} ' + ' '.join(f'{p:>10s}' for p in params) +
              '   gain vs 2PCF')
        ref = np.array(results['2PCF'][key]) if '2PCF' in results else None
        for name, r in results.items():
            sig = np.array(r[key])
            gain = ' '.join(f'{g:4.1f}' for g in ref / sig) if ref is not None else ''
            print(f'{name:20s} {r["ndim"]:4d} ' + ' '.join(f'{s:10.2e}' for s in sig) + f'   {gain}')
            if key == 'sigma_marg_hod':
                m, sd = np.array(r['sigma_marg_hod_members_mean']), np.array(r['sigma_marg_hod_members_std'])
                print(f'{"   single networks":20s} {"":4s} ' + ' '.join(f'{100 * e:8.0f}% ' for e in sd / m)
                      + '  (relative scatter of sigma over ensemble members)')


def plot_triangle(params, results, names, x0, key, out):
    from matplotlib.patches import Ellipse
    n = len(params)
    fid = [x0[E.COSMO_INPUTS.index(INPUT_OF[p])] for p in params]
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    width = [max(np.sqrt(np.array(results[nm][key.replace('sigma', 'cov')])[i, i]) for nm in names) * 3.5
             for i in range(n)]
    for nm in names:
        C = np.array(results[nm][key.replace('sigma', 'cov')])
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if j > i:
                    ax.axis('off')
                    continue
                if i == j:
                    sg = np.sqrt(C[i, i])
                    x = fid[i] + np.linspace(-width[i], width[i], 200)
                    ax.plot(x, np.exp(-0.5 * ((x - fid[i]) / sg)**2), color=COLORS.get(nm), lw=1.4)
                    ax.set_yticks([])
                else:
                    sub = C[np.ix_([j, i], [j, i])]
                    vals, vecs = np.linalg.eigh(sub)
                    ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
                    for nsig, alpha in ((1.52, 1.), (2.48, .5)):
                        ax.add_patch(Ellipse((fid[j], fid[i]), 2 * nsig * np.sqrt(vals[-1]),
                                             2 * nsig * np.sqrt(vals[0]), angle=ang, fill=False,
                                             color=COLORS.get(nm), lw=1.4, alpha=alpha))
                    ax.set_ylim(fid[i] - width[i], fid[i] + width[i])
                ax.set_xlim(fid[j] - width[j], fid[j] + width[j])
                ax.tick_params(labelsize=7)
    for i in range(n):
        axes[-1, i].set_xlabel(f'${LABELS[params[i]]}$')
        if i:
            axes[i, 0].set_ylabel(f'${LABELS[params[i]]}$')
    fig.legend(handles=[plt.Line2D([], [], color=COLORS.get(nm), lw=2, label=nm) for nm in names],
               loc='upper right', frameon=False)
    fig.suptitle(f'Emulator Fisher, V = 0.125 $(h^{{-1}}{{\\rm Gpc}})^3$, '
                 f'{"HOD marginalised" if "marg" in key else "fixed HOD"}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'saved {out}')


def compare_with_pairs(stats_plot=('gg', 'astra_rand_cross_q1', 'astra_rand_cross_q4',
                                   'ds_rand_cross_q1', 'ds_rand_cross_q4')):
    """Figure: (a) measured vs emulated pair differences, (b) emulator vs GP-corrected derivatives."""
    sel = A.selection()
    table = A.cosmology_table()
    emus = emulators(stats_plot)
    s = emus['gg'].s
    ns = len(s)
    params = list(sel['pairs'])
    fig, axes = plt.subplots(len(params), len(stats_plot), figsize=(3.3 * len(stats_plot), 2.6 * len(params)),
                             sharex=True)
    summary = {}
    resp = A.GPHODResponse(list(stats_plot), loo=False)
    for i, param in enumerate(params):
        pp, pm = sel['pairs'][param]['plus'], sel['pairs'][param]['minus']
        step = table.loc[pp['cosmo'], A.PARAM_COLUMNS[param]] - table.loc[pm['cosmo'], A.PARAM_COLUMNS[param]]
        xp, xm = input_point(pp['cosmo'], pp['hod']), input_point(pm['cosmo'], pm['hod'])
        hp, hm = A.hod_vector(pp['cosmo'], pp['hod']), A.hod_vector(pm['cosmo'], pm['hod'])
        mp, sp, ip = A.load(pp['cosmo'], pp['hod'])
        mm, sm, im = A.load(pm['cosmo'], pm['hod'])
        raw = A.paired_derivative(A.data_vector(mp, sp, list(stats_plot))[1],
                                  A.data_vector(mm, sm, list(stats_plot))[1], ip, im, step)
        raw_mean, raw_err = raw.mean(0), raw.std(0, ddof=1) / np.sqrt(len(raw))
        corrected = raw_mean - resp.correction(hp, hm) / step
        emu_raw = (predict(emus, stats_plot, xp) - predict(emus, stats_plot, xm)) / step
        xmid = 0.5 * (xp + xm)
        ipar = E.COSMO_INPUTS.index(INPUT_OF[param])
        emu_der = jacobian(emus, stats_plot, xmid)[:, ipar]
        summary[param] = {}
        for j, st in enumerate(stats_plot):
            sl = slice(2 * ns * j, 2 * ns * j + ns)            # monopole of statistic j
            ax = axes[i, j]
            ax.fill_between(s, s**2 * (raw_mean[sl] - raw_err[sl]), s**2 * (raw_mean[sl] + raw_err[sl]),
                            color='0.7', label='pair difference (sims)')
            ax.plot(s, s**2 * emu_raw[sl], color='0.3', ls='--', lw=1.2, label='pair difference (emulator)')
            ax.plot(s, s**2 * corrected[sl], color='#1f78b4', lw=1.5, label='HOD-corrected pair (GP)')
            ax.plot(s, s**2 * emu_der[sl], color='#e31a1c', lw=1.5, label='emulator, fixed HOD')
            ax.axhline(0, color='0.6', lw=0.5)
            if i == 0:
                ax.set_title(st.replace('_', ' '), fontsize=9)
            if j == 0:
                ax.set_ylabel(rf'$s^2\partial\xi_0/\partial {LABELS[param]}$', fontsize=9)
            ax.tick_params(labelsize=7)
            summary[param][st] = {
                'emu_vs_sim_pair_diff': float(np.median(np.abs(emu_raw[sl] - raw_mean[sl]) / np.abs(raw_mean[sl]).max())),
                'emu_vs_gp_corrected': float(np.median(np.abs(emu_der[sl] - corrected[sl]) / np.abs(emu_der[sl]).max()))}
    for ax in axes[-1]:
        ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle('Emulator vs simulation pairs: raw pair differences (grey) and cosmological derivatives at fixed '
                 'HOD (red: emulator, blue: GP-corrected pairs)', fontsize=10)
    fig.tight_layout()
    out = ROOT / 'plots' / 'emulator_vs_pairs.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'saved {out}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--params', nargs='+', default=['omega_b', 'omega_cdm', 'n_s', 'sigma8'])
    parser.add_argument('--hod-prior', type=float, default=0.29)
    parser.add_argument('--vectors', nargs='+', default=list(A.DATA_VECTORS))
    parser.add_argument('--no-emu-cov', action='store_true')
    parser.add_argument('--skip-pairs', action='store_true')
    parser.add_argument('--models-dir', type=Path, default=E.MODELS)
    parser.add_argument('--validation-tag', default='', help='emulator error from data/emulator_validation{tag}.npz')
    args = parser.parse_args()
    global MODELS_DIR, VALIDATION_TAG
    MODELS_DIR, VALIDATION_TAG = args.models_dir, args.validation_tag
    sel = A.selection()['fiducial']
    x0 = input_point(sel['cosmo'], sel['hod'])
    results = compute(args.vectors, args.params, args.hod_prior, x0, not args.no_emu_cov)
    report(args.params, results)
    tag = ('_'.join(args.params) if len(args.params) != 4 else 'base') + \
        ('' if args.models_dir == E.MODELS else '_' + Path(args.models_dir).name)
    (A.DATA / f'fisher_emulator_{tag}.json').write_text(json.dumps(dict(params=args.params, args=vars(args),
                                                                        results=results), indent=1))
    for key in ('sigma_fixed_hod', 'sigma_marg_hod'):
        for group, names in {'methods': ['2PCF', 'ASTRA rand', 'DS rand'],
                             'combined': ['2PCF', '2PCF + ASTRA rand', '2PCF + DS rand'],
                             'galaxy_quartiles': ['2PCF', 'ASTRA gal', 'DS gal']}.items():
            names = [nm for nm in names if nm in results]
            plot_triangle(args.params, results, names, x0, key,
                          ROOT / 'plots' / f'fisher_emulator_{group}_{key.split("_", 1)[1]}_{tag}.png')
    if not args.skip_pairs:
        summary = compare_with_pairs()
        print('\nmedian |difference| / max|signal| (monopole):')
        for param, d in summary.items():
            print(f'  {param:10s} ' + '  '.join(f'{st}: emu-vs-sims {v["emu_vs_sim_pair_diff"]:.2f}, '
                                                f'emu-vs-GP {v["emu_vs_gp_corrected"]:.2f}' for st, v in d.items()
                                                if st in ('gg', 'astra_rand_cross_q4', 'ds_rand_cross_q4')))


if __name__ == '__main__':
    main()
