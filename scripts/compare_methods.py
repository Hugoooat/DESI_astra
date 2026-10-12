#!/usr/bin/env python
"""
Head-to-head comparison of ASTRA split and density split (emulator Fisher).

For the same type of data vector (random or galaxy quartiles, crosses with or
without autos, per environment, with or without the 2PCF) and the same scale
cut, we compare the marginalised errors of the two methods:

  1. ratio sigma_DS / sigma_ASTRA per parameter (> 1: ASTRA more constraining),
     with the scatter over the emulator ensemble members as error bar;
  2. complementarity: sigma(ASTRA + DS) vs the best of the two alone;
  3. environments: voids (Q1) and clusters (Q4) separately;
  4. scale dependence: s_min = 20, 40, 60 Mpc/h;
  5. sensitivity to the galaxy-halo connection: sigma(HOD marginalised) /
     sigma(fixed HOD).

    python scripts/compare_methods.py [--models-dir models10] [--validation-tag _10seeds]
Outputs: data/compare_methods.json, plots/compare_*.png
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
import emulator as E
import astra_ds
from fisher_emulator import input_point, INPUT_OF, LABELS

PARAMS = ['omega_b', 'omega_cdm', 'n_s', 'sigma8']


def q(method, pop='rand', kinds=('cross',), qs=(1, 4)):
    return [f'{method}_{pop}_{k}_q{i}' for k in kinds for i in qs]


# (label, ASTRA vector, DS vector)
PAIRS = [
    ('random Q1+Q4 cross', q('astra'), q('ds')),
    ('random Q1+Q4 cross+auto', q('astra', kinds=('cross', 'auto')), q('ds', kinds=('cross', 'auto'))),
    ('galaxy Q1+Q4 cross', q('astra', 'data'), q('ds', 'data')),
    ('random Q1 (voids)', q('astra', qs=(1,), kinds=('cross', 'auto')), q('ds', qs=(1,), kinds=('cross', 'auto'))),
    ('random Q4 (clusters)', q('astra', qs=(4,), kinds=('cross', 'auto')), q('ds', qs=(4,), kinds=('cross', 'auto'))),
    ('2PCF + random Q1+Q4 cross', ['gg'] + q('astra'), ['gg'] + q('ds')),
]
COMBINED = {'2PCF': ['gg'], 'ASTRA': q('astra'), 'DS': q('ds'), 'ASTRA + DS': q('astra') + q('ds'),
            '2PCF + ASTRA': ['gg'] + q('astra'), '2PCF + DS': ['gg'] + q('ds'),
            '2PCF + ASTRA + DS': ['gg'] + q('astra') + q('ds')}


class Forecaster:
    """Emulator Fisher for arbitrary lists of statistics and scale cuts (cached emulators and covariance)."""

    def __init__(self, models_dir, validation_tag, hod_prior=0.29, compress=False, regression=None, nboot=50,
                 cov='subbox', fiducial=None):
        self.compress = compress
        self.cov = cov
        # optional: cosmological derivatives from scripts/pair_regression.py instead of the emulator
        # (bootstrap over subboxes replaces the ensemble members); HOD derivatives stay from the emulator
        self.reg = np.load(regression) if regression else None
        self.nboot = nboot
        sel = A.selection()
        if fiducial in (None, 'hod239'):
            self.x0 = input_point(sel['fiducial']['cosmo'], sel['fiducial']['hod'])
        elif fiducial == 'official':
            from mcmc import OFFICIAL_HOD
            table = A.cosmology_table()
            self.x0 = np.array([float(table.loc[0, c]) for c in E.COSMO_INPUTS] +
                               [OFFICIAL_HOD[k] for k in astra_ds.HOD_PARAMS])
        else:
            c, h = map(int, fiducial.split('_'))
            self.x0 = input_point(c, h)
        if cov == 'covbox':
            from mcmc import covariance_samples
            self._covbox = covariance_samples
        self.meta, self.fstats, _ = A.load(sel['fiducial']['cosmo'], sel['fiducial']['hod'])
        self.val = np.load(A.DATA / f'emulator_validation{validation_tag}.npz')
        self.models_dir = models_dir
        lo, hi = np.array(sel['prior_lo']), np.array(sel['prior_hi'])
        self.hod_prior = np.diag(1. / (hod_prior * (hi - lo))**2)
        self.ipar = [E.COSMO_INPUTS.index(INPUT_OF[p]) for p in PARAMS]
        self.ihod = list(range(len(E.COSMO_INPUTS), len(self.x0)))
        self._jac = {}

    def jac(self, stat):
        if stat not in self._jac:
            J = E.Emulator.load(stat, self.models_dir).jacobian(self.x0, members=True)   # (nmembers, ndim, 20)
            if self.reg is not None:
                d = np.stack([self.reg[f'{p}/{stat}'] for p in PARAMS], axis=-1)      # (nsub, ndim, 4)
                rng = np.random.default_rng(1)
                boot = [d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(self.nboot)]
                Jr = np.repeat(J.mean(0, keepdims=True), self.nboot, axis=0)
                Jr[:, :, self.ipar] = np.array(boot)
                # put the full-sample estimate first in place of nothing: members = bootstrap realisations
                J = Jr
                self._jac_mean = getattr(self, '_jac_mean', {})
                self._jac_mean[stat] = d.mean(0)
            self._jac[stat] = J
        return self._jac[stat]

    def __call__(self, stats, smin=20.):
        s_emu = E.Emulator.load(stats[0], self.models_dir).s
        keep = np.tile(s_emu >= smin, 2)                       # monopole and quadrupole bins
        if self.cov == 'covbox':
            samples = self._covbox('covbox', stats)
            s_all = E.Emulator.load(stats[0], self.models_dir).s
            samples = samples[:, np.tile(np.concatenate([s_all >= smin] * 1), len(stats) * 2)]
        else:
            _, samples = A.data_vector(self.meta, self.fstats, stats, smin=smin)
        n, p = samples.shape
        emu_var = np.concatenate([self.val[f'{st}/res'].var(0)[keep] for st in stats])
        J = np.concatenate([self.jac(st)[:, keep] for st in stats], axis=1)   # (nmembers, ndim, 20)
        # central derivatives: ensemble mean, or for the regression the mean over all subboxes
        J0 = J.mean(0, keepdims=True)
        if self.reg is not None:
            J0 = J0.copy()
            J0[0][:, self.ipar] = np.concatenate([self._jac_mean[st][keep] for st in stats])
        if self.compress:
            # project on the 16 informative directions t = W^T d, W = C_shrunk^-1 D (lossy if C_shrunk != C,
            # never biased), then estimate the 16x16 covariance of t from the samples
            from sklearn.covariance import LedoitWolf
            Cs = LedoitWolf().fit(samples).covariance_ + np.diag(emu_var)
            W = np.linalg.solve(Cs, J0[0][:, self.ipar + self.ihod])
            samples = samples @ W
            emu_cov = W.T @ np.diag(emu_var) @ W
            J = np.einsum('dk,mdi->mki', W, J)
            J0 = np.einsum('dk,mdi->mki', W, J0)
            n, p = samples.shape
            C = np.cov(samples, rowvar=False) + emu_cov
        else:
            C = np.cov(samples, rowvar=False) + np.diag(emu_var)
        prec = (n - p - 2) / (n - 1) * np.linalg.inv(C)
        out = {'fixed': [], 'marg': []}
        npar = len(PARAMS)
        out_cov = None
        for Jm in np.concatenate([J0, J]):
            D = Jm[:, self.ipar + self.ihod]
            F = D.T @ prec @ D
            out['fixed'].append(np.sqrt(np.diag(np.linalg.inv(F[:npar, :npar]))))
            F[npar:, npar:] += self.hod_prior
            Cm = np.linalg.inv(F)[:npar, :npar]
            out['marg'].append(np.sqrt(np.diag(Cm)))
            out.setdefault('cov_marg', Cm)          # first entry: central derivatives
        # first entry: ensemble-mean derivatives; others: individual members
        cov_marg = out.pop('cov_marg')
        return {k: (np.array(v[0]), np.array(v[1:])) for k, v in out.items()} | {'cov_marg': cov_marg,'ndim': samples.shape[1] if not self.compress else J.shape[1], 'ndim_raw': len(emu_var)}


def ratio_with_error(fc, va, vb, smin=20., key='marg'):
    """
    sigma_b / sigma_a (ensemble-mean derivatives) and its error: the scatter of
    the ratio over (paired) ensemble members divided by sqrt(nmembers), i.e.
    the uncertainty of the ensemble mean due to the training noise.
    """
    ra, rb = fc(va, smin), fc(vb, smin)
    r = rb[key][0] / ra[key][0]
    rm = rb[key][1] / ra[key][1]
    # ensemble members: error of the mean; bootstrap realisations: their scatter is already the error
    err = rm.std(0, ddof=1) / (1. if fc.reg is not None else np.sqrt(len(rm)))
    return r, err, ra, rb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models-dir', type=Path, default=E.MODELS)
    parser.add_argument('--validation-tag', default='')
    parser.add_argument('--cov', choices=['subbox', 'covbox'], default='subbox',
                        help='covariance: 192 subboxes of the fiducial, or the 1786 official small boxes')
    parser.add_argument('--fiducial', default=None, help="'official', or COSMO_HOD (default: c000 hod239)")
    parser.add_argument('--regression', type=Path, default=None,
                        help='cosmological derivatives from pair_regression.py (npz) instead of the emulator')
    parser.add_argument('--compress', action='store_true',
                        help='compress each data vector to 16 numbers (4 cosmology + 12 HOD) before the Fisher')
    args = parser.parse_args()
    fc = Forecaster(args.models_dir, args.validation_tag, compress=args.compress, regression=args.regression,
                    cov=args.cov, fiducial=args.fiducial)
    tag = (f'_{args.cov}' if args.cov != 'subbox' else '') + (f'_fid{args.fiducial}' if args.fiducial else '') + ('_compressed' if args.compress else '') + \
        (f'_regression_{Path(args.regression).stem.replace("derivatives_regression", "").strip("_") or "lin"}'
         if args.regression else '') + \
        ('' if Path(args.models_dir).name in ('models', 'models10') else f'_{Path(args.models_dir).name}')
    res = {}

    # 1, 3, 5: ASTRA vs DS for each type of data vector
    print('\nsigma_DS / sigma_ASTRA (HOD marginalised, s >= 20)   [> 1: ASTRA more constraining]')
    print(f'{"data vector":28s} ' + ' '.join(f'{p:>15s}' for p in PARAMS) + '   HOD degradation ASTRA / DS')
    res['pairs'] = {}
    for label, va, vd in PAIRS:
        r, err, ra, rd = ratio_with_error(fc, va, vd)
        deg_a = ra['marg'][0] / ra['fixed'][0]
        deg_d = rd['marg'][0] / rd['fixed'][0]
        res['pairs'][label] = dict(ratio=r.tolist(), err=err.tolist(), sigma_astra=ra['marg'][0].tolist(),
                                   sigma_ds=rd['marg'][0].tolist(), hod_degradation_astra=deg_a.tolist(),
                                   hod_degradation_ds=deg_d.tolist(), ndim=ra['ndim'])
        print(f'{label:28s} ' + ' '.join(f'{x:6.2f} +- {e:4.2f}' for x, e in zip(r, err))
              + '   ' + ' '.join(f'{a:.2f}/{d:.2f}' for a, d in zip(deg_a, deg_d)))

    # 2: complementarity
    print('\nmarginalised sigma, s >= 20, V = 0.125 (Gpc/h)^3')
    res['combined'] = {}
    for label, v in COMBINED.items():
        r = fc(v)
        res['combined'][label] = dict(sigma=r['marg'][0].tolist(),
                                      scatter=(r['marg'][1].std(0, ddof=1) /
                                               (1. if fc.reg is not None else np.sqrt(len(r['marg'][1])))).tolist(),
                                      ndim=r['ndim'])
        print(f'{label:20s} ndim {r["ndim_raw"]:3d}  ' + ' '.join(f'{x:.2e}' for x in r['marg'][0]))

    # 4: scale dependence
    res['smin'] = {}
    smins = [20., 40., 60.]
    for label in ['2PCF', 'ASTRA', 'DS', 'ASTRA + DS']:
        res['smin'][label] = [fc(COMBINED[label], smin)['marg'][0].tolist() for smin in smins]

    (A.DATA / f'compare_methods{tag}.json').write_text(json.dumps(res, indent=1))

    # figure 1: ratios
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(14, 3.8), sharey=True)
    labels = [lab for lab, _, _ in PAIRS]
    for k, (ax, p) in enumerate(zip(axes, PARAMS)):
        vals = [res['pairs'][lab]['ratio'][k] for lab in labels]
        errs = [res['pairs'][lab]['err'][k] for lab in labels]
        ax.barh(range(len(labels)), vals, xerr=errs, color=['#d95f02' if v > 1 else '#1b9e77' for v in vals],
                alpha=0.8, capsize=3)
        ax.axvline(1, color='k', lw=0.8)
        ax.set_title(f'${LABELS[p]}$')
        ax.set_xlabel(r'$\sigma_{\rm DS} / \sigma_{\rm ASTRA}$')
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
    fig.suptitle('ASTRA split vs density split at equal data-vector type (HOD marginalised, $s\\geq20\\,h^{-1}$Mpc); '
                 'orange: ASTRA tighter, green: DS tighter; error bars: emulator training noise (ensemble)', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / f'compare_ratios{tag}.png', dpi=140)
    plt.close(fig)

    # figure 2: complementarity + scale dependence
    fig, axes = plt.subplots(2, len(PARAMS), figsize=(14, 6.5))
    colors = {'2PCF': 'k', 'ASTRA': '#d95f02', 'DS': '#1b9e77', 'ASTRA + DS': '#7570b3',
              '2PCF + ASTRA': '#e6ab02', '2PCF + DS': '#66a61e', '2PCF + ASTRA + DS': '#e7298a'}
    for k, p in enumerate(PARAMS):
        ax = axes[0, k]
        labs = list(COMBINED)
        ax.bar(range(len(labs)), [res['combined'][l]['sigma'][k] for l in labs],
               yerr=[res['combined'][l]['scatter'][k] for l in labs], color=[colors[l] for l in labs], capsize=2)
        ax.set_xticks(range(len(labs)), labs, rotation=60, ha='right', fontsize=7)
        ax.set_title(f'$\\sigma({LABELS[p]})$', fontsize=10)
        ax = axes[1, k]
        for lab in res['smin']:
            ax.plot(smins, [v[k] for v in res['smin'][lab]], 'o-', color=colors[lab], label=lab)
        ax.set_xlabel(r'$s_{\min}\ [h^{-1}\,\mathrm{Mpc}]$')
        ax.set_yscale('log')
    axes[1, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].set_ylabel(r'$\sigma$ (HOD marginalised)')
    fig.suptitle('Top: complementarity of ASTRA and density split. Bottom: scale dependence. '
                 'V = 0.125 $(h^{-1}{\\rm Gpc})^3$', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / f'compare_combined_scales{tag}.png', dpi=140)
    plt.close(fig)
    print(f'\nsaved data/compare_methods{tag}.json, plots/compare_ratios{tag}.png, plots/compare_combined_scales{tag}.png')


if __name__ == '__main__':
    main()
