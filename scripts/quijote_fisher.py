#!/usr/bin/env python
"""
Fisher forecasts with Quijote: clean derivatives, ASTRA split vs density split.

* derivatives: paired central differences over the realisations shared by the
  + and - sets (same initial seeds), for Om, Ob, h, ns, s8; for Mnu the
  standard three-point forward difference on Mnu = 0 (fiducial_ZA), 0.1, 0.2 eV:
      d/dMnu = (-3 f_0 + 4 f_0.1 - f_0.2) / (2 x 0.1);
* covariance: fiducial realisations, Hartlap-corrected, volume (1 Gpc/h)^3;
* noisy derivatives bias the standard Fisher estimate high; we quote the
  combined estimator of Coulton & Wandelt (2023), geometric mean of the
  standard and of a compressed estimator (biased low), and both bounds;
  the convergence is tested by recomputing with fewer realisations.

    python scripts/quijote_fisher.py [--report]  ->  data/quijote_fisher.json, plots/quijote_*.png
With --report, also writes data/quijote/CONTINUE if the constraints are not
converged with respect to the number of fiducial realisations (used to decide
whether the remaining fiducial realisations should be processed).
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

QDATA = ROOT / 'data' / 'quijote'
FIDUCIAL = {'Om': 0.3175, 'Ob': 0.049, 'h': 0.6711, 'ns': 0.9624, 's8': 0.834, 'Mnu': 0.}
STEPS = {'Om': ('Om_p', 'Om_m', 0.02), 'Ob': ('Ob2_p', 'Ob2_m', 0.004), 'h': ('h_p', 'h_m', 0.04),
         'ns': ('ns_p', 'ns_m', 0.04), 's8': ('s8_p', 's8_m', 0.03)}
LABELS = {'Om': r'\Omega_m', 'Ob': r'\Omega_b', 'h': 'h', 'ns': 'n_s', 's8': r'\sigma_8', 'Mnu': r'M_\nu'}
PARAMS = list(FIDUCIAL)


def q(method, pop='rand', kinds=('cross',), qs=(1, 4)):
    return [f'{method}_{pop}_{k}_q{i}' for k in kinds for i in qs]


VECTORS = {'2PCF': ['gg'], 'ASTRA rand': q('astra'), 'DS rand': q('ds'),
           'ASTRA gal': q('astra', 'data'), 'DS gal': q('ds', 'data'),
           'ASTRA rand + auto': q('astra', kinds=('cross', 'auto')), 'DS rand + auto': q('ds', kinds=('cross', 'auto')),
           'ASTRA Q1 (voids)': q('astra', qs=(1,), kinds=('cross', 'auto')), 'DS Q1 (voids)': q('ds', qs=(1,), kinds=('cross', 'auto')),
           'ASTRA Q4 (clusters)': q('astra', qs=(4,), kinds=('cross', 'auto')), 'DS Q4 (clusters)': q('ds', qs=(4,), kinds=('cross', 'auto')),
           'ASTRA + DS': q('astra') + q('ds'), '2PCF + ASTRA': ['gg'] + q('astra'), '2PCF + DS': ['gg'] + q('ds'),
           '2PCF + ASTRA + DS': ['gg'] + q('astra') + q('ds')}


def load_set(name, stats_needed):
    """{realisation: {stat: (nell, ns)}} for all processed realisations of a set."""
    out, meta = {}, None
    for f in sorted((QDATA / name).glob('*.npz'), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        if not f.stem.isdigit():
            continue
        d = np.load(f)
        if meta is None:
            meta = {'s': d['s'], 'ells': d['ells']}
        out[int(f.stem)] = {st: d[st][0] for st in stats_needed}
    return out, meta


REBIN = 2
ELLS = (0, 2)
KEEP = None   # parameters kept in the Fisher matrix (None: all)


def vec(meta, real, stats, smin):
    st = {k: np.array([real[k]]) for k in stats}
    return A.data_vector(meta, st, stats, ells=ELLS, smin=smin, rebin=REBIN)[1][0]


def compute(sets, meta, stats, smin, nderiv=None, ncov=None):
    fid = sets['fiducial']
    ids = sorted(fid)[:ncov]
    cov_samples = np.array([vec(meta, fid[i], stats, smin) for i in ids])
    prec, _, hartlap = A.precision(cov_samples)
    derivs = {}
    for p, (sp, sm, step) in STEPS.items():
        common = sorted(set(sets[sp]) & set(sets[sm]))[:nderiv]
        derivs[p] = np.array([(vec(meta, sets[sp][i], stats, smin) - vec(meta, sets[sm][i], stats, smin)) / step
                              for i in common])
    common = sorted(set(sets['fiducial_ZA']) & set(sets['Mnu_p']) & set(sets['Mnu_pp']))[:nderiv]
    derivs['Mnu'] = np.array([(-3 * vec(meta, sets['fiducial_ZA'][i], stats, smin)
                               + 4 * vec(meta, sets['Mnu_p'][i], stats, smin)
                               - vec(meta, sets['Mnu_pp'][i], stats, smin)) / 0.2 for i in common])
    if KEEP:
        derivs = {p: derivs[p] for p in KEEP}
    params = list(derivs)
    D = np.array([derivs[p].mean(0) for p in params])                 # (npar, ndim)
    F_s = D @ prec @ D.T                                               # standard: biased high by derivative noise
    F_deb, noise = A.fisher(derivs, prec)                               # standard with the noise bias subtracted
    # compressed estimator (Coulton & Wandelt 2023): compress with the derivatives of one half of the
    # simulations, measure the Fisher information with the other half; biased low by the noise
    n = min(len(d) for d in derivs.values())
    half = n // 2
    DA = np.array([derivs[p][:half].mean(0) for p in params])
    DB = np.array([derivs[p][half:n].mean(0) for p in params])
    W = prec @ DA.T                                                    # (ndim, npar)
    Ct = W.T @ np.linalg.inv(prec) @ W
    TB = W.T @ DB.T                                                    # (npar, npar)
    F_c = TB.T @ np.linalg.inv(Ct) @ TB

    def sig(F):
        return np.sqrt(np.diag(np.linalg.inv(F))) if np.all(np.linalg.eigvalsh(F) > 0) else np.full(len(F), np.nan)
    s_s, s_c = sig(F_s), sig(F_c)
    s_comb = np.sqrt(s_s * s_c)                                       # geometric mean: nearly unbiased
    return dict(sigma=s_comb.tolist(), sigma_standard=s_s.tolist(), sigma_compressed=s_c.tolist(),
                sigma_debiased=sig(F_deb).tolist(), noise_frac=noise.tolist(), ndim=cov_samples.shape[1],
                ncov=len(ids), nderiv=n, hartlap=hartlap, cov=np.linalg.inv(F_s).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smin', type=float, default=20.)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--rebin', type=int, default=2, help='5 Mpc/h bins merged (2: 10 Mpc/h)')
    parser.add_argument('--params', nargs='+', default=None, help='parameters kept (others fixed)')
    parser.add_argument('--no-convergence', action='store_true')
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    global REBIN, KEEP, PARAMS, ELLS
    REBIN, KEEP, ELLS = args.rebin, args.params, tuple(args.ells)
    if KEEP:
        PARAMS = KEEP
    stats = sorted({s for v in VECTORS.values() for s in v})
    names = ['fiducial', 'fiducial_ZA', 'Mnu_p', 'Mnu_pp'] + [s for p in STEPS.values() for s in p[:2]]
    sets, meta = {}, None
    for n in names:
        sets[n], m = load_set(n, stats)
        meta = meta or m
        print(f'{n}: {len(sets[n])} realisations')
    res = {name: compute(sets, meta, v, args.smin) for name, v in VECTORS.items()}

    print(f'\nmarginalised 1-sigma, V = 1 (Gpc/h)^3, s >= {args.smin:g}: combined [standard, compressed]')
    print(f'{"data vector":22s} {"ndim":>4s} ' + ' '.join(f'{p:>24s}' for p in PARAMS))
    for name, r in res.items():
        print(f'{name:22s} {r["ndim"]:4d} ' + ' '.join(f'{s:8.2e} [{a:.1e},{b:.1e}]' for s, a, b in
                                                      zip(r['sigma'], r['sigma_standard'], r['sigma_compressed'])))
    print('\nsigma_DS / sigma_ASTRA   [> 1: ASTRA more constraining]')
    for a, d in [('ASTRA rand', 'DS rand'), ('ASTRA gal', 'DS gal'), ('ASTRA rand + auto', 'DS rand + auto'),
                 ('ASTRA Q1 (voids)', 'DS Q1 (voids)'), ('ASTRA Q4 (clusters)', 'DS Q4 (clusters)'),
                 ('2PCF + ASTRA', '2PCF + DS')]:
        print(f'{a:22s} ' + ' '.join(f'{x:6.2f}' for x in np.array(res[d]['sigma']) / np.array(res[a]['sigma'])))

    # convergence with the number of realisations
    conv = {}
    nfid, nder = len(sets['fiducial']), res['2PCF']['nderiv']
    for name in ([] if args.no_convergence else ['2PCF', 'ASTRA rand', 'DS rand', '2PCF + ASTRA + DS']):
        conv[name] = {'ncov': {}, 'nderiv': {}}
        for nc in [nfid // 4, nfid // 2, nfid]:
            conv[name]['ncov'][nc] = compute(sets, meta, VECTORS[name], args.smin, ncov=nc)['sigma']
        for nd in [nder // 4, nder // 2, nder]:
            conv[name]['nderiv'][nd] = compute(sets, meta, VECTORS[name], args.smin, nderiv=nd)['sigma']
    print('\nconvergence: sigma(N/2) / sigma(N) - 1 for the covariance (N fiducial) and the derivatives (N pairs)')
    worst_cov, worst_der = 0., 0.
    for name, c in conv.items():
        kc, kd = sorted(c['ncov']), sorted(c['nderiv'])
        dc = np.array(c['ncov'][kc[1]]) / np.array(c['ncov'][kc[2]]) - 1
        dd = np.array(c['nderiv'][kd[1]]) / np.array(c['nderiv'][kd[2]]) - 1
        worst_cov, worst_der = max(worst_cov, np.nanmax(np.abs(dc))), max(worst_der, np.nanmax(np.abs(dd)))
        print(f'  {name:20s} cov {np.round(dc, 3)}  deriv {np.round(dd, 3)}')
    (A.DATA / f'quijote_fisher{args.tag}.json').write_text(json.dumps(
        dict(results=res, convergence={k: {kk: {str(n): s for n, s in vv.items()} for kk, vv in v.items()}
                                       for k, v in conv.items()}, nfid=nfid, nderiv=nder), indent=1))

    # bar chart of the ratios
    pairs = [('ASTRA rand', 'DS rand'), ('ASTRA gal', 'DS gal'), ('ASTRA rand + auto', 'DS rand + auto'),
             ('ASTRA Q1 (voids)', 'DS Q1 (voids)'), ('ASTRA Q4 (clusters)', 'DS Q4 (clusters)'), ('2PCF + ASTRA', '2PCF + DS')]
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(17, 3.8), sharey=True)
    for k, (ax, p) in enumerate(zip(axes, PARAMS)):
        vals = [res[d]['sigma'][k] / res[a]['sigma'][k] for a, d in pairs]
        ax.barh(range(len(pairs)), vals, color=['#d95f02' if v > 1 else '#1b9e77' for v in vals])
        ax.axvline(1, color='k', lw=0.8)
        ax.set_title(f'${LABELS[p]}$')
        ax.set_xlabel(r'$\sigma_{\rm DS}/\sigma_{\rm ASTRA}$')
    axes[0].set_yticks(range(len(pairs)), [a.replace('ASTRA ', '') for a, _ in pairs], fontsize=8)
    fig.suptitle(f'Quijote halos, Fisher with clean derivatives ({nder} pairs, {nfid} fiducial realisations)', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / f'quijote_ratios{args.tag}.png', dpi=140)

    if args.report:
        flag = QDATA / 'CONTINUE'
        if worst_cov > 0.05 and np.isfinite(worst_cov):
            flag.write_text(f'covariance not converged: max |sigma(N/2)/sigma(N) - 1| = {worst_cov:.3f}\n')
            print(f'\n-> CONTINUE: covariance not converged ({worst_cov:.1%}); process the remaining fiducial realisations')
        else:
            flag.unlink(missing_ok=True)
            print(f'\n-> STOP: covariance converged to {worst_cov:.1%}; derivatives {worst_der:.1%}')


if __name__ == '__main__':
    main()
