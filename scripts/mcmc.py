#!/usr/bin/env python
"""
Simulation-based inference with the emulator (SUNBIRD-like): posterior of the
8 cosmological and 12 HOD parameters for a measured data vector.

* likelihood : Gaussian, d - emulator(theta), covariance = statistical covariance
  of the data volume + emulator error (variance of the test residuals);
* statistical covariance: from the 1786 official small boxes (--cov covbox,
  one (500 Mpc/h)^3 volume each) or from the 192 subboxes of the fiducial
  emulator catalogue (--cov subbox), rescaled to the volume of the data by
  --volume-factor (64 for a full 2 Gpc/h box), Hartlap-corrected;
* priors : uniform over the range of the emulator training set;
* data  : --data official:PHASE[:SUBBOX] (official DESI base box of c000, mean
  of its 64 subboxes or one subbox; true HOD = the official best fit) or
  --data cosmo:COSMO_HOD[:SUBBOX] (a catalogue of the emulator suite, e.g. a
  held-out test cosmology). With one subbox use --volume-factor 1;
* sampler: emcee, the emulator being evaluated for all walkers at once.

    python scripts/mcmc.py --data official:000 --vector ASTRA --cov covbox
Outputs: data/chains/{tag}.npz, plots/mcmc_{tag}.png
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import emcee

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A
import emulator as E
import astra_ds


def q(method, pop='rand', kinds=('cross',), qs=(1, 4)):
    return [f'{method}_{pop}_{k}_q{i}' for k in kinds for i in qs]


VECTORS = {'2PCF': ['gg'], 'ASTRA': q('astra'), 'DS': q('ds'), 'ASTRA+DS': q('astra') + q('ds'),
           '2PCF+ASTRA': ['gg'] + q('astra'), '2PCF+DS': ['gg'] + q('ds'),
           '2PCF+ASTRA+DS': ['gg'] + q('astra') + q('ds'),
           'ASTRA+auto': q('astra', kinds=('cross', 'auto')), 'DS+auto': q('ds', kinds=('cross', 'auto'))}
OFFICIAL_HOD = {'LOGM_CUT': 12.78128755928713, 'LOGM1': 13.71839856014125, 'SIGMA': 0.2533790608887552,
                'ALPHA': 1.43145922093541, 'KAPPA': 0.2777860701866452, 'ALPHA_C': 0.1402734659580316,
                'ALPHA_S': 0.7860565032680378, 'S': 0., 'ACENT': 0., 'ASAT': 0., 'BCENT': 0., 'BSAT': 0.}


def covariance_samples(kind, stats):
    """(nsample, ndim) statistical samples of one (500 Mpc/h)^3 volume."""
    if kind == 'covbox':
        files = sorted((A.DATA / 'official').glob('covbox_ph*.npz'))
        st = {name: [] for name in stats}
        for f in files:
            d = np.load(f)
            meta = {'s': d['s'], 'ells': d['ells']}
            for name in stats:
                st[name].append(d[name][0])
        return A.data_vector(meta, {k: np.array(v) for k, v in st.items()}, stats)[1]
    sel = A.selection()['fiducial']
    meta, fst, _ = A.load(sel['cosmo'], sel['hod'])
    return A.data_vector(meta, fst, stats)[1]


def data_and_truth(spec, stats):
    """Measured data vector (mean over the subboxes) and true parameters (or None)."""
    parts = spec.split(':')
    kind, ident = parts[0], parts[1]
    sub = int(parts[2]) if len(parts) > 2 else None      # one subbox (volume 0.125 (Gpc/h)^3) or all (mean)
    pick = (lambda x: x[sub]) if sub is not None else (lambda x: x.mean(0))
    table = A.cosmology_table()
    if kind == 'official':
        d = np.load(A.DATA / 'official' / f'cubicbox_ph{ident}.npz')
        meta = {'s': d['s'], 'ells': d['ells']}
        x = pick(A.data_vector(meta, {k: d[k] for k in stats}, stats)[1])
        truth = [float(table.loc[0, c]) for c in E.COSMO_INPUTS] + [OFFICIAL_HOD[k] for k in astra_ds.HOD_PARAMS]
        return x, np.array(truth)
    cosmo, hod = map(int, ident.split('_'))
    meta, st, _ = A.load(cosmo, hod, los=['z'])
    x = pick(A.data_vector(meta, st, stats)[1])
    h = astra_ds.read_hod_header(astra_ds.hod_path(cosmo, hod))
    truth = [float(table.loc[cosmo, c]) for c in E.COSMO_INPUTS] + [h[k] for k in astra_ds.HOD_PARAMS]
    return x, np.array(truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='official:000')
    parser.add_argument('--vector', default='ASTRA', choices=list(VECTORS))
    parser.add_argument('--cov', default='covbox', choices=['covbox', 'subbox'])
    parser.add_argument('--volume-factor', type=float, default=64., help='data volume / (500 Mpc/h)^3')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models_pairs')
    parser.add_argument('--validation-tag', default='_pairs')
    parser.add_argument('--nwalkers', type=int, default=96)
    parser.add_argument('--nsteps', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    stats = VECTORS[args.vector]
    emus = [E.Emulator.load(st, args.models_dir) for st in stats]
    val = np.load(A.DATA / f'emulator_validation{args.validation_tag}.npz')

    samples = covariance_samples(args.cov, stats)
    n, p = samples.shape
    C = np.cov(samples, rowvar=False) / args.volume_factor
    C_emu = np.diag(np.concatenate([val[f'{st}/res'].var(0) for st in stats]))
    prec = (n - p - 2) / (n - 1) * np.linalg.inv(C + C_emu)
    data, truth = data_and_truth(args.data, stats)

    X = np.load(A.DATA / 'emulator_dataset.npz')
    names = list(X['x_names'])
    cols = [names.index(c) for c in E.COSMO_INPUTS] + [names.index(h) for h in names[10:]]
    lo, hi = X['X'][:, cols].min(0), X['X'][:, cols].max(0)
    labels = E.COSMO_INPUTS + astra_ds.HOD_PARAMS
    ptorch = torch.as_tensor(prec)
    dtorch = torch.as_tensor(data)

    def log_prob(theta):
        theta = np.atleast_2d(theta)
        ok = np.all((theta >= lo) & (theta <= hi), axis=1)
        lp = np.full(len(theta), -np.inf)
        if ok.any():
            with torch.no_grad():
                t = torch.as_tensor(theta[ok])
                pred = torch.cat([emu(t) for emu in emus], dim=1)
                r = pred - dtorch
                lp[ok] = (-0.5 * torch.einsum('ni,ij,nj->n', r, ptorch, r)).numpy()
        return lp

    rng = np.random.default_rng(args.seed)
    start = truth if truth is not None else 0.5 * (lo + hi)
    p0 = np.clip(start + 0.01 * (hi - lo) * rng.standard_normal((args.nwalkers, len(lo))), lo, hi)
    sampler = emcee.EnsembleSampler(args.nwalkers, len(lo), log_prob, vectorize=True)
    sampler.run_mcmc(p0, args.nsteps, progress=False)
    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau = np.full(len(lo), np.nan)
    burn = int(min(args.nsteps // 2, 5 * np.nanmax(tau))) if np.isfinite(np.nanmax(tau)) else args.nsteps // 2
    chain = sampler.get_chain(discard=burn, flat=True, thin=max(1, int(np.nanmin(tau) // 2) if np.isfinite(np.nanmin(tau)) else 1))
    tag = f'{args.data.replace(":", "_")}_{args.vector}_{args.cov}_v{args.volume_factor:g}'
    (A.DATA / 'chains').mkdir(exist_ok=True)
    np.savez(A.DATA / 'chains' / f'{tag}.npz', chain=chain, labels=np.array(labels), truth=truth, tau=tau,
             acceptance=sampler.acceptance_fraction.mean(), lo=lo, hi=hi)
    summary = {}
    print(f'{tag}: acceptance {sampler.acceptance_fraction.mean():.2f}, max autocorr time {np.nanmax(tau):.0f} steps, '
          f'{len(chain)} samples')
    for k, lab in enumerate(labels[:8]):
        m, s = chain[:, k].mean(), chain[:, k].std()
        summary[lab] = {'mean': float(m), 'std': float(s), 'truth': float(truth[k]),
                        'bias_sigma': float((m - truth[k]) / s)}
        print(f'  {lab:10s} {m:9.4f} +- {s:8.4f}   truth {truth[k]:9.4f}   ({(m - truth[k]) / s:+.2f} sigma)')
    (A.DATA / 'chains' / f'{tag}.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
