#!/usr/bin/env python
"""
Cosmological derivatives from a regression over all HOD pairs of the
derivative grid (an alternative to the emulator).

For a +/- pair of cosmologies, every catalogue k of cosmology c (local
draws around the fiducial HOD h0, the Fisher-pair draw and the emulator draws
within --rmax of h0) gives a measurement d_c(h_k). We fit

    d_c(h) = a_c + G (h - h0)            (c = +, -; G shared: default)
    d_c(h) = a_c + G_c (h - h0)          (--separate-g)
    d_c(h) = a_c + G (h - h0) + Q (h - h0)^2    (--quadratic, diagonal terms)

by least squares, *subbox by subbox* (all catalogues share the phase and the
random points, so each subbox gives an estimate with cosmic variance largely
cancelled), and the derivative is (a_+ - a_-) / (theta_+ - theta_-): the
difference of the two cosmologies at exactly the fiducial HOD. Curvature in h
that is common to both cosmologies cancels in the difference.

Diagnostics: leave-one-catalogue-out prediction error of the model, and the
change of the derivatives between the shared-G, separate-G and quadratic fits
(an estimate of the modelling systematic).

    python scripts/pair_regression.py [--rmax 1.0] [--separate-g] [--quadratic] [--ndir 4]
Outputs: data/derivatives_regression{tag}.npz (per-subbox derivatives, same
format as scripts/derivatives.py), data/derivatives_regression{tag}.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A


def catalogues(cosmo, h0, rmax):
    """(hod, normalised h - h0) of the measured catalogues of a cosmology within rmax of h0 (line of sight z)."""
    out = []
    for f in sorted(A.DATA.glob(f'c{cosmo:03d}_hod*_z_losz.npz')):
        hod = int(f.stem.split('_')[1][3:])
        dh = A.hod_vector(cosmo, hod) - h0
        if np.linalg.norm(dh) <= rmax:
            out.append((hod, dh))
    return out


def design(dh, which, ncosmo, quadratic):
    """Rows: [one-hot cosmology | dh (shared or per cosmology) | dh^2]."""
    n, nh = len(dh), dh.shape[1]
    onehot = np.zeros((n, ncosmo))
    onehot[np.arange(n), which] = 1.
    return onehot, np.hstack([dh, dh**2]) if quadratic else dh


def fit(Y, dh, which, separate_g=False, quadratic=False):
    """
    Least-squares fit for every column of Y (n_catalogues, m).
    Returns the intercepts (2, m) and the leave-one-out residuals (n, m).
    """
    onehot, hterm = design(dh, which, 2, quadratic)
    if separate_g:
        X = np.hstack([onehot] + [hterm * (which == c)[:, None] for c in (0, 1)])
    else:
        X = np.hstack([onehot, hterm])
    if X.shape[0] <= X.shape[1]:
        raise ValueError(f'{X.shape[0]} catalogues for {X.shape[1]} coefficients')
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    # leave-one-out residuals of a linear least-squares fit: r_i / (1 - H_ii)
    H = X @ np.linalg.pinv(X)
    loo = (Y - X @ beta) / (1 - np.diag(H))[:, None]
    return beta[:2], loo


def hod_directions(k, stats=('gg', 'astra_rand_cross_q1', 'astra_rand_cross_q4', 'ds_rand_cross_q1',
                              'ds_rand_cross_q4', 'astra_data_cross_q4', 'ds_data_cross_q4')):
    """
    The k combinations of HOD parameters that change the statistics most, from
    the c000 calibration catalogues only (independent of the derivative grid):
    right singular vectors of the linear HOD response, each statistic weighted
    by its statistical error. Returns (k, 12) and the singular values.
    """
    import emulator as E
    G = []
    for st in stats:
        R = A.HODResponse([st])
        G.append(R.G / E.statistical_error(st)[:, None])
    _, sv, Vt = np.linalg.svd(np.concatenate(G), full_matrices=False)
    return Vt[:k], sv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rmax', type=float, default=1.0, help='max distance to h0 (prior-range units)')
    parser.add_argument('--separate-g', action='store_true')
    parser.add_argument('--quadratic', action='store_true')
    parser.add_argument('--ndir', type=int, default=0,
                        help='regress on the ndir most important HOD directions only (0: all 12 parameters)')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    V = None
    if args.ndir:
        V, sv = hod_directions(args.ndir)
        print(f'HOD directions: singular values {np.round(sv / sv[0], 3)} (first {args.ndir} kept)')
    sel, table = A.selection(), A.cosmology_table()
    h0 = A.hod_vector(sel['fiducial']['cosmo'], sel['fiducial']['hod'])
    stats = [str(s) for s in np.load(A.DATA / 'emulator_dataset.npz')['stat_names']]
    out, diag = {}, {}
    for param, pair in sel['pairs'].items():
        cp, cm = pair['plus']['cosmo'], pair['minus']['cosmo']
        step = table.loc[cp, A.PARAM_COLUMNS[param]] - table.loc[cm, A.PARAM_COLUMNS[param]]
        cats = [(cp, h, dh) for h, dh in catalogues(cp, h0, args.rmax)] + \
               [(cm, h, dh) for h, dh in catalogues(cm, h0, args.rmax)]
        which = np.array([0 if c == cp else 1 for c, _, _ in cats])
        dh = np.array([d for _, _, d in cats])
        if V is not None:
            dh = dh @ V.T
        print(f'{param}: {np.sum(which == 0)} + {np.sum(which == 1)} catalogues within {args.rmax} of h0, '
              f'median distance {np.median(np.linalg.norm(dh, axis=1)):.2f}')
        # per-subbox measurements: (n_catalogues, n_subbox, ndim) for each statistic
        data = {}
        ids_common = None
        for c, h, _ in cats:
            meta, st, ids = A.load(c, h, los=['z'])
            for name in stats:
                data.setdefault(name, []).append(A.data_vector(meta, st, [name])[1])
            ids_common = ids
        diag[param] = {'ncat_plus': int(np.sum(which == 0)), 'ncat_minus': int(np.sum(which == 1))}
        for name in stats:
            Y = np.array(data[name])                                   # (ncat, nsub, ndim)
            ncat, nsub, ndim = Y.shape
            a, loo = fit(Y.reshape(ncat, -1), dh, which, args.separate_g, args.quadratic)
            a = a.reshape(2, nsub, ndim)
            out[f'{param}/{name}'] = (a[0] - a[1]) / step                # per-subbox derivatives
            # LOO error of the subbox-mean prediction, relative to the spread of the measurements
            ym = Y.mean(1)
            loo_mean = loo.reshape(ncat, nsub, ndim).mean(1)
            diag[param][name] = float(np.sqrt((loo_mean**2).mean() / ((ym - ym.mean(0))**2).mean()))
        out[f'{param}/step'] = step
        print(f'  LOO error / HOD spread: gg {diag[param]["gg"]:.2f}, '
              f'astra_rand_cross_q4 {diag[param]["astra_rand_cross_q4"]:.2f}, '
              f'ds_rand_cross_q4 {diag[param]["ds_rand_cross_q4"]:.2f}')
    s = A.data_vector(meta, st, ['gg'])[0]
    out['s'] = s
    tag = args.tag or (('_sepG' if args.separate_g else '') + ('_quad' if args.quadratic else '')
                       + (f'_ndir{args.ndir}' if args.ndir else ''))
    np.savez(A.DATA / f'derivatives_regression{tag}.npz', **out)
    (A.DATA / f'derivatives_regression{tag}.json').write_text(json.dumps(dict(args=vars(args), diag=diag), indent=1))
    print(f'saved data/derivatives_regression{tag}.npz')


if __name__ == '__main__':
    main()
