"""
Shared tools for the analysis of the measurements: loading, data vectors,
covariance, paired derivatives, Fisher matrices.

A measurement file data/c{cosmo}_hod{hod}_{space}_los{los}.npz holds, for each
statistic, an array (nsubbox, nell, ns). A *sample* is one (los, subbox):
the covariance uses all samples of the fiducial, the derivatives difference
the +/- cosmologies sample by sample (same subbox, same line of sight).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
CONFIG = ROOT / 'config'

PARAMS = ['omega_b', 'omega_cdm', 'n_s', 'sigma8']
PARAM_COLUMNS = {'omega_b': 'omega_b', 'omega_cdm': 'omega_cdm', 'n_s': 'n_s', 'sigma8': 'sigma8_m'}
PARAM_LABELS = {'omega_b': r'\omega_b', 'omega_cdm': r'\omega_{\rm cdm}', 'n_s': r'n_s', 'sigma8': r'\sigma_8'}

# Data vectors compared in the paper: name -> list of statistics.
# Quartiles: Q1 (voids) and Q4 (clusters) carry most of the signal; Q2-Q3 are
# close to the mean field and mostly add dimension (Hartlap penalty). With at
# most 3 x 64 covariance samples, the default vectors use the quartile x galaxy
# crosses only; the "+ auto" variants add the quartile autos.


def _q(method, pop, kinds=('cross',), qs=(1, 4)):
    return [f'{method}_{pop}_{k}_q{q}' for k in kinds for q in qs]


DATA_VECTORS = {
    '2PCF': ['gg'],
    'ASTRA rand': _q('astra', 'rand'),
    'DS rand': _q('ds', 'rand'),
    'ASTRA gal': _q('astra', 'data'),
    'DS gal': _q('ds', 'data'),
    '2PCF + ASTRA rand': ['gg'] + _q('astra', 'rand'),
    '2PCF + DS rand': ['gg'] + _q('ds', 'rand'),
    'ASTRA rand + auto': _q('astra', 'rand', ('cross', 'auto')),
    'DS rand + auto': _q('ds', 'rand', ('cross', 'auto')),
}


def selection():
    return json.loads((CONFIG / 'hod_selection.json').read_text())


def cosmology_table():
    df = pd.read_csv(CONFIG / 'cosmologies.csv', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df.index = df['root'].str.extract(r'cosm(\d+)')[0].astype(int)
    return df


def measurement_files(cosmo, hod, space='z'):
    """{los: path} for all lines of sight measured for this catalogue."""
    files = sorted(DATA.glob(f'c{cosmo:03d}_hod{hod:03d}_{space}_los[xyz].npz'))
    return {f.stem[-1]: f for f in files}


def load(cosmo, hod, space='z', los=None):
    """
    Measurements of one catalogue, stacked over lines of sight.
    Returns (meta, {stat: array (nsample, nell, ns)}, sample_ids) where
    sample_ids are (los, subbox) tuples identifying each sample.
    """
    files = measurement_files(cosmo, hod, space)
    if los is not None:
        files = {k: v for k, v in files.items() if k in los}
    if not files:
        raise FileNotFoundError(f'no measurement for c{cosmo:03d}_hod{hod:03d}_{space}')
    stats, ids, meta = {}, [], None
    for l, fn in sorted(files.items()):
        d = np.load(fn)
        if meta is None:
            meta = {k: d[k] for k in ('s', 'ells', 'sedges', 'stat_names')}
        for name in meta['stat_names']:
            stats.setdefault(name, []).append(d[name])
        ids += [(l, int(b)) for b in d['subboxes']]
    stats = {k: np.concatenate(v) for k, v in stats.items()}
    return meta, stats, ids


def data_vector(meta, stats, names, ells=(0, 2), smin=20., smax=150., rebin=2):
    """
    Concatenate statistics into data vectors (nsample, ndim).

    Multipoles are rebinned by averaging `rebin` adjacent bins (5 -> 10 Mpc/h
    by default), then bins with smin <= s <= smax are kept.
    """
    s = meta['s']
    nb = len(s) // rebin * rebin
    s = s[:nb].reshape(-1, rebin).mean(-1)
    keep = (s >= smin) & (s <= smax)
    il = [list(meta['ells']).index(ell) for ell in ells]
    blocks = []
    for name in names:
        x = stats[name][:, il, :nb]
        x = x.reshape(x.shape[0], len(il), -1, rebin).mean(-1)[..., keep]
        blocks.append(x.reshape(x.shape[0], -1))
    return s[keep], np.concatenate(blocks, axis=1)


def paired_derivative(plus, minus, ids_plus, ids_minus, step):
    """Per-sample derivatives (x+ - x-) / step on the samples common to both runs."""
    common = sorted(set(ids_plus) & set(ids_minus))
    ip = [ids_plus.index(i) for i in common]
    im = [ids_minus.index(i) for i in common]
    return (plus[ip] - minus[im]) / step


def precision(samples):
    """Hartlap-corrected inverse covariance from (nsample, ndim) samples."""
    n, p = samples.shape
    if p >= n - 2:
        raise ValueError(f'data vector ({p}) too large for {n} samples')
    cov = np.cov(samples, rowvar=False)
    hartlap = (n - p - 2) / (n - 1)
    return hartlap * np.linalg.inv(cov), cov, hartlap


def fisher(derivs, prec):
    """
    Fisher matrix from per-sample derivatives {param: (nsample, ndim)}.

    The mean derivative is noisy, which biases dᵀC⁻¹d high by Tr(C⁻¹ Σ_ij),
    Σ_ij = cov(mean d_i, mean d_j). This noise term is subtracted; its
    fraction of the raw Fisher diagonal is returned as a diagnostic.
    """
    params = list(derivs)
    mean = np.array([derivs[p].mean(0) for p in params])
    n = min(len(derivs[p]) for p in params)
    raw = mean @ prec @ mean.T
    noise = np.zeros_like(raw)
    for a, pa in enumerate(params):
        for b, pb in enumerate(params):
            da, db = derivs[pa][:n], derivs[pb][:n]
            cab = (da - da.mean(0)).T @ (db - db.mean(0)) / (n - 1) / n
            noise[a, b] = np.trace(prec @ cab)
    return raw - noise, np.diag(noise) / np.diag(raw)
