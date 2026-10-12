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


def hod_vector(cosmo, hod):
    """HOD parameters of a catalogue, normalised to the prior range of the suite ([0, 1])."""
    import astra_ds
    sel = selection()
    lo, hi = np.array(sel['prior_lo']), np.array(sel['prior_hi'])
    h = astra_ds.read_hod_header(astra_ds.hod_path(cosmo, hod))
    return (np.array([h[k] for k in astra_ds.HOD_PARAMS]) - lo) / (hi - lo)


def calibration_hods():
    """c000 HOD draws of config/runs_hod_calibration.txt that have been measured."""
    out = []
    for line in (CONFIG / 'runs_hod_calibration.txt').read_text().split('\n'):
        if line.strip():
            c, h, space, los = line.split()
            if (DATA / f'c{int(c):03d}_hod{int(h):03d}_{space}_los{los}.npz').exists():
                out.append(int(h))
    return out


class HODResponse:
    """
    Linear response of a data vector to the HOD parameters at the fiducial cosmology,

        d(h) - d(h_fid) = G (h - h_fid),     h normalised to the prior range,

    fitted by least squares on the c000 calibration catalogues. Each catalogue
    is differenced with the fiducial subbox by subbox (same phase, same randoms)
    before averaging, so that cosmic variance cancels.
    """

    def __init__(self, stats, space='z', los=('z',), **dv_kwargs):
        sel = selection()
        fid = sel['fiducial']
        self.h_fid = hod_vector(fid['cosmo'], fid['hod'])
        meta, fstats, fids = load(fid['cosmo'], fid['hod'], space, los)
        _, xf = data_vector(meta, fstats, stats, **dv_kwargs)
        dh, dy = [], []
        self.hods = calibration_hods()
        for hod in self.hods:
            m, st, ids = load(0, hod, space, los)
            x = data_vector(m, st, stats, **dv_kwargs)[1]
            dy.append(paired_derivative(x, xf, ids, fids, 1.).mean(0))
            dh.append(hod_vector(0, hod) - self.h_fid)
        self.dh, self.dy = np.array(dh), np.array(dy)
        if len(self.hods) <= self.dh.shape[1]:
            raise ValueError(f'only {len(self.hods)} calibration catalogues for {self.dh.shape[1]} HOD parameters')
        self.G = np.linalg.lstsq(self.dh, self.dy, rcond=None)[0].T          # (ndim, nhod)
        # leave-one-out prediction error, relative to the size of the response
        loo = []
        for k in range(len(self.hods)):
            keep = np.arange(len(self.hods)) != k
            Gk = np.linalg.lstsq(self.dh[keep], self.dy[keep], rcond=None)[0].T
            loo.append(self.dy[k] - Gk @ self.dh[k])
        self.loo_residual = np.array(loo)

    def predict(self, h):
        return self.G @ (np.asarray(h) - self.h_fid)

    def correction(self, h_plus, h_minus):
        """Predicted difference of the data vector due to the HOD mismatch of a pair."""
        return self.predict(h_plus) - self.predict(h_minus)


class GPHODResponse(HODResponse):
    """
    Non-linear HOD response: the calibration differences are compressed with a
    PCA and each component is modelled by a Gaussian process (anisotropic RBF
    kernel, one length scale per HOD parameter) in the normalised HOD space.
    The linear-model attributes (G, loo_residual) are replaced by the GP ones;
    G is the gradient of the GP at the fiducial HOD.
    """

    def __init__(self, stats, ncomp=8, eps=0.02, loo=True, **kwargs):
        super().__init__(stats, **kwargs)
        self.ncomp = min(ncomp, len(self.hods) - 1)
        self.models = self._fit(self.dh, self.dy)
        nh = self.dh.shape[1]
        self.G = np.array([(self._predict(self.models, eps * np.eye(nh)[j])
                            - self._predict(self.models, -eps * np.eye(nh)[j])) / (2 * eps)
                           for j in range(nh)]).T
        if loo:   # leave-one-out validation (slow: refits all the GPs for each catalogue)
            res = []
            for k in range(len(self.hods)):
                keep = np.arange(len(self.hods)) != k
                res.append(self.dy[k] - self._predict(self._fit(self.dh[keep], self.dy[keep]), self.dh[k]))
            self.loo_residual = np.array(res)

    def _fit(self, dh, dy):
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        pca = PCA(self.ncomp).fit(dy)
        z = pca.transform(dy)
        gps = []
        for j in range(self.ncomp):
            kernel = ConstantKernel(1.) * RBF(np.ones(dh.shape[1]), (1e-2, 1e2)) + WhiteKernel(1e-3)
            gps.append(GaussianProcessRegressor(kernel, normalize_y=True).fit(dh, z[:, j]))
        # the response vanishes at the fiducial by construction: subtract the GP value there
        zero = np.array([gp.predict(np.zeros((1, dh.shape[1])))[0] for gp in gps])
        return pca, gps, zero

    @staticmethod
    def _predict(models, dh):
        pca, gps, zero = models
        z = np.array([gp.predict(np.atleast_2d(dh))[0] for gp in gps]) - zero
        return pca.inverse_transform(z[None])[0] - pca.inverse_transform(np.zeros((1, len(gps))))[0]

    def predict(self, h):
        return self._predict(self.models, np.asarray(h) - self.h_fid)


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
