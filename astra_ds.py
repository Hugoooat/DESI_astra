"""
astra_ds — environment-split clustering on AbacusSummit HOD mocks.

Compares two ways of splitting a tracer field into density environments:

* ASTRA split   : Delaunay tessellation of galaxies + uniform randoms;
                  r = (N_data - N_rand) / (N_data + N_rand) over the Delaunay
                  neighbours of each point.
* Density split : galaxies painted on a mesh (CIC), Gaussian-smoothed
                  (R = 10 Mpc/h), density contrast read at each point
                  (Paillas et al. 2023, 2024).

Both estimators are evaluated on the *full periodic* 2 Gpc/h box (no edge
effects), on the same set of random points. Clustering is then measured in
the 64 non-overlapping 500 Mpc/h subboxes, treated as open (non-periodic)
volumes with a Landy-Szalay estimator and a fixed line of sight (z).

For each subbox and each method we measure, for each quartile Q1..Q4
(Q1 = most underdense):
    data  quartiles : galaxies ranked by their own environment value
    rand  quartiles : random points ranked by their environment value
                      (= the "query points" of standard density split)
and for each population the quartile auto-correlation (QQ) and the
cross-correlation with the full galaxy sample (QG), plus the galaxy 2PCF.
"""

import time
import logging
from pathlib import Path

import numpy as np
import fitsio
from scipy import fft
from scipy.spatial import Delaunay
from pycorr import TwoPointCounter, LandySzalayTwoPointEstimator

logger = logging.getLogger('astra_ds')

HOD_DIR = Path('/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior')
FULL_BOX = 2000.
HOD_PARAMS = ['LOGM_CUT', 'LOGM1', 'SIGMA', 'ALPHA', 'KAPPA', 'ALPHA_C',
              'ALPHA_S', 'S', 'ACENT', 'ASAT', 'BCENT', 'BSAT']

METHODS = ('astra', 'ds')
POPS = ('data', 'rand')
N_QUANTILES = 4


# ---------------------------------------------------------------------------
# Catalogues
# ---------------------------------------------------------------------------

def hod_path(cosmo, hod, phase=0, seed=0):
    return HOD_DIR / f'c{cosmo:03d}_ph{phase:03d}' / f'seed{seed}' / f'hod{hod:03d}.fits'


def read_hod_header(path):
    header = fitsio.read_header(str(path), ext=1)
    return {k: float(header[k]) for k in HOD_PARAMS + ['Q_PAR', 'Q_PERP']}


def load_catalog(path, space='z', los='z'):
    """
    Galaxy positions wrapped into [0, L), with the line of sight `los` moved to
    the last axis: (y, z, x) for los = x, (z, x, y) for los = y. Downstream
    everything (ASTRA, density split, pair counts) then uses los = 'z'.

    X_PERP/Y_PERP/Z_PERP and {X,Y,Z}_RSD are already Alcock-Paczynski rescaled
    in these catalogues: they are used as is, the box is 2000/q_perp across
    the line of sight and 2000/q_par (resp. q_perp in real space) along it.
    """
    axes = {'x': 'YZX', 'y': 'ZXY', 'z': 'XYZ'}[los]
    columns = [f'{a}_PERP' for a in axes[:2]] + [f'{axes[2]}_RSD' if space == 'z' else f'{axes[2]}_PERP']
    data, header = fitsio.read(str(path), columns=columns, header=True)
    q_par, q_perp = header['Q_PAR'], header['Q_PERP']
    q_z = q_par if space == 'z' else q_perp
    boxsize = FULL_BOX / np.array([q_perp, q_perp, q_z])
    pos = np.column_stack([data[c] for c in columns]).astype('f8')
    pos = (pos + boxsize / 2.) % boxsize   # [-L/2, L/2) -> [0, L), wraps RSD overflow
    return pos, boxsize


def uniform_randoms(n, boxsize, seed):
    """Uniform points in [0, L). Drawn in the unit cube then rescaled, so the
    same seed gives the *same* points (up to AP rescaling) in every cosmology."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)) * boxsize


# ---------------------------------------------------------------------------
# Subboxes
# ---------------------------------------------------------------------------

def subbox_corners(boxsize, nside=4):
    """Lower corners of the nside^3 subboxes tiling the box, and their size."""
    size = boxsize / nside
    grid = np.stack(np.meshgrid(*[np.arange(nside)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
    return grid * size, size


def extract(positions, lo, size, boxsize, pad=0.):
    """
    Points inside the (periodically wrapped) region [lo - pad, lo + size + pad).

    Returns the indices of the selected points, their positions in the
    subbox frame (core = [0, size)) and a mask of the points in the core.
    """
    local = (positions - lo + pad) % boxsize - pad
    keep = np.all((local >= -pad) & (local < size + pad), axis=1)
    local = local[keep]
    core = np.all((local >= 0.) & (local < size), axis=1)
    return np.flatnonzero(keep), local, core


def periodic_pad(positions, boxsize, pad):
    """
    A whole periodic box treated as one subbox: the points plus their periodic
    images within `pad` of the faces. Returns (indices into positions, padded
    positions, core mask), with the original points first.
    """
    boxsize = np.asarray(boxsize)
    idx, pts = [np.arange(len(positions))], [positions]
    for shift in np.array(np.meshgrid(*[[-1, 0, 1]] * 3, indexing='ij')).reshape(3, -1).T:
        if not shift.any():
            continue
        m = np.ones(len(positions), dtype=bool)
        for ax in range(3):
            if shift[ax] == 1:
                m &= positions[:, ax] < pad
            elif shift[ax] == -1:
                m &= positions[:, ax] >= boxsize[ax] - pad
        idx.append(np.flatnonzero(m))
        pts.append(positions[m] + shift * boxsize)
    core = np.zeros(sum(len(i) for i in idx), dtype=bool)
    core[:len(positions)] = True
    return np.concatenate(idx), np.concatenate(pts), core


# ---------------------------------------------------------------------------
# Environment estimators
# ---------------------------------------------------------------------------

def astra_r(data, randoms):
    """
    ASTRA estimator for every data and random point.

    r = (N_data - N_rand) / (N_data + N_rand) over the Delaunay neighbours of
    the point in the joint data + randoms tessellation.
    """
    points = np.concatenate([data, randoms])
    is_data = np.zeros(len(points), dtype='i4')
    is_data[:len(data)] = 1
    indptr, indices = Delaunay(points).vertex_neighbor_vertices
    degree = np.diff(indptr)
    ndata = np.add.reduceat(is_data[indices], indptr[:-1])
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(degree > 0, (2. * ndata - degree) / degree, np.nan)
    return r[:len(data)], r[len(data):]


def smoothed_density(positions, boxsize, cellsize=5., smoothing_radius=10., nthreads=1):
    """Periodic CIC mesh of the galaxies, Gaussian-smoothed, as 1 + delta."""
    nmesh = np.rint(boxsize / cellsize).astype(int)
    cell = boxsize / nmesh
    mesh = np.zeros(np.prod(nmesh), dtype='f8')
    x = positions / cell
    i0 = np.floor(x).astype(int)
    f = x - i0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1])
                     * np.where(dz, f[:, 2], 1 - f[:, 2]))
                idx = (i0 + [dx, dy, dz]) % nmesh
                flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), nmesh)
                mesh += np.bincount(flat, weights=w, minlength=mesh.size)
    mesh = mesh.reshape(nmesh) / mesh.mean()
    k = [2 * np.pi * fft.fftfreq(n, d=c) for n, c in zip(nmesh[:2], cell[:2])]
    k.append(2 * np.pi * fft.rfftfreq(nmesh[2], d=cell[2]))
    k2 = k[0][:, None, None]**2 + k[1][None, :, None]**2 + k[2][None, None, :]**2
    mesh_k = fft.rfftn(mesh, workers=nthreads)
    mesh_k *= np.exp(-0.5 * k2 * smoothing_radius**2)
    return fft.irfftn(mesh_k, s=mesh.shape, workers=nthreads), cell


def read_cic(mesh, cell, positions):
    nmesh = np.array(mesh.shape)
    x = positions / cell
    i0 = np.floor(x).astype(int)
    f = x - i0
    out = np.zeros(len(positions))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1])
                     * np.where(dz, f[:, 2], 1 - f[:, 2]))
                idx = (i0 + [dx, dy, dz]) % nmesh
                out += w * mesh[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out - 1.


def quantile_labels(values, rng, nquantiles=N_QUANTILES):
    """
    Equal-count quantile labels 0..nquantiles-1 (0 = lowest value).

    ASTRA r takes discrete values (ratios of small integers), so many points
    are tied; ties are broken at random so that every quantile has exactly
    the same number of points.
    """
    order = np.lexsort((rng.random(len(values)), values))
    labels = np.empty(len(values), dtype='i4')
    labels[order] = np.arange(len(values)) * nquantiles // len(values)
    return labels


# ASTRA cosmic-web classes (thresholds of the ASTRA reference implementation)
CLASS_EDGES = (-0.9, 0., 0.9)
CLASSES = ('void', 'sheet', 'filament', 'knot')


def astra_class_labels(r):
    """0 void r <= -0.9, 1 sheet (-0.9, 0], 2 filament (0, 0.9], 3 knot r > 0.9."""
    return np.digitize(r, CLASS_EDGES, right=True)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class SubboxClustering:
    """
    Landy-Szalay multipoles in an open cubic subbox [0, size), line of sight z.

    The geometry randoms and the R1R2 counts are shared by all subboxes and
    all statistics (the subboxes all have the same shape).
    """

    def __init__(self, size, sedges, nmu=120, ells=(0, 2, 4), nrand_geometry=None,
                 seed=1000, nthreads=1):
        self.size = np.asarray(size)
        self.edges = (np.asarray(sedges), np.linspace(-1., 1., nmu + 1))
        self.ells = ells
        self.nthreads = nthreads
        self.randoms = uniform_randoms(nrand_geometry, self.size, seed)
        self.R1R2 = self._count(self.randoms)

    def _count(self, positions1, positions2=None):
        return TwoPointCounter('smu', self.edges, positions1=positions1, positions2=positions2,
                               los='z', position_type='pos', nthreads=self.nthreads)

    def randoms_x(self, positions):
        """R x D counts for a sample, reusable as the R1D2 term of crosses with it."""
        return self._count(self.randoms, positions)

    def multipoles(self, positions1, positions2=None, R1D2=None, rebin=1):
        """
        Auto-correlation of positions1, or cross-correlation positions1 x positions2.
        For crosses, pass R1D2 = randoms_x(positions2) to avoid recomputing it.
        """
        D1D2 = self._count(positions1, positions2)
        D1R2 = self._count(positions1, self.randoms)
        if positions2 is None:
            R1D2 = D1R2
        elif R1D2 is None:
            R1D2 = self.randoms_x(positions2)
        est = LandySzalayTwoPointEstimator(D1D2=D1D2, R1R2=self.R1R2, D1R2=D1R2, R1D2=R1D2)
        if rebin > 1:
            est = est[::rebin, :]
        s, poles = est(ells=self.ells, return_sep=True)
        return s, poles


class PeriodicClustering:
    """Natural-estimator multipoles in a full periodic box (analytic RR), line of sight z."""

    def __init__(self, boxsize, sedges, nmu=120, ells=(0, 2, 4), nthreads=1):
        from pycorr import TwoPointCorrelationFunction
        self._tpcf = TwoPointCorrelationFunction
        self.boxsize = np.asarray(boxsize)
        self.edges = (np.asarray(sedges), np.linspace(-1., 1., nmu + 1))
        self.ells, self.nthreads = ells, nthreads

    def randoms_x(self, positions):
        return None

    def multipoles(self, positions1, positions2=None, R1D2=None):
        res = self._tpcf('smu', self.edges, data_positions1=positions1, data_positions2=positions2,
                         boxsize=self.boxsize, los='z', position_type='pos', nthreads=self.nthreads)
        return res(ells=self.ells, return_sep=True)


# ---------------------------------------------------------------------------
# Full measurement of one catalogue
# ---------------------------------------------------------------------------

def stat_names():
    names = ['gg']
    for method in METHODS:
        for pop in POPS:
            for kind in ('auto', 'cross'):
                names += [f'{method}_{pop}_{kind}_q{q + 1}' for q in range(N_QUANTILES)]
    return names


def measure_catalog(path, space='z', los='z', nside=4, pad=30., sedges=np.arange(0., 151., 5.),
                    nrand_astra=1, nrand_geometry=5, smoothing_radius=10., cellsize=5.,
                    seed=42, subboxes=None, nthreads=1, keep_sample=2000, thin=1):
    """
    Run ASTRA and density split on one HOD catalogue and measure all the
    clustering statistics in each subbox. Returns a dict of arrays.
    """
    galaxies, boxsize = load_catalog(path, space=space, los=los)
    if thin > 1:   # random subsampling of the galaxies (robustness tests)
        galaxies = galaxies[np.random.default_rng(seed + 3000).random(len(galaxies)) < 1. / thin]
    result, _ = measure_positions(galaxies, boxsize, path.name, nside=nside, pad=pad, sedges=sedges,
                                  nrand_astra=nrand_astra, nrand_geometry=nrand_geometry,
                                  smoothing_radius=smoothing_radius, cellsize=cellsize, seed=seed,
                                  subboxes=subboxes, nthreads=nthreads, keep_sample=keep_sample)
    result.update({f'hod_{k}': v for k, v in read_hod_header(path).items()})
    return result


def measure_positions(galaxies, boxsize, label='', nside=4, pad=30., sedges=np.arange(0., 151., 5.),
                      nrand_astra=1, nrand_geometry=5, smoothing_radius=10., cellsize=5., seed=42,
                      subboxes=None, nthreads=1, keep_sample=2000, periodic_clustering=False,
                      clustering=None):
    """
    ASTRA and density split on a periodic box of tracers (positions in [0, L),
    line of sight z), clustering measured in nside^3 subboxes. With nside=1
    the whole box is one subbox (periodic images used for the tessellation)
    and periodic_clustering=True uses the natural estimator of a periodic box
    instead of Landy-Szalay. `clustering` can be passed back in to reuse the
    random-random counts across boxes of the same geometry.
    Returns (dict of arrays, clustering).
    """
    t0 = time.time()
    galaxies = np.asarray(galaxies, dtype='f8') % boxsize
    ngal = len(galaxies)
    randoms = uniform_randoms(nrand_astra * ngal, boxsize, seed)
    logger.info(f'{label}: {ngal} tracers, box {np.round(boxsize, 2)}, {len(randoms)} randoms')

    # density split: one periodic smoothed field for the whole box
    mesh, cell = smoothed_density(galaxies, boxsize, cellsize, smoothing_radius, nthreads)
    delta_gal = read_cic(mesh, cell, galaxies)
    delta_rand = read_cic(mesh, cell, randoms)
    del mesh
    logger.info(f'density field done ({time.time() - t0:.0f}s)')

    corners, size = subbox_corners(boxsize, nside)
    ngal_box = ngal / nside**3
    if clustering is None:
        if periodic_clustering:
            clustering = PeriodicClustering(boxsize, sedges, nthreads=nthreads)
        else:
            clustering = SubboxClustering(size, sedges, nrand_geometry=int(nrand_geometry * ngal_box),
                                          seed=seed + 1000, nthreads=nthreads)
    logger.info(f'geometry RR done ({time.time() - t0:.0f}s)')

    if subboxes is None:
        subboxes = range(len(corners))
    names = stat_names()
    rng = np.random.default_rng(seed + 2000)
    out = {name: [] for name in names}
    diag = {k: [] for k in ['ngal', 'nrand', 'transition_data', 'transition_rand',
                            'spearman_data', 'spearman_rand', 'mean_env']}
    sample = {}

    for ibox in subboxes:
        tb = time.time()
        lo = corners[ibox]
        if nside == 1:
            igal, gal, gal_core = periodic_pad(galaxies, boxsize, pad)
            irand, rand, rand_core = periodic_pad(randoms, boxsize, pad)
        else:
            igal, gal, gal_core = extract(galaxies, lo, size, boxsize, pad)
            irand, rand, rand_core = extract(randoms, lo, size, boxsize, pad)

        # ASTRA on the padded region, kept on the core only
        r_gal, r_rand = astra_r(gal, rand)
        env = {('astra', 'data'): r_gal[gal_core], ('astra', 'rand'): r_rand[rand_core],
               ('ds', 'data'): delta_gal[igal][gal_core], ('ds', 'rand'): delta_rand[irand][rand_core]}
        pos = {'data': gal[gal_core], 'rand': rand[rand_core]}
        labels = {key: quantile_labels(value, rng) for key, value in env.items()}

        gal_all = pos['data']
        s, poles = clustering.multipoles(gal_all)
        out['gg'].append(poles)
        RxG = clustering.randoms_x(gal_all)
        for (method, pop), lab in labels.items():
            for q in range(N_QUANTILES):
                xq = pos[pop][lab == q]
                out[f'{method}_{pop}_auto_q{q + 1}'].append(clustering.multipoles(xq)[1])
                out[f'{method}_{pop}_cross_q{q + 1}'].append(
                    clustering.multipoles(xq, gal_all, R1D2=RxG)[1])

        # diagnostics: how similar are the two environment definitions?
        diag['ngal'].append(len(gal_all))
        diag['nrand'].append(len(pos['rand']))
        for pop in POPS:
            a, d = labels[('astra', pop)], labels[('ds', pop)]
            diag[f'transition_{pop}'].append(
                np.array([[np.mean(d[a == i] == j) for j in range(N_QUANTILES)]
                          for i in range(N_QUANTILES)]))
            ra = np.argsort(np.argsort(env[('astra', pop)]))
            rd = np.argsort(np.argsort(env[('ds', pop)]))
            diag[f'spearman_{pop}'].append(np.corrcoef(ra, rd)[0, 1])
        diag['mean_env'].append(np.array([[[env[(m, p)][labels[(m, p)] == q].mean()
                                            for q in range(N_QUANTILES)] for p in POPS]
                                          for m in METHODS]))
        if not sample and keep_sample:
            idx = rng.choice(len(gal_all), size=min(keep_sample, len(gal_all)), replace=False)
            sample = {'sample_pos': gal_all[idx], 'sample_r': env[('astra', 'data')][idx],
                      'sample_delta': env[('ds', 'data')][idx], 'sample_box': ibox}
        logger.info(f'subbox {ibox:2d}: {len(gal_all)} gal, {len(pos["rand"])} rand '
                    f'({time.time() - tb:.0f}s, total {time.time() - t0:.0f}s)')

    result = {'s': s, 'ells': np.array(clustering.ells), 'sedges': np.asarray(sedges),
              'subboxes': np.array(list(subboxes)), 'boxsize': boxsize, 'subbox_size': size,
              'stat_names': np.array(names)}
    result.update({name: np.array(value) for name, value in out.items()})
    result.update({name: np.array(value) for name, value in diag.items()})
    result.update(sample)
    return result, clustering


def class_stat_names():
    names = ['gg']
    for pop in POPS:
        for kind in ('auto', 'cross'):
            names += [f'astra_{pop}_{kind}_{c}' for c in CLASSES]
    return names


def measure_classes(galaxies, boxsize, label='', nside=4, pad=30., sedges=np.arange(0., 151., 5.),
                    nrand_astra=1, nrand_geometry=5, seed=42, subboxes=None, nthreads=1,
                    periodic_clustering=False, clustering=None, min_points=20):
    """
    Same as measure_positions, for the ASTRA cosmic-web classes (void, sheet,
    filament, knot) instead of quartiles: clustering of each class of
    galaxies and random points, and the fraction of points in each class.
    Classes with fewer than `min_points` points get NaN statistics.
    Returns (dict of arrays, clustering).
    """
    t0 = time.time()
    galaxies = np.asarray(galaxies, dtype='f8') % boxsize
    randoms = uniform_randoms(nrand_astra * len(galaxies), boxsize, seed)
    corners, size = subbox_corners(boxsize, nside)
    if clustering is None:
        if periodic_clustering:
            clustering = PeriodicClustering(boxsize, sedges, nthreads=nthreads)
        else:
            clustering = SubboxClustering(size, sedges, nrand_geometry=int(nrand_geometry * len(galaxies) / nside**3),
                                          seed=seed + 1000, nthreads=nthreads)
    if subboxes is None:
        subboxes = range(len(corners))
    names = class_stat_names()
    out = {name: [] for name in names}
    frac = {pop: [] for pop in POPS}
    for ibox in subboxes:
        if nside == 1:
            _, gal, gal_core = periodic_pad(galaxies, boxsize, pad)
            _, rand, rand_core = periodic_pad(randoms, boxsize, pad)
        else:
            _, gal, gal_core = extract(galaxies, corners[ibox], size, boxsize, pad)
            _, rand, rand_core = extract(randoms, corners[ibox], size, boxsize, pad)
        r_gal, r_rand = astra_r(gal, rand)
        pos = {'data': gal[gal_core], 'rand': rand[rand_core]}
        labels = {'data': astra_class_labels(r_gal[gal_core]), 'rand': astra_class_labels(r_rand[rand_core])}
        gal_all = pos['data']
        s, poles = clustering.multipoles(gal_all)
        out['gg'].append(poles)
        RxG = clustering.randoms_x(gal_all)
        for pop in POPS:
            frac[pop].append(np.bincount(labels[pop], minlength=len(CLASSES)) / len(labels[pop]))
            for ic, c in enumerate(CLASSES):
                xc = pos[pop][labels[pop] == ic]
                if len(xc) < min_points:
                    nan = np.full_like(poles, np.nan)
                    out[f'astra_{pop}_auto_{c}'].append(nan)
                    out[f'astra_{pop}_cross_{c}'].append(nan)
                    continue
                out[f'astra_{pop}_auto_{c}'].append(clustering.multipoles(xc)[1])
                out[f'astra_{pop}_cross_{c}'].append(clustering.multipoles(xc, gal_all, R1D2=RxG)[1])
        logger.info(f'{label} subbox {ibox}: fractions rand {np.round(frac["rand"][-1], 3)} '
                    f'data {np.round(frac["data"][-1], 3)} ({time.time() - t0:.0f}s)')
    result = {'s': s, 'ells': np.array(clustering.ells), 'sedges': np.asarray(sedges),
              'subboxes': np.array(list(subboxes)), 'stat_names': np.array(names), 'classes': np.array(CLASSES)}
    result.update({name: np.array(value) for name, value in out.items()})
    result.update({f'fraction_{pop}': np.array(v) for pop, v in frac.items()})
    return result, clustering
