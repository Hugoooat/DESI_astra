#!/usr/bin/env python
"""
Maps of the environments in a slice of the fiducial catalogue, drawn as
filled regions: each pixel of a 10 Mpc/h slab takes the environment of the
nearest random point (ASTRA r, quartile, class) or the smoothed density
contrast (density split), with the galaxies of the slab overlaid.

    python scripts/env_maps.py  ->  plots/env_maps.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import astra_ds as D

SIZE, SLAB, PIX = 250., 10., 1.          # region side, slab thickness, pixel [Mpc/h]
QCMAP = ListedColormap(['#3b4cc0', '#9ebbf2', '#f4b39b', '#b2182b'])
CCMAP = ListedColormap(['#3b4cc0', '#9ebbf2', '#f4b39b', '#b2182b'])


def main(nthreads=128, seed=42):
    gal, box = D.load_catalog(D.hod_path(0, 239))
    ran = D.uniform_randoms(len(gal), box, seed)
    mesh, cell = D.smoothed_density(gal, box, 5., 10., nthreads)
    corners, size = D.subbox_corners(box, 4)
    _, g, gc = D.extract(gal, corners[0], size, box, 30.)
    _, r, rc = D.extract(ran, corners[0], size, box, 30.)
    rg, rr = D.astra_r(g, r)
    g, r, rr = g[gc], r[rc], rr[rc]
    rng = np.random.default_rng(seed + 2000)
    qa = D.quantile_labels(rr, rng)
    dr = D.read_cic(mesh, cell, r + corners[0])
    qd_edges = np.quantile(dr, [0.25, 0.5, 0.75])
    z0 = size[2] / 2
    # pixel grid in the middle plane of the slab
    xs = np.arange(PIX / 2, SIZE, PIX)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z0)])
    near = cKDTree(r).query(pts)[1]
    img = {'r': rr[near], 'qa': qa[near], 'cls': D.astra_class_labels(rr[near])}
    delta = D.read_cic(mesh, cell, pts + corners[0])
    img['delta'] = delta
    img['qd'] = np.digitize(delta, qd_edges)
    sl = (np.abs(g[:, 2] - z0) < SLAB / 2) & (g[:, 0] < SIZE) & (g[:, 1] < SIZE)
    shape = X.shape
    ext = (0, SIZE, 0, SIZE)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11), sharex=True, sharey=True)
    panels = [
        (axes[0, 0], img['r'], dict(cmap='RdBu_r', vmin=-1, vmax=1), 'ASTRA $r$ (nearest random point)', '$r$'),
        (axes[0, 1], img['delta'], dict(cmap='RdBu_r', vmin=-1, vmax=1), r'Density split: $\delta_{R=10}$', r'$\delta_R$'),
        (axes[1, 0], img['qa'], dict(cmap=QCMAP, norm=BoundaryNorm(np.arange(-0.5, 4), 4)), 'ASTRA quartiles', None),
        (axes[1, 1], img['qd'], dict(cmap=QCMAP, norm=BoundaryNorm(np.arange(-0.5, 4), 4)), 'Density-split quartiles', None),
        (axes[1, 2], img['cls'], dict(cmap=CCMAP, norm=BoundaryNorm(np.arange(-0.5, 4), 4)), 'ASTRA classes', None),
    ]
    for ax, im, kw, title, clab in panels:
        h = ax.imshow(im.reshape(shape), origin='lower', extent=ext, interpolation='nearest', **kw)
        ax.scatter(g[sl, 0], g[sl, 1], s=1.5, c='k', lw=0)
        ax.set_title(title, fontsize=11)
        if clab:
            fig.colorbar(h, ax=ax, fraction=0.046, label=clab)
        elif 'classes' in title:
            cb = fig.colorbar(h, ax=ax, fraction=0.046, ticks=range(4))
            fracs = [np.mean(img['cls'] == i) for i in range(4)]
            cb.ax.set_yticklabels([f'{c} ({f:.1%})' for c, f in zip(D.CLASSES, fracs)], fontsize=8)
        else:
            cb = fig.colorbar(h, ax=ax, fraction=0.046, ticks=range(4))
            cb.ax.set_yticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax = axes[0, 2]
    ax.scatter(g[sl, 0], g[sl, 1], s=2, c='k', lw=0)
    ax.set_title(f'Galaxies ({sl.sum()} in the slab)', fontsize=11)
    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, SIZE)
    ax.set_aspect('equal')
    for a in axes[1]:
        a.set_xlabel(r'$x\ [h^{-1}{\rm Mpc}]$')
    for a in axes[:, 0]:
        a.set_ylabel(r'$y\ [h^{-1}{\rm Mpc}]$')
    fig.suptitle(rf'Environments in a ${SLAB:g}\,h^{{-1}}$Mpc slab, ${SIZE:g}\times{SIZE:g}\,h^{{-1}}$Mpc '
                 '(fiducial catalogue, redshift space); black: galaxies of the slab', fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / 'env_maps.png', dpi=130)
    print('saved plots/env_maps.png')


if __name__ == '__main__':
    main()
