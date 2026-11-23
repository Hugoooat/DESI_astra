#!/usr/bin/env python
"""
Real space (r-split) vs redshift space (z-split): the environments are
defined, and the clustering measured, either from real-space positions or
from redshift-space positions (line of sight z), for the fiducial catalogue
(3 lines of sight x 64 subboxes).

Quantifies how redshift-space distortions change the quartile statistics of
ASTRA split and density split: monopole amplitude and the quadrupole that is
created by RSD.

    python scripts/plot_real_space.py   ->  plots/real_vs_redshift.png, data/real_vs_redshift.json
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A

STATS = ['astra_rand_cross_q1', 'ds_rand_cross_q1', 'astra_rand_cross_q4', 'ds_rand_cross_q4']
COL = {'astra': '#d95f02', 'ds': '#1b9e77'}


def main():
    sel = A.selection()['fiducial']
    mz, sz, _ = A.load(sel['cosmo'], sel['hod'], space='z')
    mr, sr, _ = A.load(sel['cosmo'], sel['hod'], space='r')
    s = mz['s']
    ells = list(mz['ells'])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    out = {}
    for col, (title, sts) in enumerate([('galaxy 2PCF', ['gg']),
                                        ('Q1 (voids) x galaxies', ['astra_rand_cross_q1', 'ds_rand_cross_q1']),
                                        ('Q4 (clusters) x galaxies', ['astra_rand_cross_q4', 'ds_rand_cross_q4'])]):
        for st in sts:
            color = 'k' if st == 'gg' else COL[st.split('_')[0]]
            lab = 'galaxies' if st == 'gg' else st.split('_')[0].upper()
            for row, ell in enumerate((0, 2)):
                ax = axes[row, col]
                il = ells.index(ell)
                for space, d, ls in (('z', sz, '-'), ('r', sr, '--')):
                    x = d[st][:, il]
                    m, e = x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x))
                    ax.plot(s, s**2 * m, ls=ls, color=color, lw=1.5, label=f'{lab}, {space}-split' if row == 0 else None)
                    ax.fill_between(s, s**2 * (m - e), s**2 * (m + e), color=color, alpha=0.15)
                ax.axhline(0, color='0.6', lw=0.5)
                if row == 0:
                    ax.set_title(title, fontsize=10)
                else:
                    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
                if col == 0:
                    ax.set_ylabel(rf'$s^2\xi_{ell}$')
            # amplitude change of the monopole at 20-60 Mpc/h, and quadrupole created by RSD
            keep = (s > 20) & (s < 60)
            mono = (sz[st][:, ells.index(0)].mean(0)[keep] / sr[st][:, ells.index(0)].mean(0)[keep]).mean()
            quad = np.abs(sz[st][:, ells.index(2)].mean(0)[keep]).mean() / np.abs(sz[st][:, ells.index(0)].mean(0)[keep]).mean()
            out[st] = {'monopole_ratio_z_over_r_20_60': float(mono), 'quadrupole_over_monopole_z_20_60': float(quad)}
        axes[0, col].legend(fontsize=7, frameon=False)
    fig.suptitle('Environments and clustering defined in redshift space (solid) or real space (dashed), fiducial '
                 'catalogue; bands: error of the mean over 192 subboxes', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / 'real_vs_redshift.png', dpi=140)
    (A.DATA / 'real_vs_redshift.json').write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f'{k:22s} monopole z/r (20-60 Mpc/h) {v["monopole_ratio_z_over_r_20_60"]:.2f}   '
              f'|xi2|/|xi0| in z-space {v["quadrupole_over_monopole_z_20_60"]:.2f}')
    print('saved plots/real_vs_redshift.png')


if __name__ == '__main__':
    main()
