#!/usr/bin/env python
"""
Choose one HOD catalogue per cosmology for the Fisher analysis.

The emulator catalogues are not HOD-matched across cosmologies: hodNNN has
different HOD parameters in each cosmology. Averaging over many hodNNN files
therefore mixes the HOD scatter (~50% on xi_0) into the cosmological
derivatives (~1% effect). Instead we pick, for each +/- pair of the
AbacusSummit linear-derivative grid, the two catalogues with the closest HOD
parameters, both close to a common anchor: the c000 catalogue nearest to the
centre of the prior, which is also the fiducial for the covariance.

Distances are Euclidean in HOD parameters rescaled to [0, 1] by the prior
range spanned by the catalogues.

    python scripts/select_hods.py   ->  config/hod_selection.json
"""

import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import astra_ds

FIDUCIAL = 0
# parameter: (cosmo +, cosmo -, half step in the parameter, label)
PAIRS = {
    'omega_b': (100, 101),
    'omega_cdm': (102, 103),
    'n_s': (104, 105),
    'sigma8': (112, 113),
}


def read_all(cosmo):
    files = sorted((astra_ds.HOD_DIR / f'c{cosmo:03d}_ph000' / 'seed0').glob('hod*.fits'))
    hods = np.array([int(f.stem[3:]) for f in files])
    params = np.array([[astra_ds.read_hod_header(f)[k] for k in astra_ds.HOD_PARAMS] for f in files])
    return hods, params


def main():
    cosmos = [FIDUCIAL] + [c for pair in PAIRS.values() for c in pair]
    table = {c: read_all(c) for c in cosmos}
    allp = np.concatenate([p for _, p in table.values()])
    lo, hi = allp.min(0), allp.max(0)
    norm = {c: (h, (p - lo) / (hi - lo)) for c, (h, p) in table.items()}

    hods0, p0 = norm[FIDUCIAL]
    ifid = np.argmin(np.linalg.norm(p0 - 0.5, axis=1))
    anchor = p0[ifid]
    selection = {'fiducial': {'cosmo': FIDUCIAL, 'hod': int(hods0[ifid]),
                              'dist_to_centre': float(np.linalg.norm(anchor - 0.5))},
                 'pairs': {}}
    print(f'fiducial c000 hod{hods0[ifid]:03d}  |p - centre| = {np.linalg.norm(anchor - 0.5):.3f}')

    for name, (cp, cm) in PAIRS.items():
        hp, pp = norm[cp]
        hm, pm = norm[cm]
        dpm = np.linalg.norm(pp[:, None] - pm[None], axis=-1)
        cost = dpm + np.linalg.norm(pp - anchor, axis=1)[:, None] + np.linalg.norm(pm - anchor, axis=1)[None]
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        random_d = np.median(dpm)
        selection['pairs'][name] = {
            'plus': {'cosmo': cp, 'hod': int(hp[i])}, 'minus': {'cosmo': cm, 'hod': int(hm[j])},
            'dist_pm': float(dpm[i, j]), 'dist_pm_random_median': float(random_d),
            'dist_plus_anchor': float(np.linalg.norm(pp[i] - anchor)),
            'dist_minus_anchor': float(np.linalg.norm(pm[j] - anchor)),
            'max_param_diff_pm': float(np.abs(pp[i] - pm[j]).max()),
        }
        print(f'{name:10s} c{cp}_hod{hp[i]:03d} / c{cm}_hod{hm[j]:03d}   '
              f'|d+-|={dpm[i, j]:.3f} (random pairs: {random_d:.3f})   '
              f'max single-param diff={np.abs(pp[i] - pm[j]).max():.3f}')

    selection['hod_params'] = astra_ds.HOD_PARAMS
    selection['prior_lo'] = lo.tolist()
    selection['prior_hi'] = hi.tolist()
    out = ROOT / 'config' / 'hod_selection.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(selection, indent=2))
    print(f'saved {out}')


if __name__ == '__main__':
    main()
