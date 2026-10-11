#!/usr/bin/env python
"""
Assemble the emulator training set from the measured catalogues.

For every data/c{cosmo}_hod{hod}_z_losz.npz (emulator runs, Fisher runs and
HOD-calibration runs alike) we store
    X : cosmological parameters (omega_b, omega_cdm, h, A_s, n_s, alpha_s,
        N_ur, w0, wa, sigma8) + the 12 HOD parameters (physical units)
    Y : for each statistic, the multipoles averaged over the 64 subboxes
        (= the measurement in the full 2 Gpc/h box), shape (nell, ns)
    Y_err : the error on that mean (subbox scatter / sqrt(64))

    python scripts/build_emulator_dataset.py   ->  data/emulator_dataset.npz
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import astra_ds
import analysis as A

COSMO_COLUMNS = ['omega_b', 'omega_cdm', 'h', 'A_s', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld', 'sigma8_m']


def main():
    table = A.cosmology_table()
    files = sorted(A.DATA.glob('c*_hod*_z_losz.npz'))
    X, Y, E, ids = [], {}, {}, []
    for fn in files:
        d = np.load(fn)
        cosmo, hod = int(d['cosmo']), int(d['hod'])
        if cosmo not in table.index:
            print(f'skip {fn.name}: cosmology {cosmo} not in the table')
            continue
        x = [float(table.loc[cosmo, c]) for c in COSMO_COLUMNS] + [float(d[f'hod_{k}']) for k in astra_ds.HOD_PARAMS]
        X.append(x)
        ids.append((cosmo, hod))
        for name in d['stat_names']:
            Y.setdefault(name, []).append(d[name].mean(0))
            E.setdefault(name, []).append(d[name].std(0, ddof=1) / np.sqrt(len(d[name])))
        meta = {k: d[k] for k in ('s', 'ells', 'stat_names')}
    out = dict(X=np.array(X), x_names=np.array(COSMO_COLUMNS + astra_ds.HOD_PARAMS), ids=np.array(ids), **meta)
    out.update({f'Y/{k}': np.array(v) for k, v in Y.items()})
    out.update({f'Yerr/{k}': np.array(v) for k, v in E.items()})
    np.savez(A.DATA / 'emulator_dataset.npz', **out)
    print(f'{len(X)} catalogues from {len(set(c for c, _ in ids))} cosmologies -> {A.DATA / "emulator_dataset.npz"}')


if __name__ == '__main__':
    main()
