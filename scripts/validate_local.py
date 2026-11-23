#!/usr/bin/env python
"""
Test of the emulator on the quantity the Fisher derivatives rely on: the
differences between catalogues of the derivative grid near the fiducial HOD.

The catalogues listed in data/pair_dataset.npz['holdout'] (4 local HOD
draws per grid cosmology) are excluded from the training of the emulators
compared here. For every grid pair (c+, c-) we form all held-out pairs
(h_i in c+, h_j in c-) and compare the measured difference (paired subboxes,
precise) with the emulator prediction.

Metrics per statistic and parameter:
  rel  = || emu - sim || / || sim ||          (whole difference vector)
  chi  = rms((emu - sim) / sigma_box)         (in units of the error of one box)

    python scripts/validate_local.py models_val_base models_val_pairs models_val_pairs_ssw
Outputs: data/validate_local.json, plots/validate_local.png
"""

import sys
import json
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A
import emulator as E
from build_pair_dataset import GRID

STATS = ['gg', 'astra_rand_cross_q1', 'astra_rand_cross_q4', 'ds_rand_cross_q1', 'ds_rand_cross_q4',
         'astra_data_cross_q4', 'ds_data_cross_q4']
NAMES = {(100, 101): 'omega_b', (102, 103): 'omega_cdm', (104, 105): 'n_s', (106, 107): 'alpha_s',
         (108, 109): 'w0', (110, 111): 'w0-wa', (112, 113): 'sigma8'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', type=Path)
    args = parser.parse_args()
    pairs = np.load(A.DATA / 'pair_dataset.npz')
    d = np.load(A.DATA / 'emulator_dataset.npz')
    held = [tuple(map(int, x)) for x in pairs['holdout_ids']]
    sub = {k: A.load(*k, los=['z']) for k in held}
    res = {}
    for st in STATS:
        X, Y, ids, s, _ = E.load_dataset(st)
        row = {(int(c), int(h)): i for i, (c, h) in enumerate(ids)}
        sig = E.statistical_error(st)
        emus = {m.name: E.Emulator.load(st, m) for m in args.models}
        for cp, cm in GRID:
            hp = [k for k in held if k[0] == cp]
            hm = [k for k in held if k[0] == cm]
            for a, b in product(hp, hm):
                ma, sa, ia = sub[a]
                mb, sb, ib = sub[b]
                sim = A.paired_derivative(A.data_vector(ma, sa, [st])[1], A.data_vector(mb, sb, [st])[1],
                                          ia, ib, 1.).mean(0)
                for name, emu in emus.items():
                    with torch.no_grad():
                        pred = (emu(X[row[a]][None]) - emu(X[row[b]][None]))[0].numpy()
                    r = res.setdefault(name, {}).setdefault(NAMES[(cp, cm)], {}).setdefault(st, {'rel': [], 'chi': []})
                    r['rel'].append(float(np.linalg.norm(pred - sim) / np.linalg.norm(sim)))
                    r['chi'].append(float(np.sqrt(np.mean(((pred - sim) / sig)**2))))
    summary = {m: {p: {st: {k: float(np.median(v)) for k, v in dd.items()} for st, dd in dp.items()}
                   for p, dp in dm.items()} for m, dm in res.items()}
    (A.DATA / 'validate_local.json').write_text(json.dumps(summary, indent=1))

    print('median relative error of the predicted pair differences (held-out local HODs)')
    params = list(NAMES.values())
    print(f'{"model":24s} ' + ' '.join(f'{p:>10s}' for p in params))
    for m in summary:
        print(f'{m:24s} ' + ' '.join(f'{np.median([summary[m][p][st]["rel"] for st in STATS]):10.2f}' for p in params))
    print('\nsame, rms error in units of the error of one box:')
    for m in summary:
        print(f'{m:24s} ' + ' '.join(f'{np.median([summary[m][p][st]["chi"] for st in STATS]):10.2f}' for p in params))

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(summary)
    for k, m in enumerate(summary):
        ax.bar(np.arange(len(params)) + k * width, [np.median([summary[m][p][st]['rel'] for st in STATS])
                                                    for p in params], width, label=m)
    ax.set_xticks(np.arange(len(params)) + 0.4 - width / 2, params)
    ax.set_ylabel('median |emu - sim| / |sim|')
    ax.set_title('Emulator prediction of held-out pair differences of the derivative grid', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / 'validate_local.png', dpi=140)
    print('saved data/validate_local.json, plots/validate_local.png')


if __name__ == '__main__':
    main()
