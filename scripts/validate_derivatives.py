#!/usr/bin/env python
"""
Out-of-sample test of the cosmological response of the emulator.

The emulator in --models-dir (default models_loco/) was trained WITHOUT the
cosmologies of the AbacusSummit derivative grid (c100-c113). For each +/- pair
we compare the difference measured in the simulations (paired subboxes, with
its error) to the difference predicted by that emulator for the same two
(cosmology, HOD) points. Agreement means the emulator captures the response
to a ~1-3% change of one cosmological parameter in cosmologies it has never
seen, which is what the Fisher derivatives rely on.

    python scripts/validate_derivatives.py [--models-dir models_loco]
Outputs: plots/derivatives_loco.png, data/derivatives_loco.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A
import emulator as E
from fisher_emulator import input_point, LABELS

STATS = ['gg', 'astra_rand_cross_q1', 'astra_rand_cross_q4', 'ds_rand_cross_q1', 'ds_rand_cross_q4',
         'astra_data_cross_q1', 'astra_data_cross_q4', 'ds_data_cross_q1', 'ds_data_cross_q4']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models_loco')
    args = parser.parse_args()
    sel, table = A.selection(), A.cosmology_table()
    emus = {st: E.Emulator.load(st, args.models_dir) for st in STATS}
    s = emus['gg'].s
    ns = len(s)
    out = {}
    plot_stats = STATS[:5]
    fig, axes = plt.subplots(len(sel['pairs']), len(plot_stats), figsize=(3.3 * len(plot_stats), 2.6 * len(sel['pairs'])),
                             sharex=True)
    for i, (param, pair) in enumerate(sel['pairs'].items()):
        pp, pm = pair['plus'], pair['minus']
        step = table.loc[pp['cosmo'], A.PARAM_COLUMNS[param]] - table.loc[pm['cosmo'], A.PARAM_COLUMNS[param]]
        xp, xm = input_point(pp['cosmo'], pp['hod']), input_point(pm['cosmo'], pm['hod'])
        mp, sp, ip = A.load(pp['cosmo'], pp['hod'])
        mm, sm, im = A.load(pm['cosmo'], pm['hod'])
        out[param] = {}
        for j, st in enumerate(STATS):
            d = A.paired_derivative(A.data_vector(mp, sp, [st])[1], A.data_vector(mm, sm, [st])[1], ip, im, step)
            sim, err = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
            with torch.no_grad():
                emu = (emus[st](xp[None])[0].numpy() - emus[st](xm[None])[0].numpy()) / step
            # error budget: simulation error of the pair difference + emulator error of the two points
            emu_err = np.sqrt(2) * np.load(A.DATA / 'emulator_validation_loco.npz')[f'{st}/res'].std(0) / abs(step)
            chi = (emu - sim) / np.sqrt(err**2 + emu_err**2)
            out[param][st] = {'chi2_per_bin': float(np.mean(chi**2)),
                              'rel_diff': float(np.linalg.norm(emu - sim) / np.linalg.norm(sim))}
            if st in plot_stats:
                ax = axes[i, plot_stats.index(st)]
                ax.fill_between(s, s**2 * (sim[:ns] - err[:ns]), s**2 * (sim[:ns] + err[:ns]), color='0.7',
                                label='simulations')
                ax.plot(s, s**2 * emu[:ns], color='#e31a1c', lw=1.5, label='emulator (never saw c100-c113)')
                ax.axhline(0, color='0.6', lw=0.5)
                if i == 0:
                    ax.set_title(st.replace('_', ' '), fontsize=9)
                if plot_stats.index(st) == 0:
                    ax.set_ylabel(rf'$s^2\Delta\xi_0/\Delta {LABELS[param]}$', fontsize=9)
                ax.text(0.97, 0.05, f'rel. diff {out[param][st]["rel_diff"]:.0%}', transform=ax.transAxes,
                        ha='right', fontsize=7)
                ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('Out-of-sample test: +/- pair differences (monopole), simulations vs an emulator trained without '
                 'the derivative-grid cosmologies', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / 'derivatives_loco.png', dpi=140)
    (A.DATA / 'derivatives_loco.json').write_text(json.dumps(out, indent=1))
    print('relative difference |emu - sim| / |sim| (monopole + quadrupole):')
    for param, d in out.items():
        print(f'  {param:10s} ' + '  '.join(f'{st.replace("_cross", "")}: {v["rel_diff"]:.2f}' for st, v in d.items()))
    print('saved plots/derivatives_loco.png, data/derivatives_loco.json')


if __name__ == '__main__':
    main()
