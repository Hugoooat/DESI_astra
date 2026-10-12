#!/usr/bin/env python
"""
Train the emulator of every statistic (in parallel) and validate it on the
held-out test cosmologies.

    srun -N 1 -c 256 python scripts/train_emulator.py [--stats gg ...] [--nseeds 3]

Outputs
  models/{stat}.pt                      the trained ensembles
  data/emulator_validation.npz          test residuals per statistic
  plots/emulator_validation.png         |residual| / sigma vs s, ASTRA vs DS vs 2PCF
  plots/emulator_examples.png           prediction vs truth for test catalogues
"""

import sys
import json
import argparse
from pathlib import Path
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as A


def work(args):
    stat, nseeds, epochs, models, exclude, pair_file, lam, ssw = args
    import torch
    torch.set_num_threads(4)
    import emulator as E
    print(f'{stat}: start', flush=True)
    pairs = dict(np.load(pair_file)) if pair_file else None
    emu, history = E.train(stat, nseeds=nseeds, epochs=epochs, verbose=True, exclude=exclude,
                           pairs=pairs, lam=lam, small_scale_weight=ssw)
    models.mkdir(exist_ok=True)
    emu.save(models / f'{stat}.pt')
    res, ids = E.test_residuals(emu, stat)
    sigma = E.statistical_error(stat)
    print(f'{stat:28s} test |res|/sigma_box median {np.median(np.abs(res / sigma)):.2f}  '
          f'epochs {[h[0] for h in history]}', flush=True)
    return stat, res, sigma, ids, history


def plot_validation(out, s, tag=''):
    groups = [('2PCF', ['gg'], 'k'),
              ('ASTRA rand Q1,Q4 x gal', ['astra_rand_cross_q1', 'astra_rand_cross_q4'], '#d95f02'),
              ('DS rand Q1,Q4 x gal', ['ds_rand_cross_q1', 'ds_rand_cross_q4'], '#1b9e77'),
              ('ASTRA gal Q1,Q4 x gal', ['astra_data_cross_q1', 'astra_data_cross_q4'], '#e6ab02'),
              ('DS gal Q1,Q4 x gal', ['ds_data_cross_q1', 'ds_data_cross_q4'], '#66a61e')]
    ns = len(s)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for name, stats, color in groups:
        if not all(st in out for st in stats):
            continue
        r = np.concatenate([np.abs(out[st]['res'] / out[st]['sigma']) for st in stats])
        for il, ax in enumerate(axes):
            x = r[:, il * ns:(il + 1) * ns]
            ax.plot(s, np.median(x, 0), color=color, lw=1.6, label=name)
            ax.fill_between(s, np.percentile(x, 16, 0), np.percentile(x, 84, 0), color=color, alpha=0.12)
    for il, ax in enumerate(axes):
        ax.axhline(8, color='0.4', ls='--', lw=0.8)
        ax.text(s[0], 8.3, r'1$\sigma$ of a $(500\,h^{-1}{\rm Mpc})^3$ subbox', fontsize=8, color='0.4')
        ax.axhline(1, color='0.4', ls=':', lw=0.8)
        ax.set_yscale('log')
        ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
        ax.set_title(rf'$\ell={2 * il}$')
    axes[0].set_ylabel(r'|emulator $-$ truth| / $\sigma_{\rm box}$  (test cosmologies)')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Emulator accuracy on 8 held-out cosmologies (median and 68% range); '
                 r'$\sigma_{\rm box}$ = error of the full 2 $h^{-1}$Gpc box', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / f'emulator_validation{tag}.png', dpi=140)
    plt.close(fig)


def plot_examples(out, s, tag='', models_dir=None):
    import emulator as E
    stats = ['gg', 'astra_rand_cross_q1', 'astra_rand_cross_q4', 'ds_rand_cross_q1', 'ds_rand_cross_q4']
    stats = [st for st in stats if st in out]
    fig, axes = plt.subplots(2, len(stats), figsize=(3.4 * len(stats), 6), sharex=True)
    ns = len(s)
    for col, st in enumerate(stats):
        X, Y, ids, _, _ = E.load_dataset(st)
        te = E.split(ids)[2]
        emu = E.Emulator.load(st, models_dir or E.MODELS)
        import torch
        with torch.no_grad():
            pred = emu(X[te][:4]).numpy()
        for k in range(4):
            for il in range(2):
                ax = axes[il, col]
                sl = slice(il * ns, (il + 1) * ns)
                ax.plot(s, s**2 * Y[te][k, sl], 'o', ms=3, color=f'C{k}')
                ax.plot(s, s**2 * pred[k, sl], '-', color=f'C{k}', lw=1)
        axes[0, col].set_title(st.replace('_', ' '), fontsize=9)
        axes[1, col].set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    axes[0, 0].set_ylabel(r'$s^2\xi_0$')
    axes[1, 0].set_ylabel(r'$s^2\xi_2$')
    fig.suptitle('Emulator (lines) vs measurements (points) for 4 catalogues of held-out cosmologies', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / f'emulator_examples{tag}.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', nargs='+', default=None)
    parser.add_argument('--nseeds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=6000)
    parser.add_argument('--nproc', type=int, default=33)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--exclude-cosmos', type=int, nargs='*', default=[],
                        help='cosmologies left out of training entirely (e.g. the derivative grid)')
    parser.add_argument('--tag', default='', help='suffix of the validation outputs')
    parser.add_argument('--pairs', type=Path, default=None, help='data/pair_dataset.npz: train on pair differences too')
    parser.add_argument('--lam', type=float, default=1., help='weight of the pair-difference loss')
    parser.add_argument('--small-scale-weight', type=float, default=0., help='alpha of the (25/s)^alpha bin weight')
    args = parser.parse_args()
    import emulator as E
    stats = args.stats or [str(s) for s in np.load(A.DATA / 'emulator_dataset.npz')['stat_names']]
    # 'spawn': forking a process in which torch/OpenMP threads already exist deadlocks the children
    with mp.get_context('spawn').Pool(min(args.nproc, len(stats))) as pool:
        results = pool.map(work, [(st, args.nseeds, args.epochs, args.models_dir, args.exclude_cosmos,
                                   args.pairs, args.lam, args.small_scale_weight) for st in stats])
    out = {st: {'res': res, 'sigma': sig, 'ids': ids, 'history': hist} for st, res, sig, ids, hist in results}
    np.savez(A.DATA / f'emulator_validation{args.tag}.npz',
             **{f'{st}/{k}': np.asarray(v) for st, d in out.items() for k, v in d.items()})
    s = E.load_dataset(stats[0])[3]
    (ROOT / 'plots').mkdir(exist_ok=True)
    plot_validation(out, s, args.tag)
    plot_examples(out, s, args.tag, args.models_dir)
    summary = {st: {'median_abs_res_over_sigma_box': float(np.median(np.abs(d['res'] / d['sigma']))),
                    'p68': float(np.percentile(np.abs(d['res'] / d['sigma']), 68))} for st, d in out.items()}
    (A.DATA / f'emulator_validation{args.tag}.json').write_text(json.dumps(summary, indent=1))
    print(f'saved {args.models_dir}, data/emulator_validation{args.tag}.*, plots/emulator_*{args.tag}.png')


if __name__ == '__main__':
    main()
