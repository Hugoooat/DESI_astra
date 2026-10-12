#!/usr/bin/env python
"""
Summary of the Quijote Fisher forecasts for several parameter subsets and
compressions (outputs of quijote_fisher.py --tag _p{i}c{j}).

Reliability: the standard Fisher estimate is biased high by the noise of the
derivatives and the compressed one biased low (Coulton & Wandelt 2023); when
their marginalised errors agree (ratio close to 1) the forecast does not depend
on the derivative noise. We call a forecast reliable when
sigma_compressed / sigma_standard < 1.5 for the two vectors compared.

    python scripts/summarize_quijote.py   ->  data/quijote_summary.json, plots/quijote_subsets.png
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PSETS = [['Om', 's8'], ['Om', 's8', 'Mnu'], ['Om', 'ns', 's8'], ['Om', 'ns', 's8', 'Mnu'],
         ['Om', 'Ob', 'h', 'ns', 's8'], ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']]
COMPS = ['10 Mpc/h bins, l=0,2', '20 Mpc/h bins, l=0,2', '20 Mpc/h bins, l=0']
PAIRS = [('ASTRA rand', 'DS rand'), ('ASTRA rand + auto', 'DS rand + auto'), ('ASTRA gal', 'DS gal')]
LABELS = {'Om': r'$\Omega_m$', 'Ob': r'$\Omega_b$', 'h': '$h$', 'ns': '$n_s$', 's8': r'$\sigma_8$', 'Mnu': r'$M_\nu$'}
THRESH = 1.5


def main():
    rows = []
    for i, ps in enumerate(PSETS):
        for j, comp in enumerate(COMPS):
            f = ROOT / 'data' / f'quijote_fisher_p{i}c{j}.json'
            if not f.exists():
                continue
            res = json.loads(f.read_text())['results']
            for a, d in PAIRS:
                ra, rd = res[a], res[d]
                for k, p in enumerate(ps):
                    sa, sd = ra['sigma'][k], rd['sigma'][k]
                    rel_a = ra['sigma_compressed'][k] / ra['sigma_standard'][k]
                    rel_d = rd['sigma_compressed'][k] / rd['sigma_standard'][k]
                    rows.append(dict(params=' '.join(ps), npar=len(ps), comp=comp, vector=a.replace('ASTRA ', ''),
                                     param=p, ratio=sd / sa, sigma_astra=sa, sigma_ds=sd,
                                     spread_astra=rel_a, spread_ds=rel_d,
                                     reliable=bool(rel_a < THRESH and rel_d < THRESH)))
    (ROOT / 'data' / 'quijote_summary.json').write_text(json.dumps(rows, indent=1))

    # table: random quartiles, Q1+Q4 crosses + autos, for each parameter subset and compression
    for vec in ['rand', 'rand + auto', 'gal']:
        print(f'\n=== {vec}: sigma_DS / sigma_ASTRA  (* = reliable: compressed/standard < {THRESH} for both)')
        for ps in PSETS:
            for comp in COMPS:
                rs = [r for r in rows if r['params'] == ' '.join(ps) and r['comp'] == comp and r['vector'] == vec]
                if rs:
                    print(f'  {" ".join(ps):22s} {comp:22s} ' +
                          '  '.join(f'{r["param"]}:{r["ratio"]:5.2f}{"*" if r["reliable"] else " "}' for r in rs))

    # figure: ratio per parameter, all configurations, reliable ones filled
    params = ['Om', 'Ob', 'h', 'ns', 's8', 'Mnu']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, vec in zip(axes, ['rand', 'rand + auto', 'gal']):
        for k, p in enumerate(params):
            rs = [r for r in rows if r['vector'] == vec and r['param'] == p]
            for r in rs:
                x = k + 0.08 * (r['npar'] - 4) + 0.02 * COMPS.index(r['comp'])
                ax.scatter(x, r['ratio'], s=30, marker='o',
                           facecolor='#d95f02' if r['reliable'] else 'none', edgecolor='#d95f02')
        ax.axhline(1, color='k', lw=0.8)
        ax.set_yscale('log')
        ax.set_xticks(range(len(params)), [LABELS[p] for p in params])
        ax.set_title(f'{vec} quartiles (Q1+Q4)', fontsize=10)
    axes[0].set_ylabel(r'$\sigma_{\rm DS}/\sigma_{\rm ASTRA}$  (> 1: ASTRA tighter)')
    fig.suptitle('Quijote halos: ASTRA vs density split for 6 parameter subsets x 3 compressions; '
                 f'filled: reliable (standard and compressed Fisher agree within {THRESH})', fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / 'plots' / 'quijote_subsets.png', dpi=140)
    print('\nsaved data/quijote_summary.json, plots/quijote_subsets.png')


if __name__ == '__main__':
    main()
