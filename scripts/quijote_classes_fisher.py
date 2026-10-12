#!/usr/bin/env python
"""
Fisher forecasts on Quijote halos with the ASTRA cosmic-web classes
(void, sheet, filament, knot; scripts/run_quijote.py --classes), compared
with the ASTRA and density-split quartiles measured on the same realisations.

Same estimators as quijote_fisher.py (standard, compressed, combined;
reliable when compressed/standard < 1.5). Only the parameters whose
derivative sets were measured with classes are used (Om, ns, s8).

    python scripts/quijote_classes_fisher.py [--params Om s8] [--rebin 2]
Output: data/quijote_classes_fisher{_tag}.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import analysis as A
import quijote_fisher as QF


def c(pop, kinds=('cross',), classes=('sheet', 'filament')):
    return [f'astra_{pop}_{k}_{cl}' for k in kinds for cl in classes]


CLASS_VECTORS = {
    'classes rand: sheet, filament': c('rand'),
    'classes rand: void, sheet, filament': c('rand', classes=('void', 'sheet', 'filament')),
    'classes rand + auto: sheet, filament': c('rand', kinds=('cross', 'auto')),
    'classes gal: all four': c('data', classes=('void', 'sheet', 'filament', 'knot')),
    'classes gal + auto: sheet, filament': c('data', kinds=('cross', 'auto')),
}
QUARTILE_VECTORS = {
    'quartiles ASTRA rand Q1+Q4': QF.q('astra'),
    'quartiles ASTRA rand Q1-Q4': QF.q('astra', qs=(1, 2, 3, 4)),
    'quartiles ASTRA rand + auto': QF.q('astra', kinds=('cross', 'auto')),
    'quartiles ASTRA gal Q1+Q4': QF.q('astra', 'data'),
    'quartiles DS rand Q1+Q4': QF.q('ds'),
    'quartiles DS rand + auto': QF.q('ds', kinds=('cross', 'auto')),
    'quartiles DS gal Q1+Q4': QF.q('ds', 'data'),
    '2PCF': ['gg'],
}
SETS = ['fiducial', 'Om_p', 'Om_m', 'ns_p', 'ns_m', 's8_p', 's8_m']


def load(qdata, stats, ids=None):
    QF.QDATA = qdata
    sets, meta = {}, None
    for n in SETS:
        s, m = QF.load_set(n, stats)
        if ids is not None:
            s = {i: v for i, v in s.items() if i in ids[n]}
        sets[n], meta = s, meta or m
    sets.update({k: {} for k in ('fiducial_ZA', 'Mnu_p', 'Mnu_pp')})
    return sets, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', nargs='+', default=['Om', 's8'])
    parser.add_argument('--rebin', type=int, default=2)
    parser.add_argument('--smin', type=float, default=20.)
    parser.add_argument('--ells', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    QF.REBIN, QF.KEEP, QF.ELLS = args.rebin, list(args.params), tuple(args.ells)
    QF.STEPS = {p: QF.STEPS[p] for p in ('Om', 'ns', 's8')}

    cstats = sorted({s for v in CLASS_VECTORS.values() for s in v} | {'gg'})
    csets, cmeta = load(A.DATA / 'quijote_classes', cstats)
    # drop realisations where a class used is empty (NaN statistics)
    for n in SETS:
        csets[n] = {i: v for i, v in csets[n].items() if all(np.all(np.isfinite(v[s])) for s in cstats)}
    ids = {n: set(csets[n]) for n in SETS}
    qstats = sorted({s for v in QUARTILE_VECTORS.values() for s in v})
    qsets, qmeta = load(A.DATA / 'quijote', qstats, ids)
    print('realisations (classes / quartiles): ' + ', '.join(f'{n} {len(csets[n])}/{len(qsets[n])}' for n in SETS))

    res = {}
    for name, v in CLASS_VECTORS.items():
        res[name] = QF.compute(csets, cmeta, v, args.smin)
    for name, v in QUARTILE_VECTORS.items():
        res[name] = QF.compute(qsets, qmeta, v, args.smin)
    print(f'\nparams {args.params}: marginalised sigma (combined) [compressed/standard]  V = 1 (Gpc/h)^3')
    for name, r in res.items():
        rel = np.array(r['sigma_compressed']) / np.array(r['sigma_standard'])
        print(f'{name:40s} ndim {r["ndim"]:3d} ' +
              '  '.join(f'{p}: {s:.2e} [{x:.2f}]' for p, s, x in zip(args.params, r['sigma'], rel)))
    out = A.DATA / f'quijote_classes_fisher{args.tag}.json'
    out.write_text(json.dumps({'params': args.params, 'rebin': args.rebin, 'ells': args.ells,
                               'results': {k: {kk: vv for kk, vv in r.items() if kk != 'cov'} for k, r in res.items()}},
                              indent=1))
    print(f'saved {out}')


if __name__ == '__main__':
    main()
