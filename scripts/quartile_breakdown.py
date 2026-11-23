#!/usr/bin/env python
"""
Marginalised errors (baseline: official covariance, official fiducial, 6 cosmological parameters,
HOD marginalised, compressed) of each quartile alone and of combinations, for ASTRA and density split.
    python scripts/quartile_breakdown.py  ->  data/quartile_breakdown.json
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import compare_methods
P = ['omega_b', 'omega_cdm', 'n_s', 'sigma8', 'w0', 'wa']
compare_methods.PARAMS = P
fc = compare_methods.Forecaster(ROOT / 'models_pairs', '_pairs', compress=True, cov='covbox', fiducial='official')
def q(m, kinds=('cross',), qs=(1, 4), pop='rand'):
    return [f'{m}_{pop}_{k}_q{i}' for k in kinds for i in qs]
vecs = {'2PCF': ['gg']}
for m in ('astra', 'ds'):
    for i in (1, 2, 3, 4):
        vecs[f'{m} Q{i} cross'] = q(m, qs=(i,))
    vecs[f'{m} Q1+Q4 cross'] = q(m)
    vecs[f'{m} Q1-Q4 cross'] = q(m, qs=(1, 2, 3, 4))
    vecs[f'{m} Q1+Q4 cross+auto'] = q(m, kinds=('cross', 'auto'))
    vecs[f'{m} Q1 cross+auto'] = q(m, kinds=('cross', 'auto'), qs=(1,))
    vecs[f'{m} Q4 cross+auto'] = q(m, kinds=('cross', 'auto'), qs=(4,))
out = {}
for k, v in vecs.items():
    r = fc(v)
    out[k] = {'sigma': r['marg'][0].tolist(), 'err': r['marg'][1].std(0).tolist() if len(r['marg'][1]) else None}
    print(f'{k:24s} ' + ' '.join(f'{x:.3g}' for x in out[k]['sigma']), flush=True)
(ROOT / 'data' / 'quartile_breakdown.json').write_text(json.dumps({'params': P, 'results': out}, indent=1))
