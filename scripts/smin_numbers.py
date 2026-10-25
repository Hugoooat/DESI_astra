#!/usr/bin/env python
"""log10 sigma vs s_min (baseline, 6 parameters) for 2PCF, ASTRA, DS, ASTRA+DS -> data/smin_numbers.json"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import compare_methods
compare_methods.PARAMS = ['omega_b', 'omega_cdm', 'n_s', 'sigma8', 'w0', 'wa']
fc = compare_methods.Forecaster(ROOT / 'models_pairs', '_pairs', compress=True, cov='covbox', fiducial='official')
q = lambda m: [f'{m}_rand_cross_q1', f'{m}_rand_cross_q4']
vec = {'2PCF': ['gg'], 'ASTRA': q('astra'), 'DS': q('ds'), 'ASTRA+DS': q('astra') + q('ds')}
out = {k: {str(s): np.log10(fc(v, s)['marg'][0]).tolist() for s in (20., 30., 40., 50., 60., 70.)} for k, v in vec.items()}
(ROOT / 'data' / 'smin_numbers.json').write_text(json.dumps(out, indent=1))
for k in out: print(k, {s: np.round(x, 2).tolist() for s, x in out[k].items()})
