import sys, numpy as np
sys.path.insert(0, '/global/u1/h/hugon/home/astra-ds'); sys.path.insert(0, '/global/u1/h/hugon/home/astra-ds/scripts')
import astra_ds as D
from run_quijote import read_halos, QUIJOTE, BOX
r = {}
gal, box = D.load_catalog(D.hod_path(0, 239))
res, _ = D.measure_classes(gal, box, 'abacus', nside=4, subboxes=[0, 1, 2, 3], nthreads=64)
print('ABACUS fractions rand', res['fraction_rand'].mean(0).round(3), 'data', res['fraction_data'].mean(0).round(3))
for i in (1000, 1001):
    pos = read_halos(QUIJOTE / 'fiducial' / str(i) / 'out_3_pid.list')
    res, _ = D.measure_classes(pos, np.full(3, BOX), 'quijote', nside=1, periodic_clustering=True, nthreads=64)
    print('QUIJOTE', i, 'fractions rand', res['fraction_rand'][0].round(3), 'data', res['fraction_data'][0].round(3))
