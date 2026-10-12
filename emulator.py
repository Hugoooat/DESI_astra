"""
Neural-network emulator of the clustering statistics, in the spirit of
SUNBIRD (Cuesta-Lazaro et al. 2024): one fully-connected network per
statistic maps (cosmology, HOD) parameters to the multipoles measured in the
full 2 Gpc/h box (mean of the 64 subboxes).

* inputs : 8 cosmological parameters (omega_b, omega_cdm, sigma8, n_s,
  alpha_s, N_ur, w0, wa; h and A_s are fixed by theta_* and sigma8 in
  AbacusSummit) + 12 HOD parameters, rescaled to [0, 1] on the training set;
* outputs: monopole and quadrupole, s in [smin, smax], rebinned like the
  Fisher data vectors, standardised by the statistical error of the full box
  (fiducial subbox scatter / 8), so the loss is a chi^2 per bin;
* smooth activations (SiLU) so that the emulator can be differentiated;
* an ensemble of networks with different seeds per statistic: the mean is the
  prediction, the scatter an estimate of the training noise.

Training/validation/test splits are made by *cosmology*: no catalogue of a
held-out cosmology is seen during training.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

import analysis as A

MODELS = A.ROOT / 'models'
COSMO_INPUTS = ['omega_b', 'omega_cdm', 'sigma8_m', 'n_s', 'alpha_s', 'N_ur', 'w0_fld', 'wa_fld']
TEST_COSMOS = [130, 135, 140, 145, 150, 160, 170, 180]
VALID_COSMOS = [131, 137, 143, 151, 157, 165, 173, 177]


def load_dataset(stat, ells=(0, 2), smin=20., smax=150., rebin=2):
    """Inputs X (n, 20), outputs Y (n, ndim) for one statistic, and catalogue ids."""
    d = np.load(A.DATA / 'emulator_dataset.npz')
    names = list(d['x_names'])
    cols = [names.index(c) for c in COSMO_INPUTS] + [names.index(h) for h in names[10:]]
    X = d['X'][:, cols]
    meta = {'s': d['s'], 'ells': d['ells']}
    s, Y = A.data_vector(meta, {stat: d[f'Y/{stat}']}, [stat], ells, smin, smax, rebin)
    return X, Y, d['ids'], s, [names[c] for c in cols]


def statistical_error(stat, **dv):
    """Error of one full box: scatter of the fiducial subboxes (3 lines of sight) / 8."""
    sel = A.selection()['fiducial']
    meta, stats, _ = A.load(sel['cosmo'], sel['hod'])
    return A.data_vector(meta, stats, [stat], **dv)[1].std(0, ddof=1) / 8.


class MLP(nn.Module):

    def __init__(self, nin, nout, width=256, depth=3):
        super().__init__()
        layers, n = [], nin
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.SiLU()]
            n = width
        layers.append(nn.Linear(n, nout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Emulator:
    """Ensemble of MLPs for one statistic, with its input/output normalisation."""

    def __init__(self, stat, nets, xmin, xmax, ymean, ysigma, s, input_names):
        self.stat, self.nets = stat, nets
        self.xmin, self.xmax = torch.as_tensor(xmin), torch.as_tensor(xmax)
        self.ymean, self.ysigma = torch.as_tensor(ymean), torch.as_tensor(ysigma)
        self.s, self.input_names = s, input_names

    def __call__(self, x, members=False):
        """Prediction in physical units; x is (n, 20) or (20,), numpy or torch."""
        x = torch.as_tensor(x, dtype=torch.float64)
        xn = (x - self.xmin) / (self.xmax - self.xmin)
        ys = torch.stack([net(xn) for net in self.nets])
        ys = ys * self.ysigma + self.ymean
        return ys if members else ys.mean(0)

    def jacobian(self, x, members=False):
        """d prediction / d inputs at a single point x: (ndim, 20), or (nmembers, ndim, 20)."""
        x = torch.as_tensor(x, dtype=torch.float64)
        if members:
            return torch.autograd.functional.jacobian(lambda t: self(t[None], members=True)[:, 0], x).detach().numpy()
        return torch.autograd.functional.jacobian(lambda t: self(t[None])[0], x).detach().numpy()

    def save(self, path):
        torch.save({'stat': self.stat, 'nets': [n.state_dict() for n in self.nets],
                    'arch': (self.nets[0].net[0].in_features, self.nets[0].net[-1].out_features),
                    'xmin': self.xmin, 'xmax': self.xmax, 'ymean': self.ymean, 'ysigma': self.ysigma,
                    's': self.s, 'input_names': self.input_names}, path)

    @classmethod
    def load(cls, stat, directory=MODELS):
        state = torch.load(Path(directory) / f'{stat}.pt', weights_only=False)
        nets = []
        for sd in state['nets']:
            net = MLP(*state['arch']).double()
            net.load_state_dict(sd)
            net.eval()
            nets.append(net)
        return cls(state['stat'], nets, state['xmin'], state['xmax'], state['ymean'], state['ysigma'],
                   state['s'], state['input_names'])


def split(ids, test=TEST_COSMOS, valid=VALID_COSMOS, exclude=()):
    """Boolean masks (train, valid, test); `exclude` cosmologies are removed from all three."""
    cos = ids[:, 0]
    ex = np.isin(cos, exclude)
    te, va = np.isin(cos, test) & ~ex, np.isin(cos, valid) & ~ex
    return ~(te | va | ex), va, te


def train(stat, nseeds=3, epochs=4000, lr=1e-3, weight_decay=1e-6, patience=500, holdout=True, verbose=False,
          exclude=(), pairs=None, lam=1., small_scale_weight=0., sigma_floor=0.05):
    """
    Train the ensemble for one statistic. With holdout=True the test and
    validation cosmologies are excluded from training (validation drives early
    stopping); with holdout=False only the test cosmologies are excluded.

    pairs: optional dict from data/pair_dataset.npz (option b). The loss then
    also includes the chi^2 of the predicted *differences* between catalogues,
        lam * < ((f(x_i) - f(x_j)) - (y_i - y_j))^2 / (sigma_ij^2 + floor^2) >,
    where sigma_ij is the (small) error of the paired difference; catalogues in
    pairs['holdout'] are removed from training altogether.
    small_scale_weight: alpha of the per-bin weight (25 / s)^alpha (option c).
    """
    X, Y, ids, s, names = load_dataset(stat)
    tr, va, te = split(ids, exclude=exclude)
    if not holdout:
        tr = tr | va
    if pairs is not None:
        tr[np.asarray(pairs['holdout'], dtype=int)] = False
    xmin, xmax = X[tr].min(0), X[tr].max(0)
    ymean, ysigma = Y[tr].mean(0), statistical_error(stat)
    tx = lambda m: torch.as_tensor((X[m] - xmin) / (xmax - xmin))
    ty = lambda m: torch.as_tensor((Y[m] - ymean) / ysigma)
    xtr, ytr, xva, yva = tx(tr), ty(tr), tx(va), ty(va)
    w = torch.as_tensor(np.tile((25. / s)**small_scale_weight, len(Y[0]) // len(s)))
    w = w / w.mean()
    if pairs is not None:
        keep = tr[pairs['i']] & tr[pairs['j']]
        pi, pj = torch.as_tensor(pairs['i'][keep]), torch.as_tensor(pairs['j'][keep])
        pdelta = torch.as_tensor(pairs[f'delta/{stat}'][keep] / ysigma)
        pvar = torch.as_tensor((pairs[f'sigma/{stat}'][keep] / ysigma)**2 + sigma_floor**2)
        xall = torch.as_tensor((X - xmin) / (xmax - xmin))
    nets, history = [], []
    for seed in range(nseeds):
        torch.manual_seed(seed)
        net = MLP(X.shape[1], Y.shape[1]).double()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=150, min_lr=1e-6)
        best, best_state, wait = np.inf, None, 0
        for epoch in range(epochs):
            net.train()
            opt.zero_grad()
            loss = (w * (net(xtr) - ytr)**2).mean()
            if pairs is not None:
                pred = net(xall)
                loss = loss + lam * (w * ((pred[pi] - pred[pj]) - pdelta)**2 / pvar).mean()
            loss.backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                vloss = ((net(xva) - yva)**2).mean().item()
            sched.step(vloss)
            if verbose and epoch % 1000 == 0:
                print(f'  {stat} seed {seed} epoch {epoch}: train {loss.item():.2f} valid {vloss:.2f}', flush=True)
            if vloss < best:
                best, best_state, wait = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait > patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        nets.append(net)
        history.append((epoch, loss.item(), best))
        if verbose:
            print(f'  {stat} seed {seed}: {epoch} epochs, train chi2/bin {loss.item():.2f}, valid {best:.2f}', flush=True)
    return Emulator(stat, nets, xmin, xmax, ymean, ysigma, s, names), history


def test_residuals(emu, stat):
    """(true - predicted) for the test cosmologies, in physical units, and the test ids."""
    X, Y, ids, _, _ = load_dataset(stat)
    te = split(ids)[2]
    with torch.no_grad():
        pred = emu(X[te]).numpy()
    return Y[te] - pred, ids[te]
