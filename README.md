# astra-ds — ASTRA split vs density split on DESI-like HOD mocks

Environment-dependent galaxy clustering measured on AbacusSummit HOD mocks
(DESI LRG-like, z = 0.5), comparing two ways of splitting the density field
into quartiles:

| | ASTRA split | Density split (Paillas et al.) |
|---|---|---|
| environment value | `r = (N_data - N_rand)/(N_data + N_rand)` over the Delaunay neighbours in the joint galaxy + random tessellation | Gaussian-smoothed (R = 10 Mpc/h) galaxy density contrast read at the point |
| evaluated on | the full periodic 2 Gpc/h box (periodic padding of 30 Mpc/h around each subbox) | the full periodic 2 Gpc/h box (FFT smoothing) |

Both use **the same random points**, so the two classifications can be
compared point by point.

## What is measured

For each HOD catalogue the 2 Gpc/h box is cut into **64 subboxes of 500 Mpc/h**.
In each subbox (open volume, Landy–Szalay, line of sight z, ℓ = 0, 2, 4,
s ∈ [0, 150] Mpc/h in 5 Mpc/h bins):

* `gg` — galaxy 2PCF,
* `{astra,ds}_{data,rand}_{auto,cross}_q{1..4}` — for each method and each
  population (galaxies ranked by their environment, or random "query points"
  ranked by theirs; equal-count quartiles, Q1 = most underdense), the quartile
  auto-correlation and its cross-correlation with all galaxies.

The 64 subboxes of one catalogue give the covariance (one fixed HOD, c000);
paired subbox differences between the ± cosmologies of the AbacusSummit
derivative grid give the derivatives.

## Why the previous pipeline (`acm/acm/create_dataset_fisher.py`) was replaced

1. `hod000…hod499` are 500 *different HOD parameter sets* (emulator Latin
   hypercube), not realisations. Averaging them mixed the HOD scatter (~55 % on
   ξ₀) into covariances and derivatives (cosmological signal ~1 %): the old
   derivatives were compatible with noise (median |S/N| < 1 for ω_b, n_s, σ₈).
   Now: one HOD per cosmology, matched within each ± pair (`select_hods.py`).
2. Only the central 500 Mpc/h cube was used; now all 64 subboxes.
3. Pair counts used periodic boundary conditions on a non-periodic cut box;
   now Landy–Szalay with geometry randoms.
4. The 2PCF used `los="midpoint"` (radial from the box centre) which erased the
   quadrupole; now `los="z"`, the RSD axis.
5. Quartiles were computed jointly on galaxies + randoms (unequal,
   realisation-dependent counts); now separately, with exactly equal counts.
6. ASTRA ran a scipy Delaunay on the cut box (edge effects); now on a
   periodically padded region, and vectorised (~1 s per subbox).

## Usage (NERSC)

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python scripts/select_hods.py                      # -> config/hod_selection.json
sbatch slurm/run_catalog.sbatch 0 239 z            # one catalogue, ~10 min on 1 CPU node
python scripts/plot_multipoles.py data/c000_hod239_z.npz
python scripts/derivatives.py                      # once the 8 ± catalogues are done
```

Catalogues: `/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/c{cosmo}_ph000/seed0/hod{NNN}.fits`.
Positions `X_PERP, Y_PERP, Z_RSD` (redshift space) or `Z_PERP` (real space) are
already Alcock–Paczynski rescaled and are used as is.

## Files

* `astra_ds.py` — the whole method (loading, subboxes, ASTRA, density split, clustering)
* `scripts/select_hods.py` — HOD-matched catalogues for the derivatives
* `scripts/run_catalog.py` — measure one catalogue → `data/c{cosmo}_hod{hod}_{space}.npz`
* `scripts/plot_multipoles.py` — ASTRA vs DS multipoles, environment comparison
* `scripts/derivatives.py` — paired central differences
* `config/` — cosmology table, HOD selection
