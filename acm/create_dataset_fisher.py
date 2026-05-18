# ============================================================
# ABACUS + ASTRA
# QUICK DATASET BUILDER
# 2 SETS:
#   1) covariance set  -> many fiducial realizations (cosmo 0)
#   2) derivative set  -> cosmology variations (+/- params)
#
# Goal:
# save everything needed later for Fisher matrices + figures
# ============================================================


'''
 512/512] moy= 11.7s/fichier | écoulé= 100.1min | ETA≈   0.0min | RAM≈21.4GB | OK=512 échecs=0 
 N_BINS = 25
R_MIN = 5
size_box=2000 
cut_length= 250
N_workers=16
R_MAX=200
N_COV_FILES =160 
N_PER_VARIATION = 64



Saved: dataset_covariance_astra_1500_1_250_200.npz
  Total=23.0min | moy=2.8s/file | OK=500/500

Total=23.0min | moy=2.8s/file | OK=500/500
Saved: dataset_derivatives_astra_64_1_250_200.npz
  Total=23.3min | moy=2.7s/file | OK=512/512
'''
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import fitsio

# your libs
import estimators.galaxy_clustering.astra_split as astra_split
from estimators.galaxy_clustering.density_split import DensitySplit
from pycorr import TwoPointCorrelationFunction

# ============================================================
# FAST TEST CONFIG
# ============================================================

SUBSAMPLE_FACTOR = 1
N_WORKERS = 64 # 20 max in the terminal
NTHREADS = 1 # doesn't work, stop at cross-correlation
SEED = 42


rng = np.random.default_rng(SEED)

# pair counts
N_BINS = 25
R_MIN = 5

size_box=2000 # total full 2000
cut_length= int(500/2) #250 
#R_MAX = int(200*cut_length*2/size_box)
R_MAX=200

# 100 o 150

# ------------------------------------------------------------
# how many realizations
# ------------------------------------------------------------

# covariance = many fiducial realizations
N_COV_FILES = 1500#100

# derivatives = a few per varied cosmology
N_PER_VARIATION = 64 #30 (*8 )

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/")
COSMOLOGY_FILE = "/global/homes/h/hugon/home/acm/nb/cosmologies.csv"

OUT_COV = f"dataset_covariance_astra_{N_COV_FILES}_{SUBSAMPLE_FACTOR}_{cut_length}_{R_MAX}.npz"
OUT_DERIV = f"dataset_derivatives_astra_{N_PER_VARIATION}_{SUBSAMPLE_FACTOR}_{cut_length}_{R_MAX}.npz"




# ============================================================
# COSMOLOGY CHOICES
# ============================================================

# Fiducial
FID_COSMO = 0

# Available Abacus variations
COSMOS_VAR = [
    100, 101,   # +/- omega_b
    102, 103,   # +/- omega_cdm
    104, 105,   # +/- n_s
    112, 113    # +/- sigma8
]

COSMOS_KEEP = [FID_COSMO] + COSMOS_VAR


# ============================================================
# COSMOLOGY TABLE
# ============================================================

FEATURE_COLS = [
    "omega_b",
    "omega_cdm",
    "h",
    "A_s",
    "n_s",
    "sigma8_m",
    "sigma8_cb"
]

cosmo_df = pd.read_csv(COSMOLOGY_FILE)
cosmo_df.columns = cosmo_df.columns.str.strip()
cosmo_df["cosmo_id"] = cosmo_df["root"].str.extract(r"cosm(\d+)").astype(int)
cosmo_df = cosmo_df.set_index("cosmo_id")


# ============================================================
# DISCOVER FILES
# ============================================================

def get_all_hod_files(base_dir):

    rows = []

    for d in base_dir.glob("c*_ph*"):

        cosmo_id = int(d.name.split("_")[0][1:])
        phase_id = int(d.name.split("_ph")[1])

        for seed_dir in sorted(d.glob("seed*")):

            seed_id = int(seed_dir.name.replace("seed", ""))

            for f in sorted(seed_dir.glob("hod*.fits")):

                rows.append(
                    (f, cosmo_id, phase_id, seed_id)
                )

    return rows


all_files = get_all_hod_files(BASE_DIR)

# keep only wanted cosmologies
all_files = [x for x in all_files if x[1] in COSMOS_KEEP]

print("TOTAL FILES =", len(all_files))


# ============================================================
# BUILD 2 DATASETS
# ============================================================

# ------------------------------------------------------------
# 1) covariance set = many fiducial realizations
# ------------------------------------------------------------

fid_files = [x for x in all_files if x[1] == FID_COSMO]

print(f"FIDUCIAL FILES =", len(fid_files))

# use many different phases / seeds
files_cov = fid_files[:N_COV_FILES]


# ------------------------------------------------------------
# 2) derivative set = variations
# ------------------------------------------------------------

files_deriv = []

for cid in COSMOS_VAR:

    subset = [x for x in all_files if x[1] == cid][:N_PER_VARIATION]
    files_deriv.extend(subset)


print("COV FILES   =", len(files_cov))
print("DERIV FILES =", len(files_deriv))


# ============================================================
# LOADER
# ============================================================

def get_positions(hod_file, los="z"):

    data, header = fitsio.read(hod_file, header=True)

    qpar  = header["Q_PAR"]
    qperp = header["Q_PERP"]

    if los == "z":

        # real space
        pos_r = np.c_[
            data["X_PERP"],
            data["Y_PERP"],
            data["Z_PERP"]
        ]

        # observed redshift space
        pos_z = np.c_[
            data["X_PERP"],
            data["Y_PERP"],
            data["Z_RSD"]
        ]

        boxsize = np.array([
            size_box/qperp,
            size_box/qperp,
            size_box/qpar
        ])

    elif los == "x":

        pos_r = np.c_[
            data["X_PERP"],
            data["Y_PERP"],
            data["Z_PERP"]
        ]

        pos_z = np.c_[
            data["X_RSD"],
            data["Y_PERP"],
            data["Z_PERP"]
        ]

        boxsize = np.array([
            size_box/qpar,
            size_box/qperp,
            size_box/qperp
        ])

    elif los == "y":

        pos_r = np.c_[
            data["X_PERP"],
            data["Y_PERP"],
            data["Z_PERP"]
        ]

        pos_z = np.c_[
            data["X_PERP"],
            data["Y_RSD"],
            data["Z_PERP"]
        ]

        boxsize = np.array([
            size_box/qperp,
            size_box/qpar,
            size_box/qperp
        ])

    else:
        raise ValueError("los must be x,y,z")

    # print(pos_r.min(axis=0), pos_r.max(axis=0))
    # print(pos_z.min(axis=0), pos_z.max(axis=0))
    # print(boxsize)

    return (
        pos_r.astype(np.float32),
        pos_z.astype(np.float32),
        boxsize.astype(np.float32)
    )

def cut_box(positions, L=250):
    # print extrema before cut
    print("Before cut:"
          f" x=[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}] | "
          f" y=[{positions[:,1].min():.1f}, {positions[:,1].max():.1f}] | "
          f" z=[{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]"
    )
    # in numpy, chained comparisons dont work like in python, forced to separate with & and parentheses
    mask = (
        (positions[:,0] > -L) & (positions[:,0] < L) &
        (positions[:,1] > -L) & (positions[:,1] < L) &
        (positions[:,2] > -L) & (positions[:,2] < L)
    )
    boxsize_cut = np.array([2*L, 2*L, 2*L], dtype=np.float32)

    print("After cut:"
            f" x=[{positions[mask][:,0].min():.1f}, {positions[mask][:,0].max():.1f}] | "
            f" y=[{positions[mask][:,1].min():.1f}, {positions[mask][:,1].max():.1f}] | "
            f" z=[{positions[mask][:,2].min():.1f}, {positions[mask][:,2].max():.1f}]"
        )


    print(f"Cutting box to {L} Mpc/h: keeping {mask.sum()} / {len(positions)} objects")


    return positions[mask],mask, boxsize_cut
# ============================================================
# ASTRA QUINTILES
# ============================================================

def build_astra_quintiles(positions, boxsize, randoms=None):

    astra = astra_split.AstraSplit(n_random=1)

    if randoms is None:
        randoms = astra.generate_uniform_randoms_from_hod(
            positions=positions,
            boxsize=boxsize,
            n_factor=1,
            seed=SEED
        )

    df_full = astra.build_dataframe_from_hod(
        positions=positions,
        random_positions=randoms
    )

    _, class_rows, _ = astra.generate_pairs_classification_probability(df_full)

    df_class = astra.build_final_classification(class_rows)

    df_data = df_class[df_class["ISDATA_BOOL"]].copy()
    print('check len df_data vs positions')
    print(len(df_data), len(positions))
    assert len(df_data) == len(positions), "len df_data should be the same as positions"
    df_data["idx"] = np.arange(len(df_data))


    # orig_idx = df_data["ORIG_INDEX"].values


    labels=np.zeros(len(positions), dtype=np.int8)
    # df_data["idx"] = np.arange(len(df_data)) # incompatible with cut_box
    # df_data["idx"] = df_data.index.values

    rvals=np.zeros(len(positions), dtype=np.float32)
    # rvals[orig_idx] = df_data["r"].values.astype(np.float32)

    # continuous density proxy
    # rvals = df_data["r"].values.astype(np.float32)
    rvals[df_data["idx"].values] = df_data["r"].values.astype(np.float32)

    quintiles = []
    # labels = np.zeros(len(df_data), dtype=np.int8)

    for i, q in enumerate([1,2,3,4]):
        # mask=df_data["QUARTILE"] == q
        # idx= orig_idx[mask]
        idx= df_data[df_data["QUARTILE"] == q]["idx"].values

        # idx = df_data[df_data["QUARTILE"] == q]["idx"].values
        quintiles.append(idx)
        labels[idx] = i
    print("fin build_astra_quintiles")

    return quintiles, labels, rvals


# ============================================================
# DS MULTIPOLES
# ============================================================

def compute_ds_multipoles(positions, quintile_indices, boxsize):
    print('check quintile indices')
    for q in quintile_indices:
        print(f'q_min: {q.min()}, q_max: {q.max()}, len: {len(q)}')

    quintile_positions = [positions[q] for q in quintile_indices]

    ds = DensitySplit(
        boxsize=boxsize,
        data_positions=positions,
        cellsize=10 # change, 25?
    )
    print("Building quantiles...")

    ds.quantiles = quintile_positions

    s_edges = np.linspace(R_MIN, R_MAX, N_BINS + 1)
    mu_edges = np.linspace(-1, 1, 121)
    edges = (s_edges, mu_edges)

    # cross
    print("Computing cross-correlations...")
    cross = ds.quantile_data_correlation(
        data_positions=positions,
        boxsize=boxsize,
        edges=edges,
        los="z",
        nthreads=NTHREADS
    )

    # auto
    print("Computing auto-correlations...")
    auto = ds.quantile_correlation(
        boxsize=boxsize,
        edges=edges,
        los="z",
        nthreads=NTHREADS
    )

    vec = []
    sep = None

    for res in cross + auto:

        s, mp = res(ells=(0, 2), return_sep=True)

        sep = s
        vec.append(mp[0])   # monopole
        vec.append(mp[1])   # quadrupole
    print('fin compute_ds_multipoles')

    return sep, np.concatenate(vec)


# ============================================================
# STANDARD 2PCF
# ============================================================

# def compute_2pcf(positions, boxsize):

#     s_edges = np.linspace(R_MIN, R_MAX, N_BINS + 1)
#     mu_edges = np.linspace(-1, 1, 121)
#     edges = (s_edges, mu_edges)

#     result = TwoPointCorrelationFunction(
#         data_positions1=positions,
#         mode="smu",
#         edges=edges,
#         boxsize=boxsize,
#         los="z",
#         nthreads=NTHREADS,
#         position_type="pos"
#     )

#     s, mp = result(ells=(0, 2), return_sep=True)


#     vec = np.concatenate([mp[0], mp[1]])

#     return s, vec



# def compute_2pcf(positions, factor_randoms=1):

#     s_edges = np.linspace(R_MIN, R_MAX, N_BINS + 1)
#     mu_edges = np.linspace(-1, 1, 121)
#     edges = (s_edges, mu_edges)

#     box_min = positions.min(axis=0)
#     box_max = positions.max(axis=0)
#     n_rand  = factor_randoms * len(positions)

#     randoms = np.random.uniform(
#         low=box_min, high=box_max,
#         size=(n_rand, 3)
#     ).astype(np.float32)

#     DD = TwoPointCorrelationFunction(
#         data_positions1=positions,
#         data_positions2=positions,
#         randoms_positions1=randoms,
#         randoms_positions2=randoms,
#         mode="smu",
#         edges=edges,
#         estimator="natural",
#         position_type="pos",
#         los="midpoint",
#         nthreads=NTHREADS,
#     )

#     RR = TwoPointCorrelationFunction(
#         data_positions1=randoms,
#         data_positions2=randoms,
#         randoms_positions1=randoms,
#         randoms_positions2=randoms,
#         mode="smu",
#         edges=edges,
#         estimator="natural",
#         position_type="pos",
#         los="midpoint",
#         nthreads=NTHREADS,
#     )

#     s, DD_mp = DD(ells=(0, 2), return_sep=True)
#     RR_mp    = RR(ells=(0, 2), return_sep=False)

#     xi0 = DD_mp[0] / (RR_mp[0] + 1e-10) - 1
#     xi2 = DD_mp[1] / (RR_mp[1] + 1e-10)

#     vec = np.concatenate([xi0, xi2])
#     print('fin compute_2pcf')

#     return s, vec

def compute_2pcf(positions, factor_randoms=4):

    s_edges = np.linspace(R_MIN, R_MAX, N_BINS + 1)
    mu_edges = np.linspace(-1, 1, 121)
    edges = (s_edges, mu_edges)

    box_min = positions.min(axis=0)
    box_max = positions.max(axis=0)
    n_rand  = factor_randoms * len(positions)

    randoms = np.random.uniform(
        low=box_min, high=box_max,
        size=(n_rand, 3)
    ).astype(np.float32)

    result = TwoPointCorrelationFunction(
        mode="smu",
        edges=edges,
        data_positions1=positions,
        randoms_positions1=randoms,
        estimator="landyszalay",
        position_type="pos",
        los="midpoint",
        nthreads=NTHREADS,
    )

    s, mp = result(ells=(0, 2), return_sep=True)
    vec = np.concatenate([mp[0], mp[1]])
    print('fin compute_2pcf')

    return s, vec


# ============================================================
# ONE FILE
# ============================================================

def process_one(entry):

    hod_file, cosmo_id, phase_id, seed_id = entry

    try:
        # params cosmologiques
        params = cosmo_df.loc[cosmo_id, FEATURE_COLS].values.astype(np.float32)


        pos_r_full, pos_z_full, boxsize = get_positions(hod_file)
        N_full = len(pos_r_full)
        assert len(pos_z_full) == N_full, "pos_r and pos_z should have the same length"

        # global index
        gal_idx = np.arange(N_full)

        _, mask_cut, boxsize = cut_box(pos_r_full, L=250)
        pos_r   = pos_r_full[mask_cut]
        pos_z   = pos_z_full[mask_cut]
        gal_idx = gal_idx[mask_cut]

        print(f"After cut: {len(pos_r)} galaxies")

        n = len(pos_r) // SUBSAMPLE_FACTOR
        idx_sub = rng.choice(len(pos_r), size=n, replace=False)

        pos_r   = pos_r[idx_sub]
        pos_z   = pos_z[idx_sub]
        gal_idx = gal_idx[idx_sub]

        assert len(pos_r) == len(pos_z), f"Not same sizes: {len(pos_r)} vs {len(pos_z)}"
        print(f"After subsample: {len(pos_r)} galaxies")



        # pos_r, pos_z, boxsize = get_positions(hod_file)

        # cut box
        # pos_r, mask, boxsize = cut_box(pos_r, L=250)
        # pos_z, mask, boxsize = cut_box(pos_z, L=250)

        # print(f"pos_r={len(pos_r)}, pos_z={len(pos_z)}")

        # # même taille pour r et z
        # n_common = min(len(pos_r), len(pos_z))
        # idx_r = rng.choice(len(pos_r), size=n_common, replace=False)
        # idx_z = rng.choice(len(pos_z), size=n_common, replace=False)
        # pos_r = pos_r[idx_r]
        # pos_z = pos_z[idx_z]

        # # subsample (SUBSAMPLE_FACTOR=1 → pas de subsample)
        # n   = n_common // SUBSAMPLE_FACTOR
        # idx = rng.choice(n_common, size=n, replace=False)
        # pos_r = pos_r[idx]
        # pos_z = pos_z[idx]

        # assert len(pos_r) == len(pos_z), f"Not same sizes: {len(pos_r)} vs {len(pos_z)}"

        # randoms ASTRA dans le même sous-volume que les données
        box_min = pos_r.min(axis=0)
        box_max = pos_r.max(axis=0)
        randoms_astra = rng.uniform(
            low=box_min, high=box_max,
            size=(len(pos_r), 3)
        ).astype(np.float32)

        # ASTRA data
        print("ASTRA real space...")
        quint_r, label_r, rvals_r = build_astra_quintiles(pos_r, boxsize)
        print("ASTRA redshift space...")
        quint_z, label_z, rvals_z = build_astra_quintiles(pos_z, boxsize)

        # ASTRA randoms
        print("ASTRA randoms...")
        quint_rand, label_rand, rvals_rand = build_astra_quintiles(randoms_astra, boxsize)

        assert len(label_r) == len(pos_r), "label_r bad dimensions"
        assert len(label_z) == len(pos_z), "label_z bad dimensions"

        # transition matrix r → z
        transition = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            mask_q = (label_r == i)
            for j in range(4):
                transition[i, j] = np.mean(label_z[mask_q] == j)

        # sample pour visualisation
        keep     = min(5000, len(pos_z))
        idx_keep = rng.choice(len(pos_z), size=keep, replace=False)

        pos_sample       = pos_z[idx_keep]
        pos_r_sample       = pos_r[idx_keep]
        label_r_sample   = label_r[idx_keep]
        label_z_sample   = label_z[idx_keep]
        rvals_r_sample   = rvals_r[idx_keep]
        rvals_z_sample   = rvals_z[idx_keep]
        gal_idx_sample   = gal_idx[idx_keep]

        # DS multipoles data
        print("Density split multipoles (redshift space)...")
        s, ds_z = compute_ds_multipoles(pos_z, quint_z, boxsize)
        print("Density split multipoles (real space)...")
        _, ds_r = compute_ds_multipoles(pos_r, quint_r, boxsize)

        # DS multipoles randoms
        print("Density split multipoles (randoms)...")
        _, ds_rand = compute_ds_multipoles(randoms_astra, quint_rand, boxsize)

        # 2PCF
        print("Global 2PCF...")
        s, pcf = compute_2pcf(pos_z)

        # counts
        counts_r    = np.array([len(q) for q in quint_r])
        counts_z    = np.array([len(q) for q in quint_z])
        counts_rand = np.array([len(q) for q in quint_rand])

        print(f"DONE {hod_file.name} | n={len(pos_z)}")

        return dict(
            cosmo_id=cosmo_id,
            phase_id=phase_id,
            seed_id=seed_id,
            params=params,

            s=s,
            ds_z=ds_z,
            ds_r=ds_r,
            ds_rand=ds_rand,
            pcf=pcf,

            counts_r=counts_r,
            counts_z=counts_z,
            counts_rand=counts_rand,

            label_r=label_r,
            label_z=label_z,
            label_rand=label_rand,

            transition=transition,

            pos_sample=pos_sample,
            pos_r_sample=pos_r_sample,

            gal_idx_sample=gal_idx_sample,
            
            label_r_sample=label_r_sample,
            label_z_sample=label_z_sample,

            rvals_r_sample=rvals_r_sample,
            rvals_z_sample=rvals_z_sample,

            rvals_r=rvals_r,
            rvals_z=rvals_z,
            rvals_rand=rvals_rand,
        )

    except Exception as e:
        print("ERROR:", hod_file, e)
        return None

# ============================================================
# RUN DATASET
# ============================================================

import time
import os
import psutil


def run_dataset(filelist, outfile):

    rows = []
    t_start = time.perf_counter()
    n_total = len(filelist)
    proc=psutil.Process(os.getpid())
    print(f"Memory usage at start: {proc.memory_info().rss / 1e9:.2f} GB")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:

        futures = {ex.submit(process_one, f) for f in filelist}

        # for fu in tqdm(as_completed(futures), total=len(futures)):
        for i, fu in enumerate(tqdm(as_completed(futures), total=n_total), 1):
            try:
                r = fu.result()
                if r is not None:
                    rows.append(r)
            except Exception as e:
                print(f"\nERROR (pool): {futures[fu]} → {e}")

            # log toutes les 5 tâches
            if i % 5 == 0 or i == n_total:
                elapsed  = time.perf_counter() - t_start
                avg      = elapsed / i
                eta_s    = avg * (n_total - i)

                # RAM du process principal + enfants
                mem_main = proc.memory_info().rss
                children = proc.children(recursive=True)
                mem_workers = sum(c.memory_info().rss for c in children
                                  if c.is_running())
                mem_total_gb = (mem_main + mem_workers) / 1e9

                tqdm.write(
                    f"  [{i:>4}/{n_total}] "
                    f"moy={avg:5.1f}s/fichier | "
                    f"elapsed={elapsed/60:6.1f}min | "
                    f"ETA≈{eta_s/60:6.1f}min | "
                    f"RAM≈{mem_total_gb:.1f}GB | "
                    f"OK={len(rows)} failed={i-len(rows)}"
                )

    if not rows:
        print("No valid lines processed, exiting without saving.")
        return

    out = {k: np.array([r[k] for r in rows], dtype=object) for k in rows[0]}
    np.savez_compressed(outfile, **out)

    total = time.perf_counter() - t_start
    print(f"\nSaved: {outfile}")
    print(f"  Total={total/60:.1f}min | moy={total/n_total:.1f}s/file | "
          f"OK={len(rows)}/{n_total}")





# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n--- BUILD COVARIANCE SET ---")
    run_dataset(files_cov, OUT_COV)

    print("\n--- BUILD DERIVATIVE SET ---")
    run_dataset(files_deriv, OUT_DERIV)