

import argparse
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt

import os


# attention nthreads, increase max--files and careful with memory

# import jax
# jax.config.update("jax_enable_x64", False)

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# import jax
# print(jax.devices())
from pycorr import TwoPointCorrelationFunction


import numpy as np
import pandas as pd
import fitsio
import yaml
from scipy.spatial import cKDTree
from itertools import product
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Ensure local package imports work when running the script from inside acm/acm/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import estimators.galaxy_clustering.astra_split as astra_split
from estimators.galaxy_clustering.density_split import DensitySplit



# ------------------------------------------------------------
# PATHS / PARAMS
# ------------------------------------------------------------

COSMOLOGY_FILE = "/global/homes/h/hugon/home/acm/nb/cosmologies.csv"

FEATURE_COLS = [
    "omega_b","omega_cdm","h","A_s","n_s","alpha_s",
    "N_ur","N_ncdm","omega_ncdm","w0_fld","wa_fld",
    "sigma8_m","sigma8_cb"
]

BASE_DIR = Path('/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/')




SUBSAMPLE_FACTOR = 5
n_threads = 1
max_files=160
n_workers=8

N_BINS = 25
R_MIN = 5
R_MAX = 200

SEED = 42
rng = np.random.default_rng(SEED)

CHECKPOINT_FILE = f"dataset_checkpoint_3_jax{SUBSAMPLE_FACTOR}_{max_files}.npz"
OUTPUT_FILE = f"dataset_fcn_nano_3_jax{SUBSAMPLE_FACTOR}_{max_files}.npz"

def get_all_hod_files(base_dir):
    all_files = []

    for cosmo_phase_dir in base_dir.glob("c*_ph*"):
        cosmo_id = int(cosmo_phase_dir.name.split("_")[0][1:])

        for seed_dir in cosmo_phase_dir.glob("seed*"):
            for f in seed_dir.glob("hod*.fits"):
                all_files.append((f, cosmo_id))

    return all_files


all_files = get_all_hod_files(BASE_DIR)

print("Total files found:", len(all_files))


def get_hod_positions(hod_file, los='z'):
    """
    Official loader: correct RSD handling + boxsize.
    """
    data, header = fitsio.read(hod_file, header=True)

    qpar, qperp = header['Q_PAR'], header['Q_PERP']

    if los == 'x':
        positions = np.c_[data['X_RSD'], data['Y_PERP'], data['Z_PERP']]
        boxsize = np.array([2000/qpar, 2000/qperp, 2000/qperp])

    elif los == 'y':
        positions = np.c_[data['X_PERP'], data['Y_RSD'], data['Z_PERP']]
        boxsize = np.array([2000/qperp, 2000/qpar, 2000/qperp])

    elif los == 'z':
        positions = np.c_[data['X_PERP'], data['Y_PERP'], data['Z_RSD']]
        boxsize = np.array([2000/qperp, 2000/qperp, 2000/qpar])

    else:
        raise ValueError("los must be 'x', 'y', or 'z'")

    return positions, boxsize

cosmo_df = pd.read_csv(COSMOLOGY_FILE)
cosmo_df.columns = cosmo_df.columns.str.strip()
cosmo_df["cosmo_id"] = cosmo_df["root"].str.extract(r"cosm(\d+)").astype(int)
cosmo_df = cosmo_df.set_index("cosmo_id")



# ------------------------------------------------------------
# DATA COMPUTATION
# ------------------------------------------------------------



def compute_data_vector(hod_file):

    
    # 1. Load and subsample the HOD positions
    positions, boxsize = get_hod_positions(hod_file, los='z')
    n_data = len(positions) // SUBSAMPLE_FACTOR
    idx = rng.choice(len(positions), size=n_data, replace=False)
    positions_sub=np.asarray(positions[idx], dtype=np.float32)  # pour jax


    Ng = len(positions_sub)

    print("=== Check coordinates ===")
    print("positions_sub shape:", positions_sub.shape)
    print("dtype:", positions_sub.dtype)
    print("min:", positions_sub.min(axis=0))
    print("max:", positions_sub.max(axis=0))
    print("boxsize:", boxsize)

    # check NaN / inf
    print("NaNs:", np.isnan(positions_sub).sum())
    print("Infs:", np.isinf(positions_sub).sum())

    print("OUT OF BOX:", np.sum((positions_sub < - boxsize) | (positions_sub > boxsize)))



    # 2. Randoms ASTRA only for classification
    astra = astra_split.AstraSplit(n_random=1)
    random_astra = astra.generate_uniform_randoms_from_hod(
        positions=positions_sub, boxsize=boxsize, n_factor=1, seed=SEED
    )
    df_full = astra.build_dataframe_from_hod(
        positions=positions_sub, random_positions=random_astra
    )
    _, class_rows, _ = astra.generate_pairs_classification_probability(df_full)
    df_class = astra.build_final_classification(class_rows)
    
    df_data = df_class[df_class["ISDATA_BOOL"]].copy()
    df_data["idx"] = np.arange(len(df_data))  # indexes of galaxies in the subsample

    # 3. Extract quartiles from the classification dataframe
    quartile_positions = []
    for q in [1, 2, 3, 4]:
        idx_q = df_data[df_data["QUARTILE"] == q]["idx"].values
        quartile_positions.append(positions_sub[idx_q])
    
    for i, q in enumerate(quartile_positions):
        print(f"Q{i}: size={len(q)}, min={q.min() if len(q)>0 else None}, max={q.max() if len(q)>0 else None}")
    
    print("max TARGETID:", df_data["TARGETID"].max())
    print("len positions_sub:", len(positions_sub))


    # 4. DensitySplit pour les corrélations (randoms séparés, 10×Ng)
    ds = DensitySplit(boxsize=boxsize,data_positions=positions_sub,cellsize=10)   
    ds.quantiles = quartile_positions # direct injection of ASTRA's quartiles 

    # No need to generate randoms with N_R = 10 * (Ng+Nr) for the quantiles since DensitySplit will handle it internally based on the data positions and boxsize.
    s_edges = np.linspace(R_MIN, R_MAX, N_BINS + 1)
    mu_edges = np.linspace(-1, 1, 121)  # standard: 120 bins from -1 to 1
    edges = (s_edges, mu_edges)
    print("Edges shapes:", [e.shape for e in edges])

    # 5. Crossed correlations quantile × positions_sub
    results_cross = ds.quantile_data_correlation(
        data_positions=positions_sub,
        boxsize=boxsize,
        edges=edges,
        los='z',
        nthreads=n_threads,
    )

    # print("edges before 6:", edges)

    # 6. Auto-correlation quantiles
    results_auto = ds.quantile_correlation(
        boxsize=boxsize,
        edges=edges,
        los='z',
        nthreads=n_threads,
    )

    # print("edges after correlations:", edges)

    # 7. Extract xi0 et xi2
    data_vector = []
    for result in results_cross + results_auto:
        s, multipoles = result(ells=(0, 2), return_sep=True)
        data_vector.extend(multipoles[0])  # xi0
        data_vector.extend(multipoles[1])  # xi2

    
    # print("edges after extracting multipoles:", edges)
    # --------------------------------------------------
    # 7bis. Galaxy 2PCF (GLOBAL)
    # --------------------------------------------------

    print("Sanity check:")
    print("unique positions:", len(np.unique(positions_sub, axis=0)))
    print("NaNs:", np.isnan(positions_sub).sum())
    
    print("positions_sub shape:", positions_sub.shape)
    print("quartile shapes:", [q.shape for q in quartile_positions])
    print("quartile types:", [type(q) for q in quartile_positions])
    for i, q in enumerate(quartile_positions):
        print(i, q.shape, np.isnan(q).sum())

    print("boxsize:", boxsize)
    print("positions_sub shape:", positions_sub.shape)
    print("min and max positions_sub:", positions_sub.min(axis=0), positions_sub.max(axis=0))
    result_2pcf = TwoPointCorrelationFunction(
        data_positions1=positions_sub,
        mode='smu',
        edges=edges,
        boxsize=boxsize,
        los='z',
        nthreads=n_threads,
        position_type='pos',
    )

    s, multipoles = result_2pcf(ells=(0, 2), return_sep=True)

    data_vector.extend(multipoles[0])  # xi0
    data_vector.extend(multipoles[1])  # xi2

    return np.array(data_vector)

# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------

def process_entry(entry):
    hod_file, cosmo_id = entry
    try:
        params = cosmo_df.loc[cosmo_id, FEATURE_COLS].values.astype(float)
        data_vector = compute_data_vector(hod_file)
        return params, data_vector
    except Exception as e:
        print(f"Error {hod_file}: {e}")
        return None

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------



def main():
    # parser = argparse.ArgumentParser(description="Build emulator dataset from HOD files.")
    # parser.add_argument(
    #     "--max-files", type=int, default=None,
    #     help="Nombre max de fichiers à traiter (default: tous)"
    # )
    # args = parser.parse_args()


    files_to_process = all_files

    if max_files is not None:
        rng.shuffle(files_to_process)
        files_to_process = files_to_process[:max_files]

    print(f"Processing {len(files_to_process)} / {len(all_files)} files")

    N_WORKERS = min(n_workers, multiprocessing.cpu_count())

    X_list = []
    Y_list = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:

        futures = [executor.submit(process_entry, e) for e in files_to_process]

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):

            result = future.result()
            if result is None:
                continue

            X, Y = result
            X_list.append(X)
            Y_list.append(Y)

            if (i + 1) % 500 == 0:
                print(f"Checkpoint at {i+1}")
                np.savez_compressed(
                    CHECKPOINT_FILE,
                    X=np.array(X_list),
                    Y=np.array(Y_list)
                )

    np.savez_compressed(
        OUTPUT_FILE,
        X=np.array(X_list),
        Y=np.array(Y_list)
    )

    print("Done. Saved:", OUTPUT_FILE)

# ------------------------------------------------------------

if __name__ == "__main__":
    main()