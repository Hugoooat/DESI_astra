import time

import jax
from matplotlib import path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas import qcut
#from pypower import CatalogFFTPower
from pycorr import TwoPointCorrelationFunction
#from jaxpower import MeshAttrs, ParticleField, FKPField, BinMesh2SpectrumPoles, get_mesh_attrs, compute_mesh2_spectrum, compute_fkp2_shotnoise, compute_box2_normalization

import pandas as pd
from astropy.io import fits
from scipy.spatial import Delaunay
from itertools import combinations
from tqdm import tqdm
import fitsio




class AstraSplit:
    """
    Flexible ASTRA implementation.

    Can:
    - Load survey FITS files (original ASTRA format)
    - Work with HOD data
    - Work directly with custom DataFrames
    """

    def __init__(self, path=None, n_random=10):
        self.path = path
        self.n_random = n_random

    # ============================================================
    # -------------------- LOADING METHODS -----------------------
    # ============================================================

    def load_survey_fits(self):
        """
        Load original ASTRA survey FITS format.
        """
        if self.path is None:
            raise ValueError("Path must be provided for survey loading.")

        with fits.open(self.path) as hdul:
            data = hdul[1].data
            df = pd.DataFrame({
                col: np.array(data[col]).astype(data[col].dtype.newbyteorder('='))
                for col in data.columns.names
            })

        return df

    def build_dataframe_from_hod(self, positions, random_positions):
        """
        Build ASTRA-compatible dataframe from HOD + randoms.
        """

        n_data = positions.shape[0]
        n_rand = random_positions.shape[0]

        df_data = pd.DataFrame({
            "TARGETID": np.arange(n_data),
            "TRACERTYPE": "HOD",
            "XCART": positions[:, 0],
            "YCART": positions[:, 1],
            "ZCART": positions[:, 2],
            "RANDITER": -1
        })

        df_rand = pd.DataFrame({
            "TARGETID": np.arange(n_data, n_data + n_rand),
            "TRACERTYPE": "HOD",
            "XCART": random_positions[:, 0],
            "YCART": random_positions[:, 1],
            "ZCART": random_positions[:, 2],
            "RANDITER": 0
        })

        return pd.concat([df_data, df_rand], ignore_index=True)

    # ============================================================
    # -------------------- RANDOM GENERATION ---------------------
    # ============================================================

    def generate_uniform_randoms_from_hod(self, positions=None, boxsize=None, n_factor=1, seed=None):
        """
        Generate homogeneous randoms inside box.

        If `positions` and `boxsize` are omitted the method will use
        `self.positions` / `self.boxsize` if they were previously set.
        The generated array is stored on `self.random_positions` and also
        returned. An optional `seed` ensures reproducible draws.
        """

        if positions is None:
            if hasattr(self, "positions") and hasattr(self, "boxsize"):
                positions = self.positions
                boxsize = self.boxsize
            else:
                raise ValueError("positions and boxsize must be provided if not set on the AstraSplit instance")

        # persist the inputs on the instance for subsequent calls
        self.positions = positions
        self.boxsize = boxsize

        n_gal = positions.shape[0]
        n_random = n_factor * n_gal

        boxsize = np.asarray(boxsize)

        box_min = -boxsize / 2.0
        box_max = boxsize / 2.0
        # box_min = np.zeros_like(boxsize, dtype=float)
        # box_max = np.asarray(boxsize, dtype=float)

        rng = np.random.default_rng(seed)
        random_positions = rng.uniform(low=box_min, high=box_max, size=(n_random, 3))

        # save on instance for notebook convenience
        self.random_positions = random_positions

        return random_positions

    # ============================================================
    # -------------------- CORE ASTRA ALGORITHM ------------------
    # ============================================================

    def generate_pairs_classification_probability(self, df):

        pair_rows = []
        class_rows = []
        r_by_tid = {}

        coords = df[['XCART', 'YCART', 'ZCART']].values
        targetids = df['TARGETID'].values
        is_data = (df['RANDITER'] == -1).values

        n_points = len(coords)

        if n_points < 4:
            raise ValueError("Not enough points for Delaunay triangulation.")

        print(f"Performing Delaunay triangulation on {n_points} points...")
        tri = Delaunay(coords)
        
        neighbors = {i: set() for i in range(n_points)}

        # Build graph
        for simplex in tqdm(tri.simplices,total=len(tri.simplices), desc="Delaunay simplices"):
            for i, j in combinations(simplex, 2):
                neighbors[i].add(j)
                neighbors[j].add(i)

                tid1, tid2 = targetids[i], targetids[j]
                pair_rows.append((tid1, tid2, 0))  # single run
        print(f"Generated {len(pair_rows)} pairs from Delaunay triangulation.")
        # Compute class + r
        # add tqdm progress bar for this loop since it can be slow for large datasets

        
        # for i, j in combinations(simplex, 2):
        #     neighbors[i].add(j)
        #     neighbors[j].add(i)

        #     tid1, tid2 = targetids[i], targetids[j]
        #     pair_rows.append((tid1, tid2, 0))

        # print(f"Generated {len(pair_rows)} pairs from Delaunay triangulation.")
        # ==============================
        # 2️⃣ Classification loop
        # ==============================

        print("Computing local densities and probabilities...")

        for i, nbrs in tqdm(neighbors.items(),
                            total=n_points,
                            desc="Classifying nodes"):

            tid = targetids[i]

            nbr_indices = list(nbrs)
            ndata = np.sum(is_data[nbr_indices])
            nrand = len(nbr_indices) - ndata

            is_data_flag = bool(is_data[i])
            class_rows.append((tid, 0, is_data_flag, ndata, nrand))

            if is_data_flag and (ndata + nrand) > 0:
                r = (ndata - nrand) / (ndata + nrand)
                r_by_tid.setdefault(tid, []).append(r)

        print("ASTRA run completed.")

        return pair_rows, class_rows, r_by_tid


    # ============================================================
    # -------------------- CLASSIFICATION ------------------------
    # ============================================================

    def classify_type(self, r):
        if -1.0 <= r <= -0.9:
            return 'void'
        elif -0.9 < r <= 0.0:
            return 'sheet'
        elif 0.0 < r <= 0.9:
            return 'filament'
        elif 0.9 < r <= 1.0:
            return 'knot'

    # ============================================================
    # -------------------- SAVE METHODS --------------------------
    # ============================================================

    def save_pairs_fits(self, rows, output_path):

        array = np.array(rows, dtype=[
            ('TARGETID1', 'i8'),
            ('TARGETID2', 'i8'),
            ('RANDITER', 'i4')
        ])

        fits.BinTableHDU(data=array).writeto(output_path, overwrite=True)
        print(f"Saved: {output_path}")

    def save_classification_fits(self, rows, output_path):

        array = np.array(rows, dtype=[
            ('TARGETID', 'i8'),
            ('RANDITER', 'i4'),
            ('ISDATA', 'bool'),
            ('NDATA', 'i4'),
            ('NRAND', 'i4')
        ])

        fits.BinTableHDU(data=array).writeto(output_path, overwrite=True)
        print(f"Saved: {output_path}")

    def save_probability_fits(self, r_by_tid, output_path):

        rows = []

        for tid, r_list in r_by_tid.items():
            total = len(r_list)
            counts = {'void': 0, 'sheet': 0, 'filament': 0, 'knot': 0}

            for r in r_list:
                counts[self.classify_type(r)] += 1

            rows.append((
                tid,
                counts['void'] / total,
                counts['sheet'] / total,
                counts['filament'] / total,
                counts['knot'] / total
            ))

        array = np.array(rows, dtype=[
            ('TARGETID', 'i8'),
            ('PVOID', 'f4'),
            ('PSHEET', 'f4'),
            ('PFILAMENT', 'f4'),
            ('PKNOT', 'f4'),
        ])

        fits.BinTableHDU(data=array).writeto(output_path, overwrite=True)
        print(f"Saved: {output_path}")

    def build_final_classification(self, class_rows, n_quantiles=4):

        df = pd.DataFrame(
            class_rows,
            columns=["TARGETID", "RANDITER", "ISDATA", "NDATA", "NRAND"]
        )

        df["ISDATA_BOOL"] = df["ISDATA"].astype(bool)

        

        df["r"] = np.where(
            (df["NDATA"] + df["NRAND"]) > 0,
            (df["NDATA"] - df["NRAND"]) / (df["NDATA"] + df["NRAND"]),
            np.nan
)

        data_mask = df["ISDATA_BOOL"] & df["r"].notna()

        # === DEBUG ===
        print("r stats (data only):", df.loc[data_mask, "r"].describe())
        print("r unique count:", df.loc[data_mask, "r"].nunique())
        print("quantile cuts:", df.loc[data_mask, "r"].quantile([0, 0.25, 0.5, 0.75, 1.0]))
        # === END DEBUG ===

        # ← REMPLACE l'ancien pd.qcut par ceci :
        _, bin_edges = pd.qcut(
            df.loc[data_mask, "r"],
            n_quantiles,
            retbins=True,
            duplicates='drop'
        )
        n_actual_bins = len(bin_edges) - 1
        print("Actual number of bins:", n_actual_bins)

        df["QUARTILE"] = np.nan
        df.loc[data_mask, "QUARTILE"] = pd.qcut(
            df.loc[data_mask, "r"],
            n_quantiles,
            labels=list(range(1, n_actual_bins + 1)),
            duplicates='drop'
        )

        print("Quartile distribution:", df.loc[data_mask, "QUARTILE"].value_counts())

        return df