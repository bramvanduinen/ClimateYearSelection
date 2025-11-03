# %%
# %%
import itertools
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from math import comb
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from ot import sliced_wasserstein_distance
from scipy.stats import energy_distance
from tqdm import tqdm

from cluster_weather_years.constants import (
    BASE_OUTPUT,
    HYDROLOGICAL_YEARS,
    MODELS,
    PROBS,
)
from cluster_weather_years.functions import build_df_centroids, build_df_random

# %% SETTINGS

NUM_DRAWS = int(1e5)  # number of random draws to compare with, default = 1e5
NUM_CLUSTERS = 5  # Number of Climate Years to calculate
TUNE = False  # whether to run in tuning mode (for thresholds)
# ALL_MATCH_COUNTRIES = ["NL", "BE", "FR", "DE"]
ALL_MATCH_COUNTRIES = ["NL"]

# RUN_ID = "NL_20251024_1320"  # NL 30y; Run to analyze and compare to random draws
RUN_ID = "NL_20251023_0942"  # NL 5y; Run to analyze and compare to random draws

# RUN_ID_alternative = "NL_BE_FR_DE_20251015_0801"  # Multi, 30y

# %%
seed = 42
random.seed(seed)

# build df
DF_PROD_DEMAND_FNAME = os.path.join(
    BASE_OUTPUT, "dataframes/df_prod_demand_daily_multi_country_WRONG_CAPACITIES.csv"
)
SA_OUTPUT_DIR = os.path.join(BASE_OUTPUT, "simulated_annealing")

df_daily = pd.read_csv(DF_PROD_DEMAND_FNAME, index_col=0)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.query("Hydrological_year > 2014 and Hydrological_year < 2044")

# settings for random-draws
df_idx = df_daily.set_index(["model", "Hydrological_year"]).sort_index()
keys = df_idx.index.unique().to_numpy()
rng = np.random.default_rng(seed)

vars_to_match = (
    [f"{country}_OFF_prod" for country in ALL_MATCH_COUNTRIES]
    + [f"{country}_ON_prod" for country in ALL_MATCH_COUNTRIES]
    + [f"{country}_SOLAR_prod" for country in ALL_MATCH_COUNTRIES]
    + [f"{country}_Demand" for country in ALL_MATCH_COUNTRIES]
)

MODEL_YEARS = list(itertools.product(MODELS, HYDROLOGICAL_YEARS))

# %%
df_selections = pd.read_csv(
    os.path.join(SA_OUTPUT_DIR, f"run_{RUN_ID}/results_grid.csv"), index_col=0
)
df_sel_multi_country = df_selections.query(
    "optimized_metric == 'wasserstein_score'"
).sort_values("best_score", ascending=True)

centroids_k_medoids = pd.read_csv(
    os.path.join(BASE_OUTPUT, "centroids/centroid_years_load.csv"), index_col=0
)
centroids_k_medoids_sel = centroids_k_medoids[["model", "Hydrological_year"]].values

# %%
columns = vars_to_match.copy()

# Create datasets dictionary with all available selections from df_sel_combined
datasets_optimized = {}
datasets_random = {}

print("Building datasets...")
for i in range(len(df_sel_multi_country)):
    datasets_optimized[f"Multi-country {i + 1}"] = build_df_centroids(
        df_daily, eval(df_sel_multi_country.iloc[i]["selected"])
    )

for i in tqdm(range(int(NUM_DRAWS))):
    datasets_random[f"Random_N {i + 1}"] = (
        pd.DataFrame()
    )  # Placeholder for random dataset

datasets_optimized["K_medoids"] = build_df_centroids(
    df_daily, centroids_k_medoids_sel, mode="Hydrological_year"
)

# Combine optimized and random datasets, for looping
datasets = datasets_optimized | datasets_random

# %%
sample_percentage = 100 * NUM_DRAWS / comb(len(MODEL_YEARS), NUM_CLUSTERS)
print(
    f"Built {NUM_DRAWS:.0e} datasets, which is {sample_percentage:.2e}% of all possible combinations."
)
# %% thresholds for selections

if NUM_CLUSTERS == 30:
    threshold_mean = 0.01  # threshold for mean difference, witin 1% of full set
    threshold_energy = 0.075  # based on tuning above
elif NUM_CLUSTERS == 5:
    # 5 clusters is harder, so looser thresholds
    threshold_mean = 0.02  # threshold for mean difference, witin 2% of full set
    threshold_energy = 0.2  # based on tuning above
else:
    raise ValueError("Set thresholds for this number of clusters.")


# %%
def score_datasets_v0(
    datasets: dict,
    df_daily: pd.DataFrame,
    columns: list[str],
    probs: np.ndarray,
    threshold_mean: float,
    threshold_energy: float,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Score datasets based on univariate metrics: mean difference, energy distance, quantile differences.
    """

    ref = {c: df_daily[c].to_numpy() for c in columns}
    ref_q = {c: np.quantile(ref[c], probs) for c in columns}

    records, univariate_good_random_dfs, all_good_random_dfs, ds_skipped = (
        [],
        {},
        {},
        set(),
    )
    counters = {"mean_skips": 0, "energy_skips": 0}

    # TODO: PARALLELIZE THIS
    for ds_name, df_sub in tqdm(datasets.items()):
        random = False
        if ds_name in ds_skipped:
            continue
        if ds_name.startswith("Random"):
            random = True  # these can be discarded, the other ones not!
            # generate Random dataset on the fly, only has to be kept if it is good enough
            df_sub = build_df_random(df_idx, keys, rng, num_random=NUM_CLUSTERS)
        keep = True

        for var in columns:
            sub_arr = df_sub[var].to_numpy()
            full_arr = ref[var]

            # 1. mean difference
            d_mean = float(np.abs((sub_arr.mean() - full_arr.mean()) / full_arr.mean()))
            # check if score is satisfying, if not continue to next subset and don't bother with more sophisticated scores or other variables
            if d_mean > threshold_mean:
                counters["mean_skips"] += 1
                if random:
                    keep = False
                break

            # 2. energy score
            # TODO: make faster using approach of Jean. Don't understand it.
            energy_distance_score = energy_distance(sub_arr, full_arr)
            if energy_distance_score > threshold_energy:
                counters["energy_skips"] += 1
                if random:
                    keep = False
                break

            # 3. quantile differences
            q_sub = np.quantile(sub_arr, probs)
            qq = np.abs(q_sub - ref_q[var])

            records.append(
                {
                    "ds_name": ds_name,
                    "variable": var,
                    "d_mean": d_mean,
                    "energy_distance_score": energy_distance_score,
                    "qq_mae": float(qq.mean()),
                    "qq_mae_low": float(qq[:10].mean()),
                    "qq_mae_high": float(qq[-10:].mean()),
                }
            )

            if ds_name.startswith("Random"):
                univariate_good_random_dfs[ds_name] = df_sub

        if not keep:
            ds_skipped.add(ds_name)

    df_scores = pd.DataFrame.from_records(records)
    df_score_all_good = df_scores.query("ds_name not in @ds_skipped")
    ds_sufficient = df_score_all_good["ds_name"].unique().tolist()
    for ds_name, df in univariate_good_random_dfs.items():
        if ds_name in ds_sufficient:
            all_good_random_dfs[ds_name] = df
    return df_score_all_good, ds_sufficient, all_good_random_dfs, counters


# TODO: WORK IN PROGRESS
def score_df(args):
    ds_name, df_sub, ref, ref_q, columns, probs, threshold_mean, threshold_energy = args

    if ds_name in ds_skipped:
        return
    if ds_name.startswith("Random"):
        # generate Random dataset on the fly, only has to be kept if it is good enough
        df_sub = build_df_random(df_idx, keys, rng, num_random=NUM_CLUSTERS)
    keep = True

    for var in columns:
        sub_arr = df_sub[var].to_numpy()
        full_arr = ref[var]

        # 1. mean difference
        d_mean = float(np.abs((sub_arr.mean() - full_arr.mean()) / full_arr.mean()))
        # check if score is satisfying, if not continue to next subset and don't bother with more sophisticated scores or other variables
        if d_mean > threshold_mean:
            counters["mean_skips"] += 1
            keep = False
            break

        # 2. energy score
        # TODO: make faster using approach of Jean. Don't understand it.
        energy_distance_score = energy_distance(sub_arr, full_arr)
        if energy_distance_score > threshold_energy:
            counters["energy_skips"] += 1
            keep = False
            break

        # 3. quantile differences
        q_sub = np.quantile(sub_arr, probs)
        qq = np.abs(q_sub - ref_q[var])

        records.append(
            {
                "ds_name": ds_name,
                "variable": var,
                "d_mean": d_mean,
                "energy_distance_score": energy_distance_score,
                "qq_mae": float(qq.mean()),
                "qq_mae_low": float(qq[:10].mean()),
                "qq_mae_high": float(qq[-10:].mean()),
            }
        )

        if ds_name.startswith("Random"):
            univariate_good_random_dfs[ds_name] = df_sub

    if not keep:
        ds_skipped.add(ds_name)
    return records, univariate_good_random_dfs, ds_skipped, counters


def score_datasets(
    datasets, df_daily, columns, probs, threshold_mean, threshold_energy
):
    ref = {c: df_daily[c].to_numpy() for c in columns}
    ref_q = {c: np.quantile(ref[c], probs) for c in columns}

    all_records = []
    all_univariate = {}
    all_skipped = set()
    total_counters = {"mean_skips": 0, "energy_skips": 0}

    # prepare arguments for each dataset
    args_list = [
        (ds_name, df_sub, ref, ref_q, columns, probs, threshold_mean, threshold_energy)
        for ds_name, df_sub in datasets.items()
    ]

    with ProcessPoolExecutor() as exe:
        for recs, uni, skipped, counters in tqdm(
            exe.map(score_df, args_list), total=len(args_list)
        ):
            all_records.extend(recs)
            all_univariate.update(uni)
            all_skipped.update(skipped)
            total_counters["mean_skips"] += counters["mean_skips"]
            total_counters["energy_skips"] += counters["energy_skips"]

    df_scores = pd.DataFrame.from_records(all_records)
    df_score_all_good = df_scores.query("ds_name not in @all_skipped")
    ds_sufficient = df_score_all_good["ds_name"].unique().tolist()

    all_good_random_dfs = {
        ds_name: df
        for ds_name, df in all_univariate.items()
        if ds_name in ds_sufficient
    }

    return df_score_all_good, ds_sufficient, all_good_random_dfs, total_counters


# 4. Wasserstein distance for sufficient datasets on univariates
def wasserstein_for_sufficient(
    datasets: dict,
    ds_sufficient: list[str],
    df_daily: pd.DataFrame,
    columns: list[str],
    n_projections: int = 50,
) -> pd.DataFrame:
    """
    Compute multivariate sliced-Wasserstein distance for sufficient datasets.
    """
    ref_np = df_daily[columns].to_numpy()
    rows = []
    for ds_name in ds_sufficient:
        sub_np = datasets[ds_name][columns].to_numpy()
        swd = float(
            sliced_wasserstein_distance(ref_np, sub_np, n_projections=n_projections)
        )
        rows.append({"ds_name": ds_name, "Wasserstein": swd})
    return pd.DataFrame(rows)


# %%
# ---- Run --------------------------------------------------------------------

df_score_all_good, ds_sufficient, good_random_datasets, ctrs = score_datasets_v0(
    datasets=datasets,
    df_daily=df_daily,
    columns=vars_to_match,
    probs=PROBS,
    threshold_mean=threshold_mean,
    threshold_energy=threshold_energy,
)

final_datasets = datasets_optimized | good_random_datasets

print(
    f"Kept {len(ds_sufficient)} / {len(datasets)} datasets "
    f"(mean skips: {ctrs['mean_skips']}, energy skips: {ctrs['energy_skips']})."
)

# %%
# TODO: STILL REVIEW THIS PARALLEL VERSION, SEEMS TO WORK GOOD!

# --- worker globals (set once per process) ---
_REF_WASS: np.ndarray | None = None


def _init_wass(ref_np: np.ndarray) -> None:
    """Initializer to share the reference array once per process."""
    global _REF_WASS
    _REF_WASS = ref_np


def _swd_worker(payload: Tuple[str, np.ndarray, int]) -> Tuple[str, float]:
    """Compute sliced-Wasserstein for one dataset against global reference."""
    ds_name, sub_np, n_proj = payload
    swd = float(sliced_wasserstein_distance(_REF_WASS, sub_np, n_projections=n_proj))  # type: ignore[arg-type]
    return ds_name, swd


def wasserstein_for_sufficient_parallel(
    datasets: dict,
    ds_sufficient: list[str],
    df_daily: pd.DataFrame,
    columns: list[str],
    n_projections: int = 50,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Parallel sliced-Wasserstein over sufficient datasets (process-based).
    """
    ref_np = df_daily[columns].to_numpy()
    tasks: Iterable[Tuple[str, np.ndarray, int]] = (
        (ds_name, datasets[ds_name][columns].to_numpy(), n_projections)
        for ds_name in ds_sufficient
    )

    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_wass, initargs=(ref_np,)
    ) as ex:
        it = ex.map(_swd_worker, tasks, chunksize=1)
        for ds_name, swd in tqdm(it, total=len(ds_sufficient), desc="Wasserstein"):
            rows.append({"ds_name": ds_name, "Wasserstein": swd})

    return pd.DataFrame(rows)


df_wass_scores = wasserstein_for_sufficient_parallel(
    datasets=final_datasets,
    ds_sufficient=ds_sufficient,
    df_daily=df_daily,
    columns=vars_to_match,
    n_projections=200,
)


# %%

filename = (
    f"{ALL_MATCH_COUNTRIES}_{NUM_DRAWS}_draws_{NUM_CLUSTERS}_clusters_run_{RUN_ID}"
)
df_wass_scores.to_csv(f"{BASE_OUTPUT}/metrics_random_draws/wasserstein_{filename}.csv")
df_score_all_good.to_csv(
    f"{BASE_OUTPUT}/metrics_random_draws/univariate_scores_{filename}.csv"
)

with open(
    f"{BASE_OUTPUT}/metrics_random_draws/final_datasets_{filename}.pkl", "wb"
) as f:
    pickle.dump(final_datasets, f)

print("Done saving scores.")
