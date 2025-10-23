# %%
# %%
import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor
from math import comb
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ot import sliced_wasserstein_distance
from scipy.stats import energy_distance
from tqdm import tqdm

from cluster_weather_years.constants import (
    BASE_OUTPUT,
    HYDROLOGICAL_YEARS,
    MODELS,
)

# %% SETTINGS

NUM_DRAWS = int(1e5)  # number of random draws to compare with
NUM_CLUSTERS = 5  # Number of Climate Years to calculate
TUNE = False  # whether to run in tuning mode (for thresholds)
# ALL_MATCH_COUNTRIES = ["NL", "BE", "FR", "DE"]
ALL_MATCH_COUNTRIES = ["NL"]

RUN_ID = "NL_20251001_1220"  # NL 5y
RUN_ID_alternative = "NL_BE_FR_DE_20251015_0801"  # Multi, 30y

# %%
seed = 42
random.seed(seed)

# quantiles
dense_low = np.linspace(0.005, 0.05, 9, endpoint=False)
coarse = np.linspace(0.05, 0.95, 91)
dense_high = np.linspace(0.955, 0.995, 9)
PROBS = np.concatenate([dense_low, coarse, dense_high])

# build df
DF_PROD_DEMAND_FNAME = os.path.join(
    BASE_OUTPUT, "dataframes/df_prod_demand_daily_multi_country_WRONG_CAPACITIES.csv"
)
SA_OUTPUT_DIR = os.path.join(BASE_OUTPUT, "simulated_annealing")

df_daily = pd.read_csv(DF_PROD_DEMAND_FNAME, index_col=0)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.query("Hydrological_year > 2014 and Hydrological_year < 2044")


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


# %% [markdown]
# # 2. Functions


def build_df_centroids(df_daily, selected, mode="Hydrological_year"):
    """
    Build df with daily data of selected years
    """
    frames_centroids = []
    for i, (model, year) in enumerate(selected):
        # model_centroid, year_centroid = row["model"], row["Date"]
        if mode == "Hydrological_year":
            df_centroid = df_daily.query(
                f"model == '{model}' and Hydrological_year == {year}"
            )
        elif mode == "Calendar_year":
            df_centroid = df_daily.query(
                f"model == '{model}' and Calendar_year == {year}"
            )
        frames_centroids.append(df_centroid)

    return pd.concat(frames_centroids, ignore_index=True)


df_idx = df_daily.set_index(["model", "Hydrological_year"]).sort_index()
keys = df_idx.index.unique().to_numpy()
rng = np.random.default_rng(seed)


def build_df_random(
    df_idx: pd.DataFrame,
    keys: np.ndarray,
    rng: np.random.Generator,
    num_random: int,
) -> pd.DataFrame:
    sel = rng.choice(len(keys), size=num_random, replace=False)
    return df_idx.loc[keys[sel]].reset_index()


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

# %% [markdown]
# # 3. Metrics calculation

# %% [markdown]
# ## 3.1 Metrics calculation, version for tuning thresholds

# %%
threshold_mean_tune = 0.01  # threshold for mean difference, witin 1% of full set
threshold_energy_tune = 0.1  # was 0.05 based on p10 for energy distance on NL_OFF_prod, but was too strict for random. Might need some tuning

# %%
if TUNE:
    scores = []
    # keep track of energy scores, to tune threshold
    energy_scores = []

    # initialize
    mean_skips = 0
    energy_skips = 0
    ds_skipped = []

    columns = vars_to_match.copy()
    df_daily_np = df_daily[columns].to_numpy()

    # col = columns[0]

    # TODO: PUT IN FUNCTION
    # TODO: PARALLELIZE THIS
    for col in columns:
        col_full = df_daily[col]
        q_full = np.quantile(col_full, PROBS)

        for ds_name in datasets:
            if ds_name in ds_skipped:
                continue
            df_sub = datasets[ds_name]
            col_sub = df_sub[col]

            # 1. mean difference
            d_mean = np.abs((col_sub.mean() - col_full.mean()) / col_full.mean())

            # check if score is satisfying, if not continue to next subset
            if d_mean > threshold_mean_tune:
                ds_skipped.append(ds_name)
                mean_skips += 1
                continue

            # 2. energy score
            # TODO: make faster using approach of Jean. Don't understand it.
            energy_distance_score = energy_distance(col_sub, col_full)

            energy_scores.append(
                {
                    "ds_name": ds_name,
                    "variable": col,
                    "energy_distance_score": float(energy_distance_score),
                }
            )

            # arbitrary threshold, might be too tight
            if energy_distance_score > threshold_energy_tune:
                ds_skipped.append(ds_name)
                energy_skips += 1
                continue

            # 3. calc quantiles and quantile diff
            q_sub = np.quantile(col_sub, PROBS)
            qq_diff = np.abs(q_sub - q_full)
            qq_mae = np.mean(qq_diff)
            qq_mae_low = np.mean(qq_diff[:10])
            qq_mae_high = np.mean(qq_diff[-10:])

            # Save scores in list-of-dictionaries
            scores.append(
                {
                    "ds_name": ds_name,
                    "variable": col,
                    "d_mean": float(d_mean),
                    "energy_distance_score": float(energy_distance_score),
                    "qq_mae": float(qq_mae),
                    "qq_mae_low": float(qq_mae_low),
                    "qq_mae_high": float(qq_mae_high),
                }
            )

    # TODO: if a dataset is skipped for one variable, skip it for all variables
    # Convert to df
    df_scores = pd.DataFrame(scores)
    print(
        f"Kept {len(df_scores)} out of {len(datasets)} datasets after skipping {mean_skips} due to mean and {energy_skips} due to energy distance."
    )

    df_score_all_good = df_scores.query("ds_name not in @ds_skipped")

    df_energy = pd.DataFrame(energy_scores)

    # every univariate should be in top 20% of energy distance scores. So worst one of these sets threshold.
    plt.figure()
    for var in columns:
        df_energy[df_energy["variable"] == var]["energy_distance_score"].hist(label=var)
        print(
            var,
            df_energy[df_energy["variable"] == var]["energy_distance_score"]
            .quantile(0.2)
            .round(4),
        )
    plt.legend()

# %% [markdown]
# ## 3.2 Real calculation, with set threshold

# %%
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
def score_datasets(
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

    records, good_random_dfs, ds_skipped = [], {}, set()
    counters = {"mean_skips": 0, "energy_skips": 0}

    # TODO: PARALLELIZE THIS
    for ds_name, df_sub in tqdm(datasets.items()):
        if ds_name in ds_skipped:
            continue
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
                good_random_dfs[ds_name] = df_sub

        if not keep:
            ds_skipped.add(ds_name)

    df_scores = pd.DataFrame.from_records(records)
    df_score_all_good = df_scores.query("ds_name not in @ds_skipped")
    ds_sufficient = df_score_all_good["ds_name"].unique().tolist()
    return df_score_all_good, ds_sufficient, good_random_dfs, counters


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
# TODO: non-parallel version, clean up

df_score_all_good, ds_sufficient, good_random_datasets, ctrs = score_datasets(
    datasets=datasets,
    df_daily=df_daily,
    columns=vars_to_match,
    probs=PROBS,
    threshold_mean=threshold_mean,
    threshold_energy=threshold_energy,
)

final_datasets = datasets_optimized | good_random_datasets

# df_wass_scores = wasserstein_for_sufficient(
#     datasets=final_datasets,
#     ds_sufficient=ds_sufficient,
#     df_daily=df_daily,
#     columns=vars_to_match,
#     n_projections=50,
# )

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
    n_projections=100,
)


# %%
# TODO: FIX NAMES
# TODO: LATER ALSO SAVE DATASETS

filename = f"{ALL_MATCH_COUNTRIES}_{NUM_DRAWS}_draws_{NUM_CLUSTERS}_clusters"
df_wass_scores.to_csv(f"{BASE_OUTPUT}/metrics_random_draws/wasserstein_{filename}.csv")
df_score_all_good.to_csv(
    f"{BASE_OUTPUT}/metrics_random_draws/univariate_scores_{filename}.csv"
)

print("Done saving scores.")
