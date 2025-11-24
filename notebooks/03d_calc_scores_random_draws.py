# %%
import itertools
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from math import comb

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
    SEASONS,
)
from cluster_weather_years.functions import build_df_centroids, build_df_random

# %% SETTINGS
SAVE = True  # whether to save results
NUM_DRAWS = int(1e4)  # number of random draws to compare with, default = 1e5
N_BATCH = 1000  # batch size
NUM_CLUSTERS = 30  # Number of Climate Years to calculate
TUNE = False  # whether to run in tuning mode (for thresholds)
# ALL_MATCH_COUNTRIES = ["NL", "BE", "FR", "DE"]
ALL_MATCH_COUNTRIES = ["NL"]

# RUN_ID = "NL_20251024_1320"  # NL 30y; Run to analyze and compare to random draws
# RUN_ID = "NL_20251104_1108"  # NL 30y seasonal_wasserstein.
RUN_ID = "NL_20251112_1638"  # NL, 30y seasonal, incl. residual load
# RUN_ID = "NL_20251023_0942"  # NL 5y; Run to analyze and compare to random draws

# RUN_ID_alternative = "NL_BE_FR_DE_20251015_0801"  # Multi, 30y

print(f"Analyzing {NUM_DRAWS} random draws, save = {SAVE}.")

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

# for now match without residual load, maybe change later
vars_to_match = (
    [f"{country}_OFF_prod" for country in ALL_MATCH_COUNTRIES]
    + [f"{country}_ON_prod" for country in ALL_MATCH_COUNTRIES]
    + [f"{country}_SOLAR_prod" for country in ALL_MATCH_COUNTRIES]
    + [f"{country}_Demand" for country in ALL_MATCH_COUNTRIES]
    + [f"{country}_Residual_load" for country in ALL_MATCH_COUNTRIES]
)

MODEL_YEARS = list(itertools.product(MODELS, HYDROLOGICAL_YEARS))

# %%
df_selections = pd.read_csv(
    os.path.join(SA_OUTPUT_DIR, f"run_{RUN_ID}/results_grid.csv"), index_col=0
)
df_sel_multi_country = df_selections.query(
    "optimized_metric == 'wasserstein_seasonal_score'"
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
# initialize ds_items with seeds for random datasets
ss = np.random.SeedSequence(seed)
child_ss = ss.spawn(len(datasets))

ds_items = []
for (name, ds), child in zip(datasets.items(), child_ss):
    seed = int(child.generate_state(1)[0])
    ds_items.append((name, ds, seed))

# create batches of datasets, for efficient parallel processing
batches = [ds_items[i : i + N_BATCH] for i in range(0, len(ds_items), N_BATCH)]

# TODO: can probably all go in one dictionary
ref = {}
ref_mean = {}
ref_quantiles = {}
for var in columns:
    ref_arr = df_daily[var].to_numpy()
    ref[var] = ref_arr
    ref_mean[var] = ref_arr.mean()
    ref_quantiles[var] = np.quantile(ref_arr, PROBS)


# %%
def score_df(df_item, columns=vars_to_match):
    df_name, df_sub, local_seed = df_item
    skip_check = False  # keep all regular datasets, regardless of scores
    if "Random" in df_name:
        local_rng = np.random.default_rng(local_seed)
        df_sub = build_df_random(df_idx, keys, local_rng, num_random=NUM_CLUSTERS)
        skip_check = True

    d_means, energy_distance_score, qq_mae, qq_mae_high, qq_mae_low = [], [], [], [], []
    mean_skip, energy_skip = 0, 0

    for var in columns:
        sub_arr = df_sub[var].to_numpy()

        d_mean = float(abs((sub_arr.mean() - ref_mean[var]) / ref_mean[var]))
        if d_mean > threshold_mean and "Residual_load" not in var and skip_check:
            return {"ds_name": df_name, "mean_skip": 1, "energy_skip": 0}

        energy_score = float(energy_distance(sub_arr, ref[var]))
        if (
            energy_score > threshold_energy
            and "Residual_load" not in var
            and skip_check
        ):
            return {"ds_name": df_name, "mean_skip": 0, "energy_skip": 1}

        q_sub = np.quantile(sub_arr, PROBS)
        q_ref = ref_quantiles[var]
        qq_diff = abs(q_sub - q_ref)

        d_means.append(d_mean)
        energy_distance_score.append(energy_score)
        qq_mae.append(qq_diff.mean())
        qq_mae_high.append(float(qq_diff[:10].mean()))
        qq_mae_low.append(float(qq_diff[-10:].mean()))

    # compute multi-variate Wasserstein
    wasserstein = sliced_wasserstein_distance(
        df_sub[columns].to_numpy(), df_daily[columns].to_numpy(), n_projections=200
    )

    return {
        "ds_name": df_name,
        "ds": df_sub,
        "d_mean": d_means,
        "energy_distance_score": energy_distance_score,
        "qq_mae": qq_mae,
        "qq_mae_high": qq_mae_high,
        "qq_mae_low": qq_mae_low,
        "wasserstein": wasserstein,
        "mean_skip": mean_skip,
        "energy_skip": energy_skip,
    }


def batch_score(dfs_batch):
    """Score a batch of datasets."""
    results_batch = []
    for df_item in dfs_batch:
        res = score_df(df_item)
        results_batch.append(res)
    return results_batch


# %% Seasonal score calculations post-filtering


# TODO: Can be combined with the other scoring function above
def compute_univariate_metrics(
    sub_arr: np.ndarray, ref_arr: np.ndarray, probs: np.ndarray
) -> dict:
    """
    Compute univariate metrics between two 1-D arrays.
    """
    d_mean = float(np.abs((sub_arr.mean() - ref_arr.mean()) / ref_arr.mean()))
    energy_distance_score = float(energy_distance(sub_arr, ref_arr))
    q_sub = np.quantile(sub_arr, probs)
    q_ref = np.quantile(ref_arr, probs)
    qq = np.abs(q_sub - q_ref)
    return {
        "d_mean": d_mean,
        "energy_distance_score": energy_distance_score,
        "qq_mae": float(qq.mean()),
        "qq_mae_low": float(qq[:10].mean()),
        "qq_mae_high": float(qq[-10:].mean()),
    }


def score_datasets_seasonal(
    datasets: dict,
    df_daily: pd.DataFrame,
    columns: list[str],
    probs: np.ndarray,
) -> pd.DataFrame:
    """
    Score datasets (already filtered/good) on a seasonal basis.
    """

    records = []
    for ds_name, df_sub in datasets.items():
        for season_name, month_list in SEASONS.items():
            df_daily_season = df_daily.query("Date.dt.month in @month_list")
            df_sub_season = df_sub.query("Date.dt.month in @month_list")
            for var in columns:
                sub_arr = df_sub_season[var].to_numpy()
                ref_arr = df_daily_season[var].to_numpy()

                # compute metrics
                metrics = compute_univariate_metrics(sub_arr, ref_arr, probs)
                records.append(
                    {
                        "ds_name": ds_name,
                        "season": season_name,
                        "variable": var,
                        **metrics,
                    }
                )
    return pd.DataFrame.from_records(records)


def wasserstein_seasonal(
    datasets: dict,
    df_daily: pd.DataFrame,
    columns: list[str],
    probs: np.ndarray,
) -> pd.DataFrame:
    """
    Score datasets (already filtered/good) on a seasonal basis.
    """

    records = []
    for ds_name, df_sub in datasets.items():
        for season_name, month_list in SEASONS.items():
            full_data = df_daily.query("Date.dt.month in @month_list")[
                columns
            ].to_numpy()
            sub_data = df_sub.query("Date.dt.month in @month_list")[columns].to_numpy()
            wass = sliced_wasserstein_distance(sub_data, full_data, n_projections=50)

            records.append(
                {
                    "ds_name": ds_name,
                    "season": season_name,
                    "wasserstein": wass,
                }
            )
    return pd.DataFrame.from_records(records)


# %%
# parallel
if __name__ == "__main__":
    # ds_items = list(datasets.items())
    results = []

    print("Scoring datasets in parallel...")
    # simple parallel, with one task per dataset. Doesn't scale!
    # with ProcessPoolExecutor() as exe:
    #     results = list(tqdm(exe.map(score_df, ds_items), total=len(datasets)))

    with ProcessPoolExecutor() as exe:
        for res in tqdm(
            exe.map(batch_score, batches),
            total=len(batches),
        ):
            results.extend(res)

    # Count skips
    mean_skips = sum(r["mean_skip"] for r in results if r is not None)
    energy_skips = sum(r["energy_skip"] for r in results if r is not None)

    # Keep only successfully scored datasets
    valid_results = [
        r for r in results if r["mean_skip"] == 0 and r["energy_skip"] == 0
    ]

    # Split into two dictionaries
    ds_dict = {r["ds_name"]: r["ds"] for r in valid_results}
    metrics_dict = {
        r["ds_name"]: {
            "d_mean": r["d_mean"],
            "energy_distance_score": r["energy_distance_score"],
            "qq_mae": r["qq_mae"],
            "qq_mae_high": r["qq_mae_high"],
            "qq_mae_low": r["qq_mae_low"],
            "wasserstein": r["wasserstein"],
            "mean_skip": r["mean_skip"],
            "energy_skip": r["energy_skip"],
        }
        for r in valid_results
    }

    df_metrics = pd.DataFrame(metrics_dict).T
    print(f"Total datasets scored: {len(df_metrics)}")
    print(f"Mean skips: {mean_skips}, Energy skips: {energy_skips}")

    print("Calculating seasonal scores...")
    df_seasonal_scores = score_datasets_seasonal(
        datasets=ds_dict,
        df_daily=df_daily,
        columns=columns,
        probs=PROBS,
    )

    df_wass_seasonal = wasserstein_seasonal(
        datasets=ds_dict,
        df_daily=df_daily,
        columns=columns,
        probs=PROBS,
    )

    # save
    FILENAME = f"{ALL_MATCH_COUNTRIES}_{NUM_DRAWS}_draws_{NUM_CLUSTERS}_clusters_run_{RUN_ID}_v2"

    if SAVE:
        df_metrics.to_csv(f"{BASE_OUTPUT}/metrics_random_draws/metrics_{FILENAME}.csv")

        df_seasonal_scores.to_csv(
            f"{BASE_OUTPUT}/metrics_random_draws/seasonal_scores_{FILENAME}.csv"
        )

        df_wass_seasonal.to_csv(
            f"{BASE_OUTPUT}/metrics_random_draws/wasserstein_seasonal_{FILENAME}.csv"
        )

        with open(
            f"{BASE_OUTPUT}/metrics_random_draws/final_datasets_{FILENAME}.pkl", "wb"
        ) as f:
            pickle.dump(ds_dict, f)

# %%
