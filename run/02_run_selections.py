# %%
#### imports ####
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

# pylint: disable=import-error
from RUN_SETTINGS import (
    DF_PROD_DEMAND_FNAME,
    NUM_TO_SELECT,
    PARAM_GRID,
    RUNNAME,
    SEEDS,
    VARS_TO_MATCH,
    YEAR_COL,
)

# pylint: enable=import-error
from cluster_weather_years.constants import BASE_OUTPUT, HYDROLOGICAL_YEARS, MODELS

# from cluster_weather_years.helper_functions import build_df_random
from cluster_weather_years.k_medoids import k_medoids_clustering
from cluster_weather_years.sa import _sa_job

# %%
#### 0. Load data #####
print("##### 0. Loading data... #####")

df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.query("Hydrological_year in @HYDROLOGICAL_YEARS")
df_daily = df_daily.set_index(["model", "Date"])

vars_non_nan = [var for var in VARS_TO_MATCH if not df_daily[var].isna().all()]
# %%
#### 1. Create output directories #####
print("##### 1. Creating output directories... #####")

# HACK: simplified output dir creation for testing
# rundate = datetime.now().strftime("%Y%m%d_%H%M")
# output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}_{rundate}")
output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}")

output_dir_random = os.path.join(output_dir_run, "random_draws")
output_dir_kmedoids = os.path.join(output_dir_run, "k_medoids")
output_dir_sa = os.path.join(output_dir_run, "simulated_annealing")

os.makedirs(output_dir_random, exist_ok=True)
os.makedirs(output_dir_kmedoids, exist_ok=True)
os.makedirs(output_dir_sa, exist_ok=True)

# %%
#### 2. Baseline Random Draws #####
print("##### 2. Performing Baseline Random Draws... #####")

model_years = list(product(MODELS, HYDROLOGICAL_YEARS))
rng = np.random.default_rng(SEEDS[0])
# Use same amount of jobs as SA
num_jobs = len(list(product(SEEDS, product(*PARAM_GRID.values()))))

baseline_random_draws = []

for job_id in range(num_jobs):
    selected_idx = rng.choice(
        len(model_years),
        size=NUM_TO_SELECT,
        replace=False,
    )
    baseline_random_draws.append(tuple(model_years[i] for i in selected_idx))

baseline_random_draws_df = pd.DataFrame({"model_years": baseline_random_draws})
baseline_random_draws_df.to_csv(
    os.path.join(output_dir_random, "baseline_random_draws.csv")
)

print(f"Baseline random draws completed. Results saved to {output_dir_random}.")
# %%
#### 3. K-medoids #####
print("##### 3. Performing K-Medoids Clustering... #####")

# TODO: check results
k_medoids_clustering(df_daily, NUM_TO_SELECT, vars_non_nan, output_dir_kmedoids)

# %%
#### 4. Simulated Annealing #####
print("##### 4. Performing Simulated Annealing... #####")
# TODO: context should go out of there, refactor sa module. Also too much stuff here still
# TODO: Make everything work with consistent indexing
df_daily = df_daily.reset_index()
context = {}
constant_inputs = [
    df_daily,
    VARS_TO_MATCH,
    NUM_TO_SELECT,
    context,
    output_dir_sa,
    YEAR_COL,
]

jobs = [
    (seed, *vals, constant_inputs)
    for seed, vals in product(SEEDS, product(*PARAM_GRID.values()))
]
results = []

n_workers = min(os.cpu_count() or 1, len(jobs))

with ProcessPoolExecutor(max_workers=n_workers) as ex:
    futures = [ex.submit(_sa_job, j) for j in jobs]
    for i, fut in enumerate(as_completed(futures), start=1):
        sel_dict = fut.result()
        results.append(sel_dict)
        print(
            f"Experiment {i} done: {sel_dict['optimized_metric']} = {sel_dict['best_score']:.2f}"
        )

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_dir_sa, "results_grid.csv"))
best = min(results, key=lambda d: d["best_score"])
print(
    f"Run done, best score: {best['best_score']:.2f}, best selection: {best['selected']}"
)

# %%
#### 4. Hierarchical Subset Brute-Force #####
