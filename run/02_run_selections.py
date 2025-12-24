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
from tqdm import tqdm

# pylint: enable=import-error
from cluster_weather_years.constants import BASE_OUTPUT, HYDROLOGICAL_YEARS, MODELS
from cluster_weather_years.helper_functions import draw_random_modelyears
from cluster_weather_years.hierarchical_brute_force import (
    build_random_df,
    score_df,
    score_tuning,
)
from cluster_weather_years.k_medoids import k_medoids_clustering
from cluster_weather_years.sa import run_sa

# %%
#### 0. Load data #####
print("##### 0. Loading data... #####")

df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.query("Hydrological_year in @HYDROLOGICAL_YEARS")
df_daily = df_daily.set_index(["model", "Date"])

vars_non_nan = [var for var in VARS_TO_MATCH if not df_daily[var].isna().all()]
df_ref = df_daily[["Hydrological_year", "Calendar_year"] + VARS_TO_MATCH]

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

baseline_random_draws = draw_random_modelyears(
    model_years, num_jobs, NUM_TO_SELECT, rng
)

baseline_random_draws_df = pd.DataFrame({"model_years": baseline_random_draws})
baseline_random_draws_df.to_csv(
    os.path.join(output_dir_random, "baseline_random_draw_year_selections.csv")
)

print(f"Baseline random draws completed. Results saved to {output_dir_random}.")
# %%
#### 3. K-medoids #####
print("##### 3. Performing K-Medoids Clustering... #####")

# TODO: check results
k_medoids_clustering(df_ref, NUM_TO_SELECT, vars_non_nan, output_dir_kmedoids)

# %%
#### 4. Simulated Annealing #####
print("##### 4. Performing Simulated Annealing... #####")
# TODO: Make everything work with consistent indexing of df
df_ref = df_ref.reset_index()

# build jobs
constant_inputs = {
    "df_daily": df_ref,
    "vars_to_match": vars_non_nan,
    "NUM_CLUSTERS": NUM_TO_SELECT,
    "output_dir_sa": output_dir_sa,
    "YEAR_COL": YEAR_COL,
}
param_set = [
    {
        "seed": seed,
        "metric": metric,
        "t0": t0,
        "cr": cr,
        "mi": mi,
    }
    for seed, (metric, t0, cr, mi) in product(SEEDS, product(*PARAM_GRID.values()))
]

# run jobs in parallel
n_workers = min(os.cpu_count() or 1, len(param_set))

results = []
with ProcessPoolExecutor(max_workers=n_workers) as ex:
    futures = [ex.submit(run_sa, inputs=constant_inputs, **job) for job in param_set]
    for i, fut in enumerate(as_completed(futures), start=1):
        sel_dict = fut.result()
        results.append(sel_dict)
        print(
            f"Experiment {i} done: {sel_dict['optimized_metric']} = {sel_dict['best_score']:.2f}"
        )

# save results
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_dir_sa, "sa_year_selections.csv"))
best = min(results, key=lambda d: d["best_score"])
print(
    f"SA run done, best score: {best['best_score']:.2f}, best selection: {best['selected']}"
)

# %%
#### 5. Hierarchical Subset Brute-Force #####
# NOTE: scales very poorly with increasing vars_to_match
print("##### 5. Performing Hierarchical Subset Brute-Force Search... #####")

# define constants for brute-force search
p = 50  # percentile threshold
tuning_iterations = int(1e2)
iterations = int(1e4)

thresholds_mean, thresholds_energy = score_tuning(
    df_ref,
    model_years,
    tuning_iterations,
    NUM_TO_SELECT,
    vars_non_nan,
    rng,
    YEAR_COL,
    percentile_threshold=p,
)

# %%
results_pass = []
results_fail = []
modelyears_pass = []

for i in tqdm(range(iterations)):
    df_random, drawn_model_years = build_random_df(
        df_ref, model_years, NUM_TO_SELECT, rng, YEAR_COL
    )

    passed, record = score_df(
        df_ref,
        df_random,
        drawn_model_years,
        i,
        thresholds_mean,
        thresholds_energy,
        columns=vars_non_nan,
        skip_check=True,
    )

    if passed:
        results_pass.append(record)
        modelyears_pass.append(drawn_model_years)
    else:
        results_fail.append(record)

df_pass = pd.DataFrame(results_pass)
df_fail = pd.DataFrame(results_fail)

skip_rate = len(df_fail) / iterations
print(skip_rate)

# quick breakdown of failure reasons
print(
    df_fail.groupby(["fail_check", "fail_var"])
    .size()
    .sort_values(ascending=False)
    .head(20)
)

# %%
