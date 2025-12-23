# %%
#### imports ####
import os
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from RUN_SETTINGS import (
    DF_PROD_DEMAND_FNAME,
    NUM_TO_SELECT,
    PARAM_GRID,
    RUNNAME,
    SEEDS,
    VARS_TO_MATCH,
)

from cluster_weather_years.constants import BASE_OUTPUT, HYDROLOGICAL_YEARS, MODELS
from cluster_weather_years.helper_functions import build_df_random
from cluster_weather_years.k_medoids import k_medoids_clustering

# %%
#### 0. Load data #####
print("##### 0. Load data #####")
df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.query("Hydrological_year in @HYDROLOGICAL_YEARS")
df_daily = df_daily.set_index(["model", "Date"])

vars_non_nan = [var for var in VARS_TO_MATCH if not df_daily[var].isna().all()]
# %%
#### 1. Create output directories #####
print("##### 1. Create output directories #####")
# HACK: simplified output dir creation for testing
# rundate = datetime.now().strftime("%Y%m%d_%H%M")
# output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}_{rundate}")
output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}")

output_dir_random = os.path.join(output_dir_run, "random_draws")
output_dir_kmedoids = os.path.join(output_dir_run, "k_medoids")

os.makedirs(output_dir_random, exist_ok=True)
os.makedirs(output_dir_kmedoids, exist_ok=True)

# %%
#### 2. Baseline Random Draws #####
print("##### 2. Baseline Random Draws #####")

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
baseline_random_draws_df.to_parquet(
    os.path.join(output_dir_random, "baseline_random_draws.parquet")
)

# %%
#### 3. K-medoids #####
print("##### 3. K-Medoids Clustering #####")
# TODO: check results
k_medoids_clustering(df_daily, NUM_TO_SELECT, vars_non_nan, output_dir_kmedoids)

#### 3. Simulated Annealing #####

#### 4. Hierarchical Subset Brute-Force #####
