#### imports ####
import os
from datetime import datetime

import pandas as pd
from RUN_SETTINGS import DF_PROD_DEMAND_FNAME, NUM_TO_SELECT, RUNNAME, VARS_TO_MATCH

from cluster_weather_years.constants import BASE_OUTPUT, HYDROLOGICAL_YEARS
from cluster_weather_years.k_medoids import k_medoids_clustering

#### 0. Load data #####
print("##### 0. Load data #####")
df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.query("Hydrological_year in @HYDROLOGICAL_YEARS")
df_daily = df_daily.set_index(["model", "Date"])

vars_non_nan = [var for var in VARS_TO_MATCH if not df_daily[var].isna().all()]

#### 1. Create output directories #####
print("##### 1. Create output directories #####")
# HACK: simplified output dir creation for testing
# rundate = datetime.now().strftime("%Y%m%d_%H%M")
# output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}_{rundate}")
output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}")

output_dir_kmedoids = os.path.join(output_dir_run, "k_medoids")
os.makedirs(output_dir_kmedoids, exist_ok=True)

#### 2. Baseline Random Draws (same amount as SA) #####
print("##### 2. Baseline Random Draws #####")

#### 3. K-medoids #####
print("##### 3. K-Medoids Clustering #####")
# TODO: check results
k_medoids_clustering(df_daily, NUM_TO_SELECT, vars_non_nan, output_dir_kmedoids)

#### 3. Simulated Annealing #####

#### 4. Hierarchical Subset Brute-Force #####
