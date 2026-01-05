# %%
#### imports ####
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

# pylint: disable=import-error
from RUN_SETTINGS import (
    BATCH_SIZE,
    DF_PROD_DEMAND_FNAME,
    HBF_ITERATIONS,
    NUM_TO_SELECT,
    P_THRES,
    PARAM_GRID,
    RUN_HBF,
    RUN_SA,
    RUNNAME,
    SEEDS,
    TUNING_ITERATIONS,
    VARS_TO_MATCH,
    YEAR_COL,
)
from tqdm import tqdm

# pylint: enable=import-error
from cluster_weather_years.constants import BASE_OUTPUT, HYDROLOGICAL_YEARS, MODELS
from cluster_weather_years.helper_functions import draw_random_modelyears
from cluster_weather_years.hierarchical_brute_force import (
    #    build_random_df,
    init_worker,
    run_batch,
    #    score_df,
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

model_years = list(product(MODELS, HYDROLOGICAL_YEARS))
rng = np.random.default_rng(SEEDS[0])

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
output_dir_hbf = os.path.join(output_dir_run, "hierarchical_brute_force")

os.makedirs(output_dir_random, exist_ok=True)
os.makedirs(output_dir_kmedoids, exist_ok=True)
os.makedirs(output_dir_sa, exist_ok=True)
os.makedirs(output_dir_hbf, exist_ok=True)

# Copy RUN_SETTINGS.py to output dir for record keeping
shutil.copy("RUN_SETTINGS.py", os.path.join(output_dir_run, "RUN_SETTINGS_of_run.py"))

# %%
#### 2. Baseline Random Draws #####
print("##### 2. Performing Baseline Random Draws... #####")

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
if RUN_SA:
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
        futures = [
            ex.submit(run_sa, inputs=constant_inputs, **job) for job in param_set
        ]
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
else:
    df_ref = df_ref.reset_index()
    print("##### 4. Skipping Simulated Annealing as per settings... #####")

# %%
#### 5. Hierarchical Subset Brute-Force #####
# NOTE: scales very poorly with increasing vars_to_match
if RUN_HBF:
    print("##### 5. Performing Hierarchical Subset Brute-Force Search... #####")

    # TODO: make sure it's reproducible, e.g., by setting seeds properly
    thresholds_mean, thresholds_energy = score_tuning(
        df_ref,
        model_years,
        TUNING_ITERATIONS,
        NUM_TO_SELECT,
        vars_non_nan,
        rng,
        YEAR_COL,
        percentile_threshold=P_THRES,
    )

    df_thres = pd.DataFrame(
        thresholds_energy.items(), columns=["Variable", "Threshold Energy"]
    )
    df_thres_mean = pd.DataFrame(
        thresholds_mean.items(), columns=["Variable", "Threshold Mean"]
    )
    df_thres.merge(df_thres_mean, on="Variable").to_csv(
        os.path.join(output_dir_hbf, "hbf_thresholds.csv"), index=True
    )

    # %%

    n_batches = math.ceil(HBF_ITERATIONS / BATCH_SIZE)

    # deterministic per-batch seeds
    seeds = np.random.SeedSequence(SEEDS[0]).spawn(n_batches)
    seeds = [int(s.generate_state(1)[0]) for s in seeds]

    results_pass = []
    results_fail = []
    modelyears_pass = []

    max_workers = os.cpu_count()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(
            df_ref,
            model_years,
            thresholds_mean,
            thresholds_energy,
            NUM_TO_SELECT,
            YEAR_COL,
            vars_non_nan,
            True,  # skip_check
        ),
    ) as ex:
        it = ex.map(
            run_batch,
            range(n_batches),
            [BATCH_SIZE] * n_batches,
            [HBF_ITERATIONS] * n_batches,
            seeds,
        )

        for pass_rec, fail_rec, pass_years in tqdm(it, total=n_batches):
            results_pass.extend(pass_rec)
            results_fail.extend(fail_rec)
            modelyears_pass.extend(pass_years)

    df_pass = pd.DataFrame(results_pass).sort_values("wasserstein")
    df_fail = pd.DataFrame(results_fail)[["fail_check", "fail_var"]]

    skip_rate = len(df_fail) / HBF_ITERATIONS
    print(skip_rate)

    print(
        df_fail.groupby(["fail_check", "fail_var"])
        .size()
        .sort_values(ascending=False)
        .head(20)
    )

    df_pass.to_csv(os.path.join(output_dir_hbf, "hbf_df_pass.csv"), index=True)
# %%
else:
    print("##### 5. Skipping Hierarchical Brute-Force as per settings... #####")
# %%
