"""
Select representative climate years with Simulated Annealing (SA) only.

This is the recommended script for new applications. It runs the same SA set-up as
02a_run_all_selection_methods.py (same seeds and cooling rates, so identical SA results
for the same RUN_SETTINGS.py), but skips the other selection methods that were only
needed for the comparison in the paper.

Inputs:  RUN_SETTINGS.py (general + SA settings) and the dataframe built in 01.
Outputs (in BASE_OUTPUT/<RUNNAME>_<timestamp>/):
    best_selection.csv          the selected (model, year) pairs
    simulated_annealing/
        sa_year_selections.csv          raw result per SA experiment
        sa_year_selections_scored.csv   all experiments with evaluation scores
        plots/                          SA convergence plot per experiment
    run_log.txt, RUN_SETTINGS_of_run.py

Run from the run/ directory:
    python 02b_run_simulated_annealing.py
"""

# %%
#### imports ####
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

# pylint: disable=import-error
from RUN_SETTINGS import (
    DF_PROD_DEMAND_FNAME,
    HYDRO,
    MATCH_COUNTRIES,
    N_PROJECTIONS_FULL,
    N_PROJECTIONS_INIT,
    NUM_EXPERIMENTS,
    NUM_TO_SELECT,
    PARAM_GRID,
    PERFORM_TWO_STEP,
    RUNNAME,
    SEASONAL,
    SEED_OFFSET,
    VARS_TO_MATCH,
    YEAR_COL,
)

from climate_year_selection.constants import BASE_OUTPUT, HYDROLOGICAL_YEARS
from climate_year_selection.cost_functions import (
    wasserstein_score,
    wasserstein_seasonal_score,
)
from climate_year_selection.evaluation import perform_physical_evaluation
from climate_year_selection.filtered_random_search import score_df, score_df_seasonal
from climate_year_selection.helper_functions import build_df_from_modelyears
from climate_year_selection.sa import run_sa

# pylint: enable=import-error

# Set True to also make diagnostic plots per country (monthly means, correlations,
# renewable droughts) of the SA selections against the reference dataset.
RUN_PHYSICAL_EVALUATION = False

# Projections for the refinement pass, or None to skip it (see RUN_SETTINGS.py)
N_PROJECTIONS_REFINE = N_PROJECTIONS_FULL if PERFORM_TWO_STEP else None
SCORE_COL = "wasserstein_seasonal" if SEASONAL else "wasserstein"


def setup_output_directories():
    """Create the output directories and set up logging to file and console."""
    rundate = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}_{rundate}")
    output_dir_sa = os.path.join(output_dir_run, "simulated_annealing")
    output_dir_eval_country = os.path.join(output_dir_run, "evaluation", "by_country")
    os.makedirs(output_dir_sa, exist_ok=True)
    if RUN_PHYSICAL_EVALUATION:
        os.makedirs(output_dir_eval_country, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    log_file = os.path.join(output_dir_run, "run_log.txt")
    for handler in (
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Copy RUN_SETTINGS.py to output dir for record keeping
    run_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy(
        os.path.join(run_dir, "RUN_SETTINGS.py"),
        os.path.join(output_dir_run, "RUN_SETTINGS_of_run.py"),
    )

    return output_dir_run, output_dir_sa, output_dir_eval_country


def load_data():
    """Load the reference dataset (all models, all HYDROLOGICAL_YEARS)."""
    df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])
    df_daily = df_daily.query("Hydrological_year in @HYDROLOGICAL_YEARS")

    # HACK: Hydro contains 0 instead of nans, so there have to be non-zero values.
    vars_non_nan = [var for var in VARS_TO_MATCH if np.max(df_daily[var]) > 0]

    df_ref = df_daily[
        ["model", "Date", "Hydrological_year", "Calendar_year"] + VARS_TO_MATCH
    ].reset_index(drop=True)
    return df_ref, vars_non_nan


def run_simulated_annealing(df_ref, vars_non_nan, output_dir_sa):
    """Run NUM_EXPERIMENTS independent SA experiments in parallel, one seed each."""
    logger = logging.getLogger(__name__)
    sa_shared_inputs = {
        "df_daily": df_ref,
        "vars_to_match": vars_non_nan,
        "NUM_CLUSTERS": NUM_TO_SELECT,
        "output_dir_sa": output_dir_sa,
        "YEAR_COL": YEAR_COL,
        "n_projections_init": N_PROJECTIONS_INIT,
        "n_projections_full": N_PROJECTIONS_REFINE,
    }
    # Cooling rates are spread evenly over the experiments (same as in 02a)
    cooling_rates = PARAM_GRID["cooling_rate"]
    param_set = [
        {
            "seed": i + SEED_OFFSET,
            "metric": wasserstein_seasonal_score if SEASONAL else wasserstein_score,
            "t0": PARAM_GRID["temp_start"][0],
            "cr": cooling_rates[i * len(cooling_rates) // NUM_EXPERIMENTS],
            "mi": PARAM_GRID["max_iter"][0],
        }
        for i in range(NUM_EXPERIMENTS)
    ]

    sa_result_dicts = []
    n_workers = min(os.cpu_count() or 1, len(param_set))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(run_sa, inputs=sa_shared_inputs, **job) for job in param_set
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            sa_result = fut.result()
            sa_result_dicts.append(sa_result)
            logger.info(
                f"Experiment {i}/{NUM_EXPERIMENTS} done: "
                f"{sa_result['optimized_metric']} = {sa_result['best_score']:.2f}"
            )

    df_sa_results = pd.DataFrame(sa_result_dicts)
    df_sa_results.to_csv(os.path.join(output_dir_sa, "sa_year_selections.csv"))
    return df_sa_results


def score_sa_selections(df_ref, vars_non_nan, df_sa_results, output_dir_sa):
    """Re-score every SA selection with the full set of evaluation metrics.

    SA itself may have used a cheap score (N_PROJECTIONS_INIT projections). Here all
    selections are scored with the same settings as in 02a, so that the final ranking
    is consistent with the paper workflow.
    """
    score_fn = score_df_seasonal if SEASONAL else score_df
    scored_rows = []
    for _, row in df_sa_results.iterrows():
        sel_modelyears = row["model_years"]
        df_sub = build_df_from_modelyears(df_ref, sel_modelyears, YEAR_COL)
        scores = score_fn(
            df_ref,
            df_sub,
            sel_modelyears,
            None,
            None,
            None,
            vars_non_nan,
            skip_check=False,
        )[1]
        score_row = {"selection_source": "simulated_annealing", "seed": row["seed"]}
        score_row.update(scores)
        scored_rows.append(score_row)

    df_scored = pd.DataFrame(scored_rows).sort_values(SCORE_COL).reset_index(drop=True)
    df_scored.to_csv(os.path.join(output_dir_sa, "sa_year_selections_scored.csv"))
    return df_scored


def main():
    """Run the SA-only selection workflow."""
    output_dir_run, output_dir_sa, output_dir_eval_country = setup_output_directories()
    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    logger.info(
        "##### Run settings #####\n"
        f"  MATCH_COUNTRIES    : {MATCH_COUNTRIES}\n"
        f"  NUM_TO_SELECT      : {NUM_TO_SELECT}\n"
        f"  HYDRO              : {HYDRO}\n"
        f"  SEASONAL           : {SEASONAL}\n"
        f"  NUM_EXPERIMENTS    : {NUM_EXPERIMENTS}\n"
        f"  SA PARAM_GRID      : {PARAM_GRID}"
    )

    logger.info("##### 1. Loading data... #####")
    df_ref, vars_non_nan = load_data()

    logger.info(
        f"##### 2. Running Simulated Annealing in "
        f"{'seasonal' if SEASONAL else 'annual'} mode... #####"
    )
    df_sa_results = run_simulated_annealing(df_ref, vars_non_nan, output_dir_sa)

    logger.info("##### 3. Scoring SA selections... #####")
    df_scored = score_sa_selections(df_ref, vars_non_nan, df_sa_results, output_dir_sa)

    # Best selection = lowest score over all experiments
    best = df_scored.iloc[0]
    df_best = pd.DataFrame(sorted(best["model_years"]), columns=["model", YEAR_COL])
    df_best.to_csv(os.path.join(output_dir_run, "best_selection.csv"), index=False)
    logger.info(
        f"Best selection (seed {best['seed']}, {SCORE_COL} = {best[SCORE_COL]:.3f}):\n"
        f"{df_best.to_string(index=False)}"
    )

    if RUN_PHYSICAL_EVALUATION:
        logger.info("##### 4. Physical evaluation per country... #####")
        perform_physical_evaluation(
            df_scored, df_ref, MATCH_COUNTRIES, vars_non_nan, output_dir_eval_country
        )

    logger.info(f"Results saved to {output_dir_run}")
    logger.info(f"Total run time: {datetime.now() - start_time}")


# %%
if __name__ == "__main__":
    main()
