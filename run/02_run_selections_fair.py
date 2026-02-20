# %%
#### imports ####
import logging
import math
import os
import shutil
from ast import literal_eval
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
    HYDRO,
    MATCH_COUNTRIES,
    NUM_TO_SELECT,
    P_THRES,
    PARAM_GRID,
    RUN_HBF,
    RUN_SA,
    RUNNAME,
    RUNNAME_SA_FALLBACK,
    SEASONAL,
    SEEDS,
    TUNING_ITERATIONS,
    VARS_TO_MATCH,
    YEAR_COL,
)
from tqdm import tqdm

from cluster_weather_years.constants import BASE_OUTPUT, HYDROLOGICAL_YEARS, MODELS
from cluster_weather_years.evaluation import (
    perform_physical_evaluation,
    plot_lollipop_by_country,
    plot_results_with_pareto,
)
from cluster_weather_years.helper_functions import (
    build_df_from_modelyears,
    draw_random_modelyears,
)
from cluster_weather_years.hierarchical_brute_force import (
    init_worker,
    run_batch,
    score_df,
    score_df_seasonal,
    score_tuning,
    score_tuning_seasonal,
)
from cluster_weather_years.k_medoids import k_medoids_clustering
from cluster_weather_years.sa import run_sa
from cluster_weather_years.scores import wasserstein_score, wasserstein_seasonal_score

# pylint: enable=import-error


def _parse_model_years(df: pd.DataFrame) -> pd.DataFrame:
    """Parse model_years column from CSV string repr back to tuples after read_csv."""
    if "model_years" not in df.columns:
        return df
    df = df.copy()
    df["model_years"] = df["model_years"].apply(
        lambda x: (
            tuple(tuple(my) for my in literal_eval(x))
            if isinstance(x, str)
            else tuple(tuple(my) for my in x)
        )
    )
    return df


def setup_output_directories():
    """0. Create output directories."""
    rundate = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}_{rundate}")
    # output_dir_run = os.path.join(BASE_OUTPUT, f"{RUNNAME}")

    os.makedirs(output_dir_run, exist_ok=True)

    # Setup logging to file and console
    log_file = os.path.join(output_dir_run, "run_log.txt")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("##### 0. Creating output directories... #####")

    output_dir_random = os.path.join(output_dir_run, "random_draws")
    output_dir_kmedoids = os.path.join(output_dir_run, "k_medoids")
    output_dir_sa = os.path.join(output_dir_run, "simulated_annealing")
    output_dir_hbf = os.path.join(output_dir_run, "hierarchical_brute_force")
    output_dir_eval = os.path.join(output_dir_run, "evaluation")
    output_dir_eval_country = os.path.join(output_dir_eval, "by_country")

    os.makedirs(output_dir_random, exist_ok=True)
    os.makedirs(output_dir_kmedoids, exist_ok=True)
    os.makedirs(output_dir_sa, exist_ok=True)
    os.makedirs(output_dir_hbf, exist_ok=True)
    os.makedirs(output_dir_eval, exist_ok=True)
    os.makedirs(output_dir_eval_country, exist_ok=True)

    # Copy RUN_SETTINGS.py to output dir for record keeping
    shutil.copy(
        "RUN_SETTINGS.py", os.path.join(output_dir_run, "RUN_SETTINGS_of_run.py")
    )

    return (
        output_dir_run,
        output_dir_random,
        output_dir_kmedoids,
        output_dir_sa,
        output_dir_hbf,
        output_dir_eval,
        output_dir_eval_country,
    )


def load_data():
    """1. Load and prepare data."""
    logger = logging.getLogger(__name__)
    logger.info("##### 1. Loading data... #####")

    df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])
    if HYDRO:
        # HACK: overwrite default Hydrological years
        # Hydro data starts only in 2015-07-20
        HYDROLOGICAL_YEARS = range(2016, 2044)
    else:
        # HACK: Should be taken from constants, but not found?
        HYDROLOGICAL_YEARS = range(2015, 2044)
    df_daily = df_daily.query("Hydrological_year in @HYDROLOGICAL_YEARS")
    df_daily = df_daily.set_index(["model", "Date"])

    # HACK: Hydro contains 0 instead of nans, so there have to be non-zero values.
    vars_non_nan = [var for var in VARS_TO_MATCH if np.max(df_daily[var]) > 0]

    df_ref = df_daily[
        ["Hydrological_year", "Calendar_year"] + VARS_TO_MATCH
    ].reset_index()

    model_years = list(product(MODELS, HYDROLOGICAL_YEARS))
    rng = np.random.default_rng(SEEDS[0])

    return df_ref, vars_non_nan, model_years, rng


# TODO: Not used anymore, can be deleted if new one works properly
def run_baseline_random_draws_old(model_years, rng, output_dir_random):
    """2. Baseline Random Draws."""
    logger = logging.getLogger(__name__)
    logger.info("##### 2. Performing Baseline Random Draws... #####")

    # Use same amount of jobs as SA
    num_jobs = len(list(product(SEEDS, product(*PARAM_GRID.values()))))

    baseline_random_draws = draw_random_modelyears(
        model_years, num_jobs, NUM_TO_SELECT, rng
    )

    baseline_random_draws_df = pd.DataFrame({"model_years": baseline_random_draws})
    baseline_random_draws_df.to_csv(
        os.path.join(output_dir_random, "baseline_random_draw_year_selections.csv")
    )

    logger.info(
        f"Baseline random draws completed. Results saved to {output_dir_random}."
    )
    return baseline_random_draws_df


def _draw_random_batch(args):
    """Draw and score a batch of random year selections (worker function)."""
    seed, model_years, batch_size, num_to_select, vars_non_nan = args
    rng = np.random.default_rng(seed)
    draws = draw_random_modelyears(model_years, batch_size, num_to_select, rng)
    results = []
    for model_years_draw in tqdm(draws, total=len(draws)):
        score = (
            wasserstein_seasonal_score(
                df=df_ref,
                combos=model_years_draw,
                variables=vars_non_nan,
                year_col=YEAR_COL,
                n_projections=50,
            )
            if SEASONAL
            else wasserstein_score(
                df=df_ref,
                combos=model_years_draw,
                variables=vars_non_nan,
                year_col=YEAR_COL,
                n_projections=50,
            )
        )
        score_key = "wasserstein_seasonal" if SEASONAL else "wasserstein"
        results.append({"model_years": model_years_draw, score_key: score})
    return results


def run_baseline_random_draws(
    model_years, vars_non_nan, df_sa_results, output_dir_random
):
    """3. Baseline Random Draws."""
    logger = logging.getLogger(__name__)
    logger.info("##### 3. Performing Baseline Random Draws... #####")

    num_jobs = int(df_sa_results["n_iter"].sum())
    logger.info(
        f"Drawing and scoring {num_jobs} random selections (= sum of SA n_iter) in mode {'seasonal' if SEASONAL else 'annual'}."
    )

    n_workers = min(os.cpu_count() or 1, num_jobs)
    batch_size = math.ceil(num_jobs / n_workers)

    # deterministic per-batch seeds, matching HBF style
    seeds = np.random.SeedSequence(SEEDS[0]).spawn(n_workers)
    seeds = [int(s.generate_state(1)[0]) for s in seeds]

    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        job_args = [
            (seed, model_years, batch_size, NUM_TO_SELECT, vars_non_nan)
            for seed in seeds
        ]
        futures = {
            ex.submit(_draw_random_batch, args): i for i, args in enumerate(job_args)
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            batch = fut.result()
            all_results.extend(batch)

    # keep only best len(SEEDS) like in SA by score
    sort_col = "wasserstein_seasonal" if SEASONAL else "wasserstein"
    baseline_random_draws_df = (
        pd.DataFrame(all_results)
        .sort_values(sort_col)
        .head(len(SEEDS))
        .reset_index(drop=True)
    )
    baseline_random_draws_df["selection_source"] = "random"
    baseline_random_draws_df.to_csv(
        os.path.join(output_dir_random, "baseline_random_draw_year_selections.csv")
    )

    logger.info(
        f"Baseline random draws completed. Best {len(SEEDS)} of {num_jobs} saved to {output_dir_random}."
    )
    return baseline_random_draws_df


def run_kmedoids(df_ref, vars_non_nan, output_dir_kmedoids, SEASONAL):
    """4. K-medoids clustering."""
    logger = logging.getLogger(__name__)
    logger.info(
        f"##### 4. Performing K-Medoids Clustering in {'seasonal' if SEASONAL else 'annual'} mode... #####"
    )

    k_medoids_clustering(
        df_ref, NUM_TO_SELECT, vars_non_nan, output_dir_kmedoids, SEASONAL=SEASONAL
    )
    logger.info(
        f"K-Medoids clustering completed. Results saved to {output_dir_kmedoids}."
    )


def run_simulated_annealing(df_ref, vars_non_nan, output_dir_sa):
    """2. Simulated Annealing (or load fallback results)."""
    logger = logging.getLogger(__name__)
    if RUN_SA:
        logger.info(
            f"##### 2. Performing Simulated Annealing in {'seasonal' if SEASONAL else 'annual'} mode... #####"
        )

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
            for seed, (metric, t0, cr, mi) in product(
                SEEDS, product(*PARAM_GRID.values())
            )
        ]

        # run jobs in parallel
        results = []
        n_workers = min(os.cpu_count() or 1, len(param_set))

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(run_sa, inputs=constant_inputs, **job) for job in param_set
            ]
            for i, fut in enumerate(as_completed(futures), start=1):
                sel_dict = fut.result()
                results.append(sel_dict)
                logger.info(
                    f"Experiment {i} done: {sel_dict['optimized_metric']} = {sel_dict['best_score']:.2f}"
                )

        # save results
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(output_dir_sa, "sa_year_selections.csv"))
        best = min(results, key=lambda d: d["best_score"])
        logger.info(f"SA run done, best score: {best['best_score']:.2f}")
    else:
        logger.info("##### 2. Skipping Simulated Annealing as per settings... #####")

        n_workers = os.cpu_count()
        df_results = _parse_model_years(
            pd.read_csv(
                os.path.join(
                    BASE_OUTPUT,
                    f"{RUNNAME_SA_FALLBACK}/simulated_annealing/sa_year_selections.csv",
                ),
                index_col=0,
            )
        )
        logger.info(
            f"Simulated Annealing results copied from fallback run: {RUNNAME_SA_FALLBACK}"
        )
        df_results.to_csv(
            os.path.join(output_dir_sa, "sa_year_selections.csv"), index=True
        )
    logger.info(f"Simulated Annealing results saved to {output_dir_sa}.")

    return df_results, n_workers


def run_hierarchical_brute_force(
    df_ref,
    vars_non_nan,
    model_years,
    rng,
    output_dir_hbf,
):
    """5. Hierarchical Subset Brute-Force."""
    # NOTE: scales very poorly with increasing vars_to_match
    logger = logging.getLogger(__name__)
    if RUN_HBF:
        logger.info(
            "##### 5. Performing Hierarchical Subset Brute-Force Search... #####"
        )
        logger.info(
            f"Running HBF with {'seasonal scoring' if SEASONAL else 'regular scoring'}..."
        )

        # TODO: make sure it's reproducible, e.g., by setting seeds properly
        if SEASONAL:
            d_means, energy_scores, thresholds_mean, thresholds_energy = (
                score_tuning_seasonal(
                    df_ref,
                    model_years,
                    TUNING_ITERATIONS,
                    NUM_TO_SELECT,
                    vars_non_nan,
                    rng,
                    YEAR_COL,
                    percentile_threshold=P_THRES,
                )
            )
        else:
            d_means, energy_scores, thresholds_mean, thresholds_energy = score_tuning(
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
                [SEASONAL] * n_batches,
            )

            for pass_rec, fail_rec, pass_years in tqdm(it, total=n_batches):
                results_pass.extend(pass_rec)
                results_fail.extend(fail_rec)
                modelyears_pass.extend(pass_years)

        df_pass = pd.DataFrame(results_pass).sort_values("wasserstein")
        df_pass["selection_source"] = "HBF"
        df_fail = pd.DataFrame(results_fail)[["fail_check", "fail_var"]]

        skip_rate = len(df_fail) / HBF_ITERATIONS
        logger.info(f"Skip rate: {skip_rate}")

        logger.info(
            f"Failure distribution:\n{df_fail.groupby(['fail_check', 'fail_var']).size().sort_values(ascending=False).head(20)}"
        )

        df_pass.to_csv(os.path.join(output_dir_hbf, "hbf_df_pass.csv"), index=True)
        return df_pass
    else:
        logger.info(
            "##### 5. Skipping Hierarchical Brute-Force as per settings... #####"
        )
        return None


def score_all_selections(
    df_ref,
    vars_non_nan,
    baseline_random_draws_df,
    output_dir_kmedoids,
    output_dir_sa,
    output_dir_run,
    df_pass,
):
    """6. Scoring all selections."""
    logger = logging.getLogger(__name__)
    logger.info("##### 6. Scoring all selections... #####")

    k_medoids_df = pd.read_csv(
        os.path.join(output_dir_kmedoids, "kmedoids_year_selection.csv")
    )

    k_medoids_modelyears = tuple(
        zip(
            k_medoids_df["model"].astype(str),
            k_medoids_df["Hydrological_year"].astype(int),
        )
    )
    selection_sources = {
        "random_draws": baseline_random_draws_df,
        #   "k_medoids": k_medoids_modelyears,
        "simulated_annealing": _parse_model_years(
            pd.read_csv(os.path.join(output_dir_sa, "sa_year_selections.csv"))
        ),
    }

    all_rows = []
    for source_name, df_sel in selection_sources.items():
        logger.info(f"Scoring selections from {source_name}...")
        for _, row in df_sel.iterrows():
            sel_modelyears = row["model_years"]
            df_sub = build_df_from_modelyears(df_ref, sel_modelyears, YEAR_COL)
            if SEASONAL:
                scores = score_df_seasonal(
                    df_ref,
                    df_sub,
                    sel_modelyears,
                    None,
                    None,
                    None,
                    vars_non_nan,
                    skip_check=False,
                )[1]
            else:
                scores = score_df(
                    df_ref,
                    df_sub,
                    sel_modelyears,
                    None,
                    None,
                    None,
                    vars_non_nan,
                    skip_check=False,
                )[1]
            score_row = {"selection_source": source_name, "model_years": sel_modelyears}
            score_row.update(scores)
            all_rows.append(score_row)

    # K-medoids
    df_k_med = build_df_from_modelyears(df_ref, k_medoids_modelyears, YEAR_COL)
    if SEASONAL:
        kmedoid_scores = score_df_seasonal(
            df_ref,
            df_k_med,
            k_medoids_modelyears,
            None,
            None,
            None,
            vars_non_nan,
            skip_check=False,
        )[1]
    else:
        kmedoid_scores = score_df(
            df_ref,
            df_k_med,
            k_medoids_modelyears,
            None,
            None,
            None,
            vars_non_nan,
            skip_check=False,
        )[1]

    score_row = {"selection_source": "k_medoids", "model_years": k_medoids_modelyears}
    score_row.update(kmedoid_scores)
    all_rows.append(score_row)

    if RUN_HBF:
        # concat HBF pass selections (pre-calculated)
        df_all_results = pd.concat([pd.DataFrame(all_rows), df_pass], ignore_index=True)
    else:
        df_all_results = pd.DataFrame(all_rows)

    # sort by score
    df_all_results = df_all_results.sort_values(
        "wasserstein_seasonal" if SEASONAL else "wasserstein"
    ).reset_index(drop=True)

    df_all_results.to_csv(
        os.path.join(output_dir_run, "all_year_selections_scored.csv"), index=True
    )
    pd.Series(vars_non_nan).to_csv(
        os.path.join(output_dir_run, "variable_order.csv"), index=False
    )

    logger.info("All selections scored and saved.")
    return df_all_results


def process_row(row):
    """Score a single row (selection) on a country-level. Function processed in parallel."""
    sel_modelyears = row["model_years"]
    df_sub = build_df_from_modelyears(df_ref, sel_modelyears, YEAR_COL)

    row_scores_per_country = []
    for country in MATCH_COUNTRIES:
        country_vars = [v for v in vars_non_nan if v.startswith(f"{country}_")]
        if SEASONAL:
            scores = score_df_seasonal(
                df_ref,
                df_sub,
                sel_modelyears,
                None,
                None,
                None,
                country_vars,
                n_projections=50,  # reduce computation time
                skip_check=False,
            )[1]
        else:
            scores = score_df(
                df_ref,
                df_sub,
                sel_modelyears,
                None,
                None,
                None,
                country_vars,
                n_projections=50,  # reduce computation time
                skip_check=False,
            )[1]

        score_row = {
            "selection_source": row["selection_source"],
            "model_years": sel_modelyears,
            "country": country,
        }
        score_row.update(scores)
        row_scores_per_country.append(score_row)
    return row_scores_per_country


def calculate_country_scores(df_all_results, n_workers, output_dir_run):
    """7. Calculating country-level scores."""
    logger = logging.getLogger(__name__)
    logger.info("##### 7. Calculating country-level scores... #####")

    row_dicts = df_all_results[["selection_source", "model_years"]].to_dict("records")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        nested_rows = list(tqdm(ex.map(process_row, row_dicts), total=len(row_dicts)))

    country_rows = []
    # unpack nested list
    for sub in nested_rows:
        country_rows.extend(sub)

    df_country_scores = (
        pd.DataFrame(country_rows)
        .sort_values("wasserstein_seasonal" if SEASONAL else "wasserstein")
        .reset_index(drop=True)
    )
    df_country_scores.to_csv(
        os.path.join(output_dir_run, "all_year_selections_country_scores.csv"),
        index=True,
    )
    return df_country_scores


def run_evaluation(
    df_all_results,
    df_country_scores,
    vars_non_nan,
    output_dir_eval,
    output_dir_eval_country,
):
    """8. Plotting automatic evaluation."""
    logger = logging.getLogger(__name__)
    logger.info("##### 8. Performing automatic evaluation... #####")

    if SEASONAL:
        list_cols = [
            "d_mean_seasonal",
            "qq_mae_seasonal",
            "energy_distance_score_seasonal",
            "qq_mae_high_seasonal",
            "qq_mae_low_seasonal",
        ]
    else:
        list_cols = [
            "d_mean",
            "qq_mae",
            "energy_distance_score",
            "qq_mae_high",
            "qq_mae_low",
        ]

    for col in list_cols:
        col_max = "max_" + col
        df_all_results[col_max] = df_all_results[col].apply(
            lambda x: max(literal_eval(x)) if isinstance(x, str) else max(x)
        )

    plot_results_with_pareto(
        df_all_results,
        metric_1="wasserstein_seasonal",
        metric_2="wasserstein",
        save_dir=output_dir_eval,
    )

    if SEASONAL:
        plot_results_with_pareto(
            df_all_results,
            metric_1="max_qq_mae_seasonal",
            metric_2="max_energy_distance_score_seasonal",
            save_dir=output_dir_eval,
        )

        plot_results_with_pareto(
            df_all_results,
            metric_1="max_qq_mae_high_seasonal",
            metric_2="max_qq_mae_low_seasonal",
            save_dir=output_dir_eval,
        )
    else:
        plot_results_with_pareto(
            df_all_results,
            metric_1="max_qq_mae",
            metric_2="max_energy_distance_score",
            save_dir=output_dir_eval,
        )

        plot_results_with_pareto(
            df_all_results,
            metric_1="max_qq_mae_high",
            metric_2="max_qq_mae_low",
            save_dir=output_dir_eval,
        )

    plot_lollipop_by_country(
        df_all_results,
        df_country_scores,
        MATCH_COUNTRIES,
        "wasserstein",
        "SWD",
        save_dir=output_dir_eval,
    )

    plot_lollipop_by_country(
        df_all_results,
        df_country_scores,
        MATCH_COUNTRIES,
        "wasserstein_seasonal",
        "Worst seasonal SWD",
        save_dir=output_dir_eval,
    )

    perform_physical_evaluation(
        df_all_results,
        df_ref,
        MATCH_COUNTRIES,
        vars_non_nan,
        output_dir_eval_country,
    )


def main():
    """Run the full selection pipeline."""
    # 0. Setup
    (
        output_dir_run,
        output_dir_random,
        output_dir_kmedoids,
        output_dir_sa,
        output_dir_hbf,
        output_dir_eval,
        output_dir_eval_country,
    ) = setup_output_directories()

    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    logger.info(f"Run started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    NUM_SA_EXPERIMENTS = len(PARAM_GRID["cooling_rate"]) * len(SEEDS)
    SA_COOLING_RATES = PARAM_GRID["cooling_rate"]
    logger.info(
        "##### Run settings #####\n"
        f"  MATCH_COUNTRIES    : {MATCH_COUNTRIES}\n"
        f"  NUM_TO_SELECT      : {NUM_TO_SELECT}\n"
        f"  HYDRO              : {HYDRO}\n"
        f"  SEASONAL           : {SEASONAL}\n"
        f"  HBF_ITERATIONS     : {HBF_ITERATIONS}\n"
        f"  NUM_SA_EXPERIMENTS : {NUM_SA_EXPERIMENTS}\n"
        f"  SA_COOLING_RATES   : {SA_COOLING_RATES}"
    )

    # 1. Load data
    # HACK: make df_ref and vars_non_nan global for multiprocessing
    global df_ref, vars_non_nan
    df_ref, vars_non_nan, model_years, rng = load_data()

    # 2. Simulated Annealing (must run before random draws to determine n_iter)
    df_results, n_workers = run_simulated_annealing(df_ref, vars_non_nan, output_dir_sa)

    # 3. Baseline Random Draws (num draws = sum of SA n_iter)
    baseline_random_draws_df = run_baseline_random_draws(
        model_years, vars_non_nan, df_results, output_dir_random
    )

    # 4. K-medoids
    run_kmedoids(df_ref, vars_non_nan, output_dir_kmedoids, SEASONAL)

    # 5. Hierarchical Brute-Force
    df_pass = run_hierarchical_brute_force(
        df_ref, vars_non_nan, model_years, rng, output_dir_hbf
    )

    # 6. Score all selections
    df_all_results = score_all_selections(
        df_ref,
        vars_non_nan,
        baseline_random_draws_df,
        output_dir_kmedoids,
        output_dir_sa,
        output_dir_run,
        df_pass,
    )

    # 7. Country-level scores
    df_country_scores = calculate_country_scores(
        df_all_results, n_workers, output_dir_run
    )

    # 8. Evaluation plots
    run_evaluation(
        df_all_results,
        df_country_scores,
        vars_non_nan,
        output_dir_eval,
        output_dir_eval_country,
    )

    end_time = datetime.now()
    logger.info(f"Run completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total run time: {end_time - start_time}")


# %%
if __name__ == "__main__":
    main()
