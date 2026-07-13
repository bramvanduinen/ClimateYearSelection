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
    # BATCH_SIZE,
    DF_PROD_DEMAND_FNAME,
    FRS_ITER_MULTIPLIER,
    HYDRO,
    MATCH_COUNTRIES,
    N_PROJECTIONS_FULL,
    N_PROJECTIONS_INIT,
    NUM_EXPERIMENTS,
    NUM_TO_SELECT,
    P_THRES,
    PARAM_GRID,
    RUN_FRS,
    RUN_SA,
    RUNNAME,
    RUNNAME_SA_FALLBACK,
    SEASONAL,
    SEED_OFFSET,
    TUNING_ITERATIONS,
    VARS_TO_MATCH,
    YEAR_COL,
)
from tqdm import tqdm

from cluster_weather_years.constants import BASE_OUTPUT, HYDROLOGICAL_YEARS, MODELS
from cluster_weather_years.cost_functions import (
    wasserstein_score,
    wasserstein_seasonal_score,
)
from cluster_weather_years.evaluation import (
    perform_physical_evaluation,
    plot_lollipop_by_country,
)
from cluster_weather_years.evaluation_plot_helpers import plot_results_with_pareto
from cluster_weather_years.filtered_random_search import (
    init_worker,
    run_batch,
    score_df,
    score_df_seasonal,
    score_tuning,
    score_tuning_seasonal,
)
from cluster_weather_years.helper_functions import (
    build_df_from_modelyears,
    draw_random_modelyears,
)
from cluster_weather_years.k_medoids_stochastic import k_medoids_stochastic
from cluster_weather_years.sa import run_sa

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

    output_dir_random = os.path.join(output_dir_run, "random_search")
    output_dir_kmedoids = os.path.join(output_dir_run, "k_medoids")
    output_dir_sa = os.path.join(output_dir_run, "simulated_annealing")
    output_dir_frs = os.path.join(output_dir_run, "filtered_random_search")
    output_dir_eval = os.path.join(output_dir_run, "evaluation")
    output_dir_eval_country = os.path.join(output_dir_eval, "by_country")

    os.makedirs(output_dir_random, exist_ok=True)
    os.makedirs(output_dir_kmedoids, exist_ok=True)
    os.makedirs(output_dir_sa, exist_ok=True)
    os.makedirs(output_dir_frs, exist_ok=True)
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
        output_dir_frs,
        output_dir_eval,
        output_dir_eval_country,
    )


def load_data():
    """1. Load and prepare data."""
    logger = logging.getLogger(__name__)
    logger.info("##### 1. Loading data... #####")

    df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])
    # if HYDRO:
    #     # HACK: overwrite default Hydrological years
    #     # Hydro data starts only in 2015-07-20
    #     HYDROLOGICAL_YEARS = range(2016, 2046)
    # else:
    #     # HACK: Should be taken from constants, but not found?
    #     HYDROLOGICAL_YEARS = range(2016, 2046)
    df_daily = df_daily.query("Hydrological_year in @HYDROLOGICAL_YEARS")
    df_daily = df_daily.set_index(["model", "Date"])

    # HACK: Hydro contains 0 instead of nans, so there have to be non-zero values.
    vars_non_nan = [var for var in VARS_TO_MATCH if np.max(df_daily[var]) > 0]

    df_ref = df_daily[
        ["Hydrological_year", "Calendar_year"] + VARS_TO_MATCH
    ].reset_index()

    model_years = list(product(MODELS, HYDROLOGICAL_YEARS))
    rng = np.random.default_rng(0)

    return df_ref, vars_non_nan, model_years, rng


def _draw_random_batch(args):
    """Draw and score a batch of random year selections; return the best result and median score.

    Two-step scoring: all draws are first evaluated cheaply (n_projections_init), then
    the top 10% are re-scored at full resolution (n_projections_full). The best of those
    full-resolution top-10% candidates is returned.
    """
    (
        seed,
        model_years,
        batch_size,
        num_to_select,
        vars_non_nan,
        n_projections_init,
        n_projections_full,
    ) = args
    rng = np.random.default_rng(seed)
    draws = draw_random_modelyears(model_years, batch_size, num_to_select, rng)
    score_key = "wasserstein_seasonal" if SEASONAL else "wasserstein"
    score_fn = wasserstein_seasonal_score if SEASONAL else wasserstein_score

    # --- First pass: cheap scoring ---
    all_results: list[dict] = []
    all_scores: list[float] = []
    for model_years_draw in draws:
        score = score_fn(
            df=df_ref,
            combos=model_years_draw,
            variables=vars_non_nan,
            year_col=YEAR_COL,
            n_projections=n_projections_init,
        )
        all_scores.append(float(score))
        all_results.append({"model_years": model_years_draw, score_key: float(score)})

    median_score = float(np.median(all_scores))

    # --- Second pass: full-resolution re-scoring for top 10% ---
    k = max(1, int(len(all_results) * 0.1))
    top_indices = np.argsort(all_scores)[:k]
    best_score = float("inf")
    best_result = None
    for idx in top_indices:
        result = all_results[idx]
        full_score = score_fn(
            df=df_ref,
            combos=result["model_years"],
            variables=vars_non_nan,
            year_col=YEAR_COL,
            n_projections=n_projections_full,
        )
        result[score_key] = float(full_score)
        if full_score < best_score:
            best_score = float(full_score)
            best_result = result

    return best_result, median_score


def run_random_search(model_years, vars_non_nan, num_draws_total, output_dir_random):
    """3. Random Search.

    Draws are split into NUM_EXPERIMENTS batches (each seeded by its experiment index).
    Only the best (lowest wasserstein score) from each batch is kept, yielding
    exactly NUM_EXPERIMENTS results — one per batch.
    """
    logger = logging.getLogger(__name__)
    logger.info("##### 3. Performing Random Search... #####")
    batch_size = math.ceil(num_draws_total / NUM_EXPERIMENTS)
    logger.info(
        f"Drawing and scoring {num_draws_total} random selections in {NUM_EXPERIMENTS} batches "
        f"(batch_size={batch_size}) in mode {'seasonal' if SEASONAL else 'annual'}. "
        f"Keeping best 1 per batch."
    )

    n_workers = min(os.cpu_count() or 1, NUM_EXPERIMENTS)
    # Each experiment gets its own unique seed (experiment index)
    job_args = [
        (
            seed,
            model_years,
            batch_size,
            NUM_TO_SELECT,
            vars_non_nan,
            N_PROJECTIONS_INIT,
            N_PROJECTIONS_FULL,
        )
        for seed in range(NUM_EXPERIMENTS)
    ]

    random_search_results = []
    sort_col = "wasserstein_seasonal" if SEASONAL else "wasserstein"
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_draw_random_batch, args): i for i, args in enumerate(job_args)
        }
        for fut in as_completed(futures):
            batch_idx = futures[fut]
            best, median_score = fut.result()
            if best is not None:
                best[f"{sort_col}_median"] = median_score
                random_search_results.append(best)
                logger.info(
                    f"Batch {batch_idx} done — best {sort_col} = {best[sort_col]:.4f}, median = {median_score:.4f}"
                )
            else:
                logger.info(f"Batch {batch_idx} done — no results")

    df_random_search_results = (
        pd.DataFrame(random_search_results).sort_values(sort_col).reset_index(drop=True)
    )
    df_random_search_results["selection_source"] = "random_search"
    df_random_search_results.to_csv(
        os.path.join(output_dir_random, "random_search_year_selections.csv")
    )

    overall_best = df_random_search_results.iloc[0]
    best_batch = int(
        df_random_search_results.index[
            df_random_search_results[sort_col] == overall_best[sort_col]
        ][0]
    )
    logger.info(
        f"Random Search done — best {sort_col} = {overall_best[sort_col]:.4f} "
        f"(batch {best_batch}). "
        f"{len(df_random_search_results)} results from {num_draws_total} total draws "
        f"saved to {output_dir_random}."
    )
    return df_random_search_results


def run_kmedoids(df_ref, vars_non_nan, sa_total_iters, output_dir_kmedoids):
    """4. K-Medoids++ clustering.

    Runs ``sa_total_iters`` independent K-Medoids++ fits split into
    ``NUM_EXPERIMENTS`` parallel batches, keeping the best-scoring result
    per batch. Identical compute budget to SA and the random draws.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"##### 4. Performing K-Medoids in {'seasonal' if SEASONAL else 'annual'} mode "
        f"({sa_total_iters} total fits across {NUM_EXPERIMENTS} batches)... #####"
    )

    df_kmedoids_results = k_medoids_stochastic(
        df=df_ref,
        n_clusters=NUM_TO_SELECT,
        vars_to_match=vars_non_nan,
        output_dir=output_dir_kmedoids,
        num_runs_total=sa_total_iters,
        num_experiments=NUM_EXPERIMENTS,
        year_col=YEAR_COL,
        SEASONAL=SEASONAL,
        n_projections_init=N_PROJECTIONS_INIT,
        n_projections_full=N_PROJECTIONS_FULL,
        seed_offset=SEED_OFFSET,
    )
    logger.info(f"K-Medoids completed. Results saved to {output_dir_kmedoids}.")
    return df_kmedoids_results


def run_simulated_annealing(df_ref, vars_non_nan, output_dir_sa):
    """2. Simulated Annealing (or load fallback results)."""
    logger = logging.getLogger(__name__)
    if RUN_SA:
        logger.info(
            f"##### 2. Performing Simulated Annealing in {'seasonal' if SEASONAL else 'annual'} mode... #####"
        )

        # build jobs
        sa_shared_inputs = {
            "df_daily": df_ref,
            "vars_to_match": vars_non_nan,
            "NUM_CLUSTERS": NUM_TO_SELECT,
            "output_dir_sa": output_dir_sa,
            "YEAR_COL": YEAR_COL,
            "n_projections_init": N_PROJECTIONS_INIT,
            "n_projections_full": N_PROJECTIONS_FULL,
        }
        # Distribute cooling rates fairly across NUM_EXPERIMENTS unique seeds:
        # experiment i gets cooling_rates[i * n_cooling_rates // NUM_EXPERIMENTS]
        cooling_rates = PARAM_GRID["cooling_rate"]
        n_cooling_rates = len(cooling_rates)
        metric = wasserstein_seasonal_score if SEASONAL else wasserstein_score
        temp_start = PARAM_GRID["temp_start"][0]
        max_iter = PARAM_GRID["max_iter"][0]
        param_set = [
            {
                "seed": i + SEED_OFFSET,
                "metric": metric,
                "t0": temp_start,
                "cr": cooling_rates[i * n_cooling_rates // NUM_EXPERIMENTS],
                "mi": max_iter,
            }
            for i in range(NUM_EXPERIMENTS)
        ]

        # run jobs in parallel
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
                    f"Experiment {i} done: {sa_result['optimized_metric']} = {sa_result['best_score']:.2f}"
                )

        # save results
        df_sa_results = pd.DataFrame(sa_result_dicts)
        df_sa_results.to_csv(os.path.join(output_dir_sa, "sa_year_selections.csv"))
        best = min(sa_result_dicts, key=lambda d: d["best_score"])
        logger.info(f"SA run done, best score: {best['best_score']:.2f}")
    else:
        logger.info("##### 2. Skipping Simulated Annealing as per settings... #####")

        n_workers = os.cpu_count()
        df_sa_results = _parse_model_years(
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
        df_sa_results.to_csv(
            os.path.join(output_dir_sa, "sa_year_selections.csv"), index=True
        )
    logger.info(f"Simulated Annealing results saved to {output_dir_sa}.")

    return df_sa_results, n_workers


def run_filtered_random_search(
    df_ref,
    vars_non_nan,
    model_years,
    rng,
    output_dir_frs,
    frs_iterations,
):
    """5. Filtered Random Search (FRS)."""
    # NOTE: scales very poorly with increasing vars_to_match
    logger = logging.getLogger(__name__)
    if RUN_FRS:
        logger.info("##### 5. Performing Filtered Random Search (FRS)... #####")
        logger.info(
            f"Running FRS with {'seasonal scoring' if SEASONAL else 'regular scoring'}..."
        )

        # TODO: make sure it's reproducible, e.g., by setting seeds properly
        if SEASONAL:
            _, _, thresholds_mean, thresholds_energy = score_tuning_seasonal(
                df_ref,
                model_years,
                TUNING_ITERATIONS,
                NUM_TO_SELECT,
                vars_non_nan,
                rng,
                YEAR_COL,
                percentile_threshold=P_THRES,
            )
        else:
            _, _, thresholds_mean, thresholds_energy = score_tuning(
                df_ref,
                model_years,
                TUNING_ITERATIONS,
                NUM_TO_SELECT,
                vars_non_nan,
                rng,
                YEAR_COL,
                percentile_threshold=P_THRES,
            )

        df_thresholds_energy = pd.DataFrame(
            thresholds_energy.items(), columns=["Variable", "Threshold Energy"]
        )
        df_thresholds_mean = pd.DataFrame(
            thresholds_mean.items(), columns=["Variable", "Threshold Mean"]
        )
        df_thresholds_energy.merge(df_thresholds_mean, on="Variable").to_csv(
            os.path.join(output_dir_frs, "frs_thresholds.csv"), index=True
        )

        # Split frs_iterations into NUM_EXPERIMENTS batches (one unique seed each).
        # Only the best result (lowest wasserstein score) per batch is kept.
        n_batches = NUM_EXPERIMENTS
        batch_size_frs = math.ceil(frs_iterations / n_batches)
        frs_total = n_batches * batch_size_frs  # may be slightly > frs_iterations
        # Seeds: experiments 0..NUM_EXPERIMENTS-1
        frs_seeds = list(range(n_batches))

        # Each record in pass_rec already contains "model_years" (set inside
        # score_df / score_df_seasonal), so it lands in df_frs_results automatically.
        frs_results_pass = []  # one best-scoring record per batch
        frs_results_fail = []

        max_workers = os.cpu_count()
        score_col = "wasserstein_seasonal" if SEASONAL else "wasserstein"

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
                N_PROJECTIONS_INIT,
                N_PROJECTIONS_FULL,
            ),
        ) as ex:
            hbf_batch_iter = ex.map(
                run_batch,
                range(n_batches),
                [batch_size_frs] * n_batches,
                [frs_total] * n_batches,
                frs_seeds,
                [SEASONAL] * n_batches,
            )

            for batch_idx, (
                batch_pass_records,
                batch_fail_records,
                _pass_years,
            ) in enumerate(tqdm(hbf_batch_iter, total=n_batches)):
                frs_results_fail.extend(batch_fail_records)
                if batch_pass_records:
                    best = min(batch_pass_records, key=lambda r: r[score_col])
                    median = float(
                        np.median([r[score_col] for r in batch_pass_records])
                    )
                    best[f"{score_col}_median"] = median

                    frs_results_pass.append(best)
                    logger.info(
                        f"Batch {batch_idx} done — best {score_col} = {best[score_col]:.4f}, median = {median:.4f}"
                    )
                else:
                    logger.info(f"Batch {batch_idx} done — no passing draws")

        df_frs_results = pd.DataFrame(frs_results_pass)
        if not df_frs_results.empty:
            df_frs_results = df_frs_results.sort_values(score_col).reset_index(
                drop=True
            )
        df_frs_results["selection_source"] = "FRS"
        df_frs_fail = pd.DataFrame(frs_results_fail)[["fail_check", "fail_var"]]

        skip_rate = len(df_frs_fail) / frs_iterations
        logger.info(f"Skip rate: {skip_rate:.3f}")

        if not df_frs_results.empty:
            overall_best = df_frs_results.iloc[0]
            logger.info(
                f"FRS done — best {score_col} = {overall_best[score_col]:.4f} "
                f"(batch {int(df_frs_results.index[df_frs_results[score_col] == overall_best[score_col]][0])})."
            )

        logger.info(
            f"Failure distribution:\n{df_frs_fail.groupby(['fail_check', 'fail_var']).size().sort_values(ascending=False).head(20)}"
        )

        df_frs_results.to_csv(
            os.path.join(output_dir_frs, "frs_df_pass.csv"), index=True
        )
        return df_frs_results
    else:
        logger.info("##### 5. Skipping Filtered Random Search as per settings... #####")
        return None


def score_all_selections(
    df_ref,
    vars_non_nan,
    df_random_search_results,
    df_kmedoids_results,
    output_dir_sa,
    output_dir_run,
    df_frs_results,
):
    """6. Scoring all selections."""
    logger = logging.getLogger(__name__)
    logger.info("##### 6. Scoring all selections... #####")

    selection_sources = {
        "random_search": df_random_search_results,
        "simulated_annealing": _parse_model_years(
            pd.read_csv(os.path.join(output_dir_sa, "sa_year_selections.csv"))
        ),
        "k_medoids": _parse_model_years(df_kmedoids_results.reset_index()),
    }

    scored_rows = []
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
            scored_rows.append(score_row)

    if RUN_FRS:
        # concat FRS pass selections (pre-calculated)
        df_all_results = pd.concat(
            [pd.DataFrame(scored_rows), df_frs_results], ignore_index=True
        )
    else:
        df_all_results = pd.DataFrame(scored_rows)

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
                n_projections=N_PROJECTIONS_INIT,  # reduce computation time
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
                n_projections=N_PROJECTIONS_INIT,  # reduce computation time
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
        country_score_rows_nested = list(
            tqdm(ex.map(process_row, row_dicts), total=len(row_dicts))
        )

    country_score_rows = []
    # unpack nested list
    for sub in country_score_rows_nested:
        country_score_rows.extend(sub)

    df_country_scores = (
        pd.DataFrame(country_score_rows)
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
        metric_cols = [
            "d_mean_seasonal",
            "qq_mae_seasonal",
            "energy_distance_score_seasonal",
            "qq_mae_high_seasonal",
            "qq_mae_low_seasonal",
        ]

        score_col = "wasserstein_seasonal"
    else:
        metric_cols = [
            "d_mean",
            "qq_mae",
            "energy_distance_score",
            "qq_mae_high",
            "qq_mae_low",
        ]
        score_col = "wasserstein"

    for col in metric_cols:
        col_max = "max_" + col
        df_all_results[col_max] = df_all_results[col].apply(
            lambda x: max(literal_eval(x)) if isinstance(x, str) else max(x)
        )

    # luck analysis: how does median score compare to best score
    df_luck = (
        df_all_results.groupby("selection_source")[score_col]
        .agg(
            best_score="min",
            median_score="median",
            worst_score="max",
        )
        .assign(
            # luck_factor=lambda df: (
            #     (df["median_score"] - df["best_score"]) / df["best_score"]
            # ),
            luck_factor=lambda df: (
                (df["median_score"] - df["best_score"])
                / df.loc["random_search", "median_score"]
            ),
            misfortune_factor=lambda df: (
                (df["worst_score"] - df["median_score"])
                / df.loc["random_search", "median_score"]
            ),
        )
        .sort_values("best_score")
    )

    df_luck.to_csv(os.path.join(output_dir_eval, "luck_analysis.csv"))

    logger.info(f"Luck analysis:\n{df_luck}")

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
        output_dir_frs,
        output_dir_eval,
        output_dir_eval_country,
    ) = setup_output_directories()

    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    logger.info(f"Run started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    sa_cooling_rates = PARAM_GRID["cooling_rate"]
    logger.info(
        "##### Run settings #####\n"
        f"  MATCH_COUNTRIES    : {MATCH_COUNTRIES}\n"
        f"  NUM_TO_SELECT      : {NUM_TO_SELECT}\n"
        f"  HYDRO              : {HYDRO}\n"
        f"  SEASONAL           : {SEASONAL}\n"
        f"  FRS_ITER_MULTIPLIER: {FRS_ITER_MULTIPLIER}x\n"
        f"  NUM_EXPERIMENTS    : {NUM_EXPERIMENTS}\n"
        f"  SA_COOLING_RATES   : {sa_cooling_rates} (distributed over {NUM_EXPERIMENTS} unique seeds)"
    )

    # 1. Load data
    # HACK: make df_ref and vars_non_nan global for multiprocessing
    global df_ref, vars_non_nan
    df_ref, vars_non_nan, model_years, rng = load_data()

    timings: dict[str, datetime] = {}

    # 2. Simulated Annealing
    _t0 = datetime.now()
    df_sa_results, n_workers = run_simulated_annealing(
        df_ref, vars_non_nan, output_dir_sa
    )
    timings["Simulated Annealing"] = datetime.now() - _t0
    # Use number of iterations across all SA experiments as compute budget for other methods
    sa_total_iters = int(df_sa_results["n_iter"].sum())

    # 3. Random Search (num draws = sum of SA n_iter)
    _t0 = datetime.now()
    df_random_search_results = run_random_search(
        model_years, vars_non_nan, sa_total_iters, output_dir_random
    )
    timings["Random Search"] = datetime.now() - _t0

    # 4. K-Medoids (same compute budget as SA)
    _t0 = datetime.now()
    df_kmedoids_results = run_kmedoids(
        df_ref, vars_non_nan, sa_total_iters, output_dir_kmedoids
    )
    timings["K-Medoids"] = datetime.now() - _t0

    # 5. Filtered Random Search (FRS)
    frs_iterations = FRS_ITER_MULTIPLIER * sa_total_iters
    logger.info(
        f"FRS will run {frs_iterations} iterations "
        f"({FRS_ITER_MULTIPLIER}x SA total of {sa_total_iters})."
    )
    _t0 = datetime.now()
    df_frs_results = run_filtered_random_search(
        df_ref, vars_non_nan, model_years, rng, output_dir_frs, frs_iterations
    )
    timings["FRS"] = datetime.now() - _t0

    # 6. Score all selections
    df_all_results = score_all_selections(
        df_ref,
        vars_non_nan,
        df_random_search_results,
        df_kmedoids_results,
        output_dir_sa,
        output_dir_run,
        df_frs_results,
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

    timing_lines = "\n".join(
        f"  {name:<25} {str(duration).split('.')[0]}"
        for name, duration in sorted(timings.items(), key=lambda x: x[1], reverse=True)
    )
    logger.info(f"Time spent per method:\n{timing_lines}")


# %%
if __name__ == "__main__":
    main()
