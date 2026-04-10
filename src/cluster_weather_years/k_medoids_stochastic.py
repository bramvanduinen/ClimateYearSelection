"""Stochastic K-Medoids clustering for weather year selection."""

# %%
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

from cluster_weather_years.constants import SEASONS
from cluster_weather_years.cost_functions import (
    wasserstein_score,
    wasserstein_seasonal_score,
)

log = logging.getLogger(__name__)


# Worker function
def _run_kmedoids_batch(args):
    """Fit KMedoids *batch_size* times and return the best result and median score.

    Parameters
    ----------
    args : tuple
        (batch_seed, batch_size, stats_scaled, yearly_stats,
         df_ref, vars_non_nan, n_clusters, year_col, seasonal)

    Returns
    -------
    tuple[dict, float]
        (best_result dict, median_score) for this batch.
    """
    (
        batch_seed,
        batch_size,
        stats_scaled,
        yearly_stats,
        df_ref,
        vars_non_nan,
        n_clusters,
        year_col,
        seasonal,
    ) = args

    score_fn = wasserstein_seasonal_score if seasonal else wasserstein_score
    score_key = "wasserstein_seasonal" if seasonal else "wasserstein"

    rng = np.random.default_rng(batch_seed)
    best_score = float("inf")
    best_result: Optional[dict] = None
    all_scores: list[float] = []

    for _ in range(batch_size):
        run_seed = int(rng.integers(0, 2**31))
        kmed = KMedoids(
            n_clusters=n_clusters, init="k-medoids++", random_state=run_seed
        ).fit(stats_scaled)

        medoid_rows = yearly_stats.iloc[kmed.medoid_indices_]
        combos = tuple(
            zip(medoid_rows["model"].tolist(), medoid_rows[year_col].tolist())
        )

        score = score_fn(
            df=df_ref,
            combos=combos,
            variables=vars_non_nan,
            year_col=year_col,
            n_projections=50,
        )
        all_scores.append(float(score))

        if score < best_score:
            best_score = float(score)
            best_result = {
                "model_years": combos,
                "run_seed": run_seed,
                "batch_seed": batch_seed,
                "inertia": float(kmed.inertia_),
                score_key: best_score,
                "selection_source": "k_medoids_stochastic",
            }

    median_score = float(np.median(all_scores))
    return best_result, median_score


def k_medoids_stochastic(
    df: pd.DataFrame,
    n_clusters: int,
    vars_to_match: List[str],
    output_dir: str,
    num_runs_total: int,
    num_experiments: int,
    year_col: str = "Hydrological_year",
    SEASONAL: bool = False,
) -> pd.DataFrame:
    """Stochastic K-Medoids with k-medoids++ initialisation.

    Runs ``num_runs_total`` independent K-Medoids++ fits, split into
    ``num_experiments`` parallel batches (one subprocess each).  Within
    every batch the fit with the lowest Wasserstein score is kept, so
    the function returns exactly ``num_experiments`` rows — one
    best-per-batch.
    """

    log.info(
        "##### Stochastic K-Medoids: %d total runs in %d batches "
        "(batch_size=%d) [%s mode] #####",
        num_runs_total,
        num_experiments,
        math.ceil(num_runs_total / num_experiments),
        "seasonal" if SEASONAL else "annual",
    )

    # Calculate features
    df_raw = df[["model", year_col] + vars_to_match].copy()
    scaler = StandardScaler()
    df_std = pd.DataFrame(
        scaler.fit_transform(df_raw[vars_to_match]),
        columns=vars_to_match,
        index=df_raw.index,
    )
    df_std[["model", year_col]] = df_raw[["model", year_col]]

    if SEASONAL:
        df_std["Date"] = pd.to_datetime(df["Date"].values)
        season_frames = []
        for season_name, months in SEASONS.items():
            in_season = df_std["Date"].dt.month.isin(months)
            df_season = df_std.loc[in_season]
            season_stats = df_season.groupby(["model", year_col])[vars_to_match].agg(
                ["mean", "var"]
            )
            season_stats.columns = [
                f"{var}_{stat}_{season_name}" for var, stat in season_stats.columns
            ]
            season_frames.append(season_stats)
        yearly_stats = pd.concat(season_frames, axis=1).reset_index()
    else:
        yearly_stats = df_std.groupby(["model", year_col]).agg(["mean", "var"])
        yearly_stats.columns = [f"{var}_{stat}" for var, stat in yearly_stats.columns]
        yearly_stats = yearly_stats.reset_index()

    feat_cols = [c for c in yearly_stats.columns if c not in (year_col, "model")]
    stats_scaled = scaler.fit_transform(yearly_stats[feat_cols])

    # Get non-zero variables for scoring
    vars_non_nan = [v for v in vars_to_match if np.max(df[v]) > 0]

    # Build one job tuple per batch
    batch_size = math.ceil(num_runs_total / num_experiments)
    n_workers = min(os.cpu_count() or 1, num_experiments)

    # HACK: Temporary seed offset for test here!

    SEED_OFFSET = 100  # change this for a different stochastic outcome
    job_args = [
        (
            SEED_OFFSET + batch_seed,  # unique seed per batch → unique RNG stream
            batch_size,
            stats_scaled,
            yearly_stats,
            df,
            vars_non_nan,
            n_clusters,
            year_col,
            SEASONAL,
        )
        for batch_seed in range(num_experiments)
    ]

    # Run batches in parallel
    all_best: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_run_kmedoids_batch, args): batch_seed
            for batch_seed, args in enumerate(job_args)
        }
        for fut in as_completed(futures):
            result, median_score = fut.result()
            result["median_score"] = median_score
            all_best.append(result)
            log.info(
                "Batch %d done — best %s = %.4f, median = %.4f",
                futures[fut],
                "wasserstein_seasonal" if SEASONAL else "wasserstein",
                result["wasserstein_seasonal" if SEASONAL else "wasserstein"],
                median_score,
            )

    # Collect, sort by score, save
    score_key = "wasserstein_seasonal" if SEASONAL else "wasserstein"
    df_results = pd.DataFrame(all_best).sort_values(score_key).reset_index(drop=True)
    df_results["selection_source"] = "k_medoids_stochastic"

    out_path = os.path.join(output_dir, "kmedoids_stochastic_year_selections.csv")
    df_results.to_csv(out_path, index=False)

    best = df_results.iloc[0]
    log.info(
        "Stochastic K-Medoids done — best %s = %.4f (from batch %d). "
        "Results saved to %s.",
        score_key,
        best[score_key],
        int(best["batch_seed"]),
        out_path,
    )
    return df_results
