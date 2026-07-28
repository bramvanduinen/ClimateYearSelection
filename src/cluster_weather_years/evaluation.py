import os
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ot import sliced_wasserstein_distance

# fmt: off
# pylint: disable=import-error
from cluster_weather_years.constants import (
    BASE_OUTPUT,
    DROUGHT_MIN_DURATION,
    DROUGHT_PERCENTILE,
    PROBS,
)
from cluster_weather_years.helper_functions import (
    build_df_from_modelyears,
)

# pylint: enable=import-error
# fmt: on
### Statistical Evaluation, functions used in 03_evaluation_plots.py for paper figures###


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for evaluation scoring metrics
# ─────────────────────────────────────────────────────────────────────────────


def _filter_season(df: pd.DataFrame, months) -> pd.DataFrame:
    """Keep only rows whose Date month is in `months`."""
    return df[pd.to_datetime(df["Date"]).dt.month.isin(months)]


def _ecdf(values):
    """Empirical CDF: returns (sorted_x, cumulative_probability)."""
    x = np.sort(np.asarray(values))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _qq_diff_and_mae(ref_values, sel_values, quantiles=PROBS):
    """Quantile differences and their mean absolute error between two series."""
    q_ref = np.quantile(ref_values.dropna().values, quantiles)
    q_sel = np.quantile(sel_values.dropna().values, quantiles)
    qq_diff = q_sel - q_ref
    return qq_diff, float(np.mean(np.abs(qq_diff)))


def _corr_deviations(corr_sub: pd.DataFrame, corr_ref: pd.DataFrame):
    """Mean and max absolute correlation deviation (strict lower triangle only)."""
    mask_lower = np.tril(np.ones(corr_ref.shape, dtype=bool), k=-1)
    abs_dev = np.abs((corr_sub - corr_ref).to_numpy()[mask_lower])
    return float(np.mean(abs_dev)), float(np.max(abs_dev))


def _compute_swd(X_sub, X_ref, n_projections: int = 200, seed: int = 42) -> float:
    """Sliced Wasserstein Distance between two multivariate sample arrays."""
    return float(
        sliced_wasserstein_distance(
            X_sub, X_ref, n_projections=n_projections, seed=seed
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main metric computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    active_subsets: dict,
    df_ref: pd.DataFrame,
    vars_non_nan: list,
    seasons: dict,
    n_projections: int = 200,
    seed: int = 42,
) -> dict:
    """Compute SWD, QQ-MAE, and correlation deviations for all active subsets.

    Parameters
    ----------
    active_subsets : dict
        {name: {"df": DataFrame, "modelyears": ...}} — filtered subsets to evaluate.
    df_ref : DataFrame
        Full reference dataset (all model-years).
    vars_non_nan : list[str]
        Variable columns to include in scoring.
    seasons : dict
        {season_name: list_of_months} — e.g. {"DJF": [12, 1, 2], "JJA": [6, 7, 8]}.
    n_projections : int
        Number of random projections for SWD (higher = more accurate, slower).
    seed : int
        RNG seed for SWD reproducibility.

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    print("Computing evaluation metrics...")
    X_ref_all = df_ref[vars_non_nan].to_numpy()

    # Annual SWD
    swd_annual = {
        name: _compute_swd(
            info["df"][vars_non_nan].to_numpy(), X_ref_all, n_projections, seed
        )
        for name, info in active_subsets.items()
    }

    # Seasonal SWD
    swd_seasonal = {}
    for season_name, months in seasons.items():
        X_ref_s = _filter_season(df_ref, months)[vars_non_nan].to_numpy()
        swd_seasonal[season_name] = {
            name: _compute_swd(
                _filter_season(info["df"], months)[vars_non_nan].to_numpy(),
                X_ref_s,
                n_projections,
                seed,
            )
            for name, info in active_subsets.items()
        }

    # SSWD = max seasonal SWD per method
    sswd = {
        name: max(swd_seasonal[s][name] for s in seasons) for name in active_subsets
    }

    # Annual QQ-MAE per variable
    qq_mae_annual = {
        name: {v: _qq_diff_and_mae(df_ref[v], info["df"][v])[1] for v in vars_non_nan}
        for name, info in active_subsets.items()
    }

    # Seasonal QQ-MAE per variable
    qq_mae_seasonal = {}
    for season_name, months in seasons.items():
        df_ref_s = _filter_season(df_ref, months)
        qq_mae_seasonal[season_name] = {
            name: {
                v: _qq_diff_and_mae(df_ref_s[v], _filter_season(info["df"], months)[v])[
                    1
                ]
                for v in vars_non_nan
            }
            for name, info in active_subsets.items()
        }

    # Correlation deviations: annual + per season
    corr_ref_annual = df_ref[vars_non_nan].corr()
    corr_dev = {}
    for name, info in active_subsets.items():
        mean_ann, _ = _corr_deviations(info["df"][vars_non_nan].corr(), corr_ref_annual)
        corr_dev[name] = {"Annual": mean_ann}
        for season_name, months in seasons.items():
            df_ref_s = _filter_season(df_ref, months)
            df_sub_s = _filter_season(info["df"], months)
            mean_s, _ = _corr_deviations(
                df_sub_s[vars_non_nan].corr(), df_ref_s[vars_non_nan].corr()
            )
            corr_dev[name][season_name] = mean_s

    print(
        f"All metrics computed ✓  ({len(vars_non_nan)} vars, n_projections={n_projections}, seed={seed})"
    )
    return {
        "swd_annual": swd_annual,
        "swd_seasonal": swd_seasonal,
        "sswd": sswd,
        "qq_mae_annual": qq_mae_annual,
        "qq_mae_seasonal": qq_mae_seasonal,
        "corr_dev": corr_dev,
        "season_list": list(seasons.keys()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Seed sensitivity of the SSWD score
# ─────────────────────────────────────────────────────────────────────────────


def _seasonal_arrays(df: pd.DataFrame, vars_non_nan: list, seasons: dict) -> dict:
    """{season_name: array of the season's rows} for the given variables."""
    return {
        season_name: _filter_season(df, months)[vars_non_nan].to_numpy()
        for season_name, months in seasons.items()
    }


def sswd_seed_ensemble(
    X_sub_seasonal: dict,
    X_ref_seasonal: dict,
    n_projections: int = 200,
    seed: int = 42,
    n_seeds: int = 25,
) -> np.ndarray:
    """SSWD of one subset, recomputed with `n_seeds` different projection seeds.

    Seeds are `seed, seed + 1, ..., seed + n_seeds - 1`; within one repetition all
    seasons share the same seed, exactly as in `compute_all_metrics`.

    Returns
    -------
    np.ndarray
        One SSWD value (max over seasons) per seed.
    """
    return np.array(
        [
            max(
                _compute_swd(
                    X_sub_seasonal[season_name],
                    X_ref_seasonal[season_name],
                    n_projections,
                    seed + i,
                )
                for season_name in X_ref_seasonal
            )
            for i in range(n_seeds)
        ]
    )


def sswd_seed_uncertainty(
    df_results: pd.DataFrame,
    df_ref: pd.DataFrame,
    vars_non_nan: list,
    seasons: dict,
    score_col: str = "wasserstein_seasonal",
    n_projections: int = 200,
    seed: int = 42,
    n_seeds: int = 25,
) -> tuple:
    """Uncertainty of the best and median SSWD score, per selection_source.

    A single SSWD value carries Monte-Carlo noise from the random projections. Here
    the best- and median-ranked selection of every `selection_source` is re-scored
    with `n_seeds` different seeds, so the spread of those repetitions quantifies
    that noise. The selections themselves are fixed (picked on the stored
    single-seed scores), so the spread is scoring noise only, not selection noise.

    Two caveats when reporting the std as an error bar:

    1. It is a *marginal* std. All subsets in one repetition share a seed, hence the
       same projection directions, so a harsh draw lifts every method at once
       (measured correlation across sources: 0.77-0.94). The noise therefore largely
       cancels in a method-to-method comparison, and the marginal std overstates the
       uncertainty on a *difference* by ~2-3x. Use the per-seed arrays in `raw` for a
       paired comparison if a difference has to be defended.
    2. It shrinks as ~1/sqrt(n_projections) — it measures how precisely SSWD was
       computed, not a property of the data. Do not read it as a statistical
       significance statement about the methods; the spread across the experiments in
       `df_results` (see the luck/misfortune factors) is the relevant one for that.

    Returns
    -------
    df_summary : DataFrame
        Indexed by selection_source, with mean/std of the best and median selection
        and the stored single-seed scores for reference.
    raw : dict
        {(selection_source, "best" | "median"): array of per-seed SSWD values}.
    """
    X_ref_seasonal = _seasonal_arrays(df_ref, vars_non_nan, seasons)

    rows, raw = [], {}
    for source, df_source in df_results.groupby("selection_source"):
        df_source = df_source.sort_values(score_col)
        scores = df_source[score_col].to_numpy()
        # closest-to-median row, so an even number of rows also maps onto a real selection
        picks = {
            "best": df_source.iloc[0],
            "median": df_source.iloc[
                int(np.argmin(np.abs(scores - np.median(scores))))
            ],
        }

        row = {
            "selection_source": source,
            "n_selections": len(df_source),
            "n_seeds": n_seeds,
            "n_projections": n_projections,
        }
        for kind, pick in picks.items():
            modelyears = pick["model_years"]
            if isinstance(modelyears, str):
                modelyears = literal_eval(modelyears)
            df_sub = build_df_from_modelyears(df_ref, modelyears)
            values = sswd_seed_ensemble(
                _seasonal_arrays(df_sub, vars_non_nan, seasons),
                X_ref_seasonal,
                n_projections=n_projections,
                seed=seed,
                n_seeds=n_seeds,
            )
            raw[(source, kind)] = values
            row[f"{kind}_stored"] = float(pick[score_col])
            row[f"{kind}_mean"] = float(np.mean(values))
            row[f"{kind}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
        print(f"  {source}: {n_seeds} seeds × (best, median) done ✓")

    df_summary = pd.DataFrame(rows).set_index("selection_source")
    return df_summary, raw


def format_mean_std(df_summary: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Render `sswd_seed_uncertainty` output as "mean ± std" strings for printing."""
    return pd.DataFrame(
        {
            kind: df_summary[f"{kind}_mean"].round(decimals).astype(str)
            + " ± "
            + df_summary[f"{kind}_std"].round(decimals).astype(str)
            for kind in ("best", "median")
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Score tables, for plotting and tables in paper
# ─────────────────────────────────────────────────────────────────────────────


def build_score_tables(metrics: dict, active_subsets: dict):
    """Build per-season detail tables and a worst-season summary table.

    Returns
    -------
    seasonal_dfs : dict
        {season_name: DataFrame} with per-season scores.
    df_scores_summary : DataFrame
        Worst-season summary across all methods.
    """
    # Per-season detail tables
    seasonal_dfs = {}
    for season_name in metrics["season_list"]:
        rows = [
            {
                "subset": name,
                "mean_qq_mae": float(
                    np.mean(
                        list(metrics["qq_mae_seasonal"][season_name][name].values())
                    )
                ),
                "max_qq_mae": float(
                    np.max(list(metrics["qq_mae_seasonal"][season_name][name].values()))
                ),
                "swd_seasonal": metrics["swd_seasonal"][season_name][name],
                "mean_corr_dev": metrics["corr_dev"][name][season_name],
            }
            for name in active_subsets
        ]
        df_s = pd.DataFrame(rows).sort_values("swd_seasonal").reset_index(drop=True)
        df_s[["mean_qq_mae", "max_qq_mae", "swd_seasonal", "mean_corr_dev"]] = df_s[
            ["mean_qq_mae", "max_qq_mae", "swd_seasonal", "mean_corr_dev"]
        ].round(3)
        seasonal_dfs[season_name] = df_s

    # Worst-season summary
    rows = [
        {
            "subset": name,
            "swd_annual": metrics["swd_annual"][name],
            "sswd": metrics["sswd"][name],
            **{
                f"swd_{s}": metrics["swd_seasonal"][s][name]
                for s in metrics["season_list"]
            },
            "worst_mean_qq_mae": max(
                np.mean(list(metrics["qq_mae_seasonal"][s][name].values()))
                for s in metrics["season_list"]
            ),
            "worst_mean_corr_dev": max(
                metrics["corr_dev"][name][s] for s in metrics["season_list"]
            ),
        }
        for name in active_subsets
    ]
    df_summary = pd.DataFrame(rows).sort_values("sswd").reset_index(drop=True)
    numeric_cols = df_summary.select_dtypes(include="number").columns
    df_summary[numeric_cols] = df_summary[numeric_cols].round(3)

    return seasonal_dfs, df_summary


def plot_lollipop_by_country(
    df_results, df_country_results, countries, metric, metric_label, save_dir=None
):
    """Plot lollipop chart of results by country. Used for auto-evaluation in 02_run_selections.py"""
    # take 5 best SA selections
    idx_best = df_results.sort_values(metric).index[:5]
    df_best = df_results.loc[idx_best]

    # Color palette for SA markers
    sa_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    y = np.arange(len(countries))
    fig, ax = plt.subplots(figsize=(4, 6))

    for i, country in enumerate(countries):
        # HACK: Some messing around to ensure model_years are strings for merging
        # TODO: implement proper handling of model_years as tuples everywhere
        df_country = df_country_results.query("country == @country").copy()
        df_country["model_years"] = df_country["model_years"].apply(str)

        # Also convert df_best model_years to string for consistent merging
        df_best_str = df_best.copy()
        df_best_str["model_years"] = df_best_str["model_years"].apply(str)

        wass_country_best = df_best_str.merge(
            df_country[["model_years", metric]], on="model_years"
        )[f"{metric}_y"].to_numpy()

        rnd = df_country.query("selection_source == 'random_draws'")[f"{metric}"].mean()

        k_med = df_country.query("selection_source == 'k_medoids'")[f"{metric}"].values[
            0
        ]

        # Safety check: skip if no matching data
        if len(wass_country_best) == 0:
            print(f"Warning: No matching model_years found for country {country}")
            continue

        x_min = np.minimum(wass_country_best.min(), rnd)
        x_max = np.maximum(wass_country_best.max(), rnd)
        better = wass_country_best <= rnd
        link_colors = np.where(better, "green", "red")

        ax.hlines(i, x_min, x_max, linewidth=1, colors=link_colors, alpha=0.9)
        ax.scatter(rnd, i, color="red", label="Random Draws" if i == 0 else None)
        ax.scatter(
            k_med,
            i,
            color="C6",
            marker="*",
            label="K-Medoids" if i == 0 else None,
        )
        # Plot SA points as X markers with distinct labels/colors
        for j, val in enumerate(wass_country_best):
            label = f"SA #{j + 1}" if i == 0 else None
            ax.scatter(
                val,
                i,
                color=sa_colors[j % len(sa_colors)],
                marker="x",
                label=label,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(countries)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric_label} (lower is better)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(
                save_dir,
                f"lollipop_{metric}_by_country.png",
            ),
            dpi=150,
        )


### Physical Evaluation ###
# Automatic evaluation in 02_run_selections.py.
# Only used as diagnostic, not for final paper figures.
# TODO: Some overlap here with functions from above used in  03_evaluation_plots.py, consider merging or at least reusing code between these two files.


def compute_monthly_mean(df, variables):
    """Compute monthly means for given variables."""
    df = df.set_index("Date")
    monthly_means = df[variables].groupby(df.index.month).mean()
    monthly_means.index.name = "Month"
    return monthly_means


def compute_correlation_matrix(df, variables):
    """Compute correlation matrix for variables."""
    return df[variables].corr()


def compute_drought_statistics(df, country, ref_percentiles=None):
    """Analyze renewable droughts in the dataframe."""
    # --- Renewable droughts ---
    percentiles = {}
    stats = {}

    n_days = len(df)

    if f"{country}_total_renewable" in df.columns:
        renewables = df[f"{country}_total_renewable"].dropna()

        if ref_percentiles is None:
            p_drought = renewables.quantile(DROUGHT_PERCENTILE / 100)
            percentiles["renewables_drought_threshold"] = p_drought
        else:
            p_drought = ref_percentiles["renewables_drought_threshold"]

        # Find drought periods (consecutive days below threshold)
        below_threshold = (renewables < p_drought).astype(int)

        # Count drought events and their durations
        drought_events = []
        current_drought = 0

        for val in below_threshold:
            if val == 1:
                current_drought += 1
            else:
                if current_drought >= DROUGHT_MIN_DURATION:
                    drought_events.append(current_drought)
                current_drought = 0
        # Don't forget last event
        if current_drought >= DROUGHT_MIN_DURATION:
            drought_events.append(current_drought)

        stats["n_drought_events"] = len(drought_events) / (n_days / 365)  # Per year
        stats["mean_drought_duration"] = (
            np.mean(drought_events) if drought_events else 0
        )
        stats["max_drought_duration"] = np.max(drought_events) if drought_events else 0
        stats["total_drought_days_freq"] = (
            sum(drought_events) / n_days if drought_events else 0
        )
    return percentiles, stats


def perform_physical_evaluation(
    df_results, df_ref, countries, variables, save_dir=None
):
    """
    Perform comprehensive physical evaluation for all countries in countries.
    Creates all plots and analyses for seasonal patterns, correlations, and drought metrics.
    """

    for country in countries:
        print(f"\n{'=' * 60}")
        print(f"Analyzing country: {country}")
        print(f"{'=' * 60}\n")

        # --- Setup country-specific variables ---
        country_vars = [var for var in variables if var.startswith(country + "_")]
        country_prod_vars = [var for var in country_vars if "prod" in var]
        country_vars.append(f"{country}_total_renewable")
        df_ref[f"{country}_total_renewable"] = df_ref[
            [var for var in country_prod_vars if var != f"{country}_total_renewable"]
        ].sum(axis=1)

        # Compute reference statistics
        ref_monthly_mean = compute_monthly_mean(df_ref, country_vars)
        ref_correlation_matrix = compute_correlation_matrix(df_ref, country_vars)
        ref_percentiles, ref_drought_stats = compute_drought_statistics(
            df_ref[country_vars], country
        )

        # Compute statistics for all subsets
        dfs_monthly_means = {}
        dfs_correlation_matrices = {}
        dfs_drought_stats = {}

        for i, row in df_results.iterrows():
            if isinstance(row["model_years"], str):
                sel_modelyears = literal_eval(row["model_years"])
            else:
                sel_modelyears = row["model_years"]

            ds_name = row["selection_source"] + "_" + str(i + 1)
            df_sub = build_df_from_modelyears(df_ref, sel_modelyears)
            df_sub[f"{country}_total_renewable"] = df_sub[
                [
                    var
                    for var in country_prod_vars
                    if var != f"{country}_total_renewable"
                ]
            ].sum(axis=1)

            dfs_monthly_means[ds_name] = compute_monthly_mean(df_sub, country_vars)
            dfs_correlation_matrices[ds_name] = compute_correlation_matrix(
                df_sub, country_vars
            )
            dfs_drought_stats[ds_name] = compute_drought_statistics(
                df_sub[country_vars], country, ref_percentiles
            )[1]

        # --- 1. Monthly means analysis ---
        df_subsets_monthly_means_list = list(dfs_monthly_means.values())
        mean_across_subsets = (
            pd.concat(df_subsets_monthly_means_list).groupby(level=0).mean()
        )
        std_across_subsets = (
            pd.concat(df_subsets_monthly_means_list).groupby(level=0).std()
        )

        best_idx = df_results.sort_values("wasserstein_seasonal").index[0]
        if isinstance(df_results.loc[best_idx, "model_years"], str):
            modelyears_best = literal_eval(df_results.loc[best_idx, "model_years"])
        else:
            modelyears_best = df_results.loc[best_idx, "model_years"]

        best_ds_name = (
            df_results.loc[best_idx, "selection_source"] + "_" + str(best_idx + 1)
        )
        df_best_monthly_means = dfs_monthly_means[best_ds_name]

        sorted_indices = df_results.sort_values("wasserstein_seasonal").index
        median_idx = sorted_indices[len(sorted_indices) // 2]

        # TODO: implement proper handling of model_years as tuples everywhere
        if isinstance(df_results.loc[median_idx, "model_years"], str):
            modelyears_median = literal_eval(df_results.loc[median_idx, "model_years"])
        else:
            modelyears_median = df_results.loc[median_idx, "model_years"]

        median_ds_name = (
            df_results.loc[median_idx, "selection_source"] + "_" + str(median_idx + 1)
        )
        df_median_monthly_means = dfs_monthly_means[median_ds_name]

        # Plot monthly means
        plot_vars = country_vars
        n_vars = len(plot_vars)
        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()
        month_positions = np.arange(len(ref_monthly_mean))

        for idx, var in enumerate(plot_vars):
            ax = axes[idx]
            ref_mean = ref_monthly_mean[var]
            subset_mean = mean_across_subsets[var]
            subset_std = std_across_subsets[var]
            best_mean = df_best_monthly_means[var]
            median_mean = df_median_monthly_means[var]

            ax.plot(
                month_positions,
                ref_mean.values,
                label="Reference",
                linewidth=2.5,
                color="black",
                marker="o",
                markersize=6,
            )
            ax.fill_between(
                month_positions,
                subset_mean - subset_std,
                subset_mean + subset_std,
                alpha=0.4,
                color="gray",
                label="Subset mean ± std",
            )
            ax.plot(
                month_positions,
                subset_mean.values,
                linewidth=2,
                color="gray",
                linestyle="--",
            )
            ax.plot(
                month_positions,
                median_mean.values,
                label="Median subset",
                linewidth=2.5,
                color="C1",
                marker="^",
                markersize=6,
            )
            ax.plot(
                month_positions,
                best_mean.values,
                label="Best subset",
                linewidth=2.5,
                color="C0",
                marker="s",
                markersize=6,
            )

            ax.set_xticks(month_positions)
            ax.set_xticklabels(ref_monthly_mean.index, rotation=45)
            ax.set_xlabel("Month")
            ax.set_ylabel(var)
            ax.set_title(f"{var}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        for idx in range(n_vars, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Monthly Mean Comparison - {country}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

        if save_dir:
            fig.savefig(
                os.path.join(save_dir, f"{country}_monthly_means.png"),
                bbox_inches="tight",
                dpi=150,
            )

        # --- 2. Correlation analysis ---
        ref_corr_country = compute_correlation_matrix(df_ref, country_vars)
        sub_best_corr_country = compute_correlation_matrix(
            build_df_from_modelyears(df_ref, modelyears_best),
            country_vars,
        )
        sub_median_corr_country = compute_correlation_matrix(
            build_df_from_modelyears(df_ref, modelyears_median),
            country_vars,
        )

        diff_best_country = sub_best_corr_country - ref_corr_country
        diff_median_country = sub_median_corr_country - ref_corr_country

        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        sns.heatmap(
            ref_corr_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[0, 0],
            cbar=False,
        )
        axs[0, 0].set_title("Reference")

        sns.heatmap(
            sub_best_corr_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[0, 1],
            cbar=False,
        )
        axs[0, 1].set_title("Best Subset")

        sns.heatmap(
            diff_best_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Difference"},
            vmin=-0.1,
            vmax=0.1,
            ax=axs[0, 2],
            cbar=False,
        )
        axs[0, 2].set_title("Difference (Best - Ref)")

        sns.heatmap(
            ref_corr_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[1, 0],
            cbar=False,
        )
        axs[1, 0].set_title("Reference")

        sns.heatmap(
            sub_median_corr_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[1, 1],
            cbar=False,
        )
        axs[1, 1].set_title("Median Subset")

        sns.heatmap(
            diff_median_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Difference"},
            vmin=-0.1,
            vmax=0.1,
            ax=axs[1, 2],
            cbar=False,
        )
        axs[1, 2].set_title("Difference (Median - Ref)")

        plt.suptitle(
            f"Correlation Matrices - {country}", fontsize=14, fontweight="bold", y=0.995
        )
        plt.tight_layout()
        plt.show()
        if save_dir:
            fig.savefig(
                os.path.join(save_dir, f"{country}_correlation_matrices.png"),
                bbox_inches="tight",
                dpi=150,
            )

        # Mean and std of correlations
        corr_matrices_list = list(dfs_correlation_matrices.values())
        corr_array = np.array([df.values for df in corr_matrices_list])
        mean_corr = pd.DataFrame(
            corr_array.mean(axis=0),
            index=corr_matrices_list[0].index,
            columns=corr_matrices_list[0].columns,
        )
        std_corr = pd.DataFrame(
            corr_array.std(axis=0),
            index=corr_matrices_list[0].index,
            columns=corr_matrices_list[0].columns,
        )

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(
            mean_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Mean Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[0],
            cbar=False,
        )
        axs[0].set_title("Mean Correlation Across All Subsets")

        sns.heatmap(
            std_corr,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": "Std of Correlation"},
            ax=axs[1],
            cbar=False,
        )
        axs[1].set_title("Std Dev of Correlation Across All Subsets")

        plt.suptitle(
            f"Correlation Variability - {country}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()
        if save_dir:
            fig.savefig(
                os.path.join(save_dir, f"{country}_correlation_variability.png"),
                bbox_inches="tight",
                dpi=150,
            )

        # --- 3. Energy drought metrics ---
        top10_indices = df_results.sort_values("wasserstein_seasonal").head(10).index
        top10_names = [
            df_results.loc[idx, "selection_source"] + "_" + str(idx + 1)
            for idx in top10_indices
        ]

        labels = ["Reference"] + top10_names + [median_ds_name]
        n_drought_events = [ref_drought_stats["n_drought_events"]]
        max_drought_duration = [ref_drought_stats["max_drought_duration"]]
        mean_drought_duration = [ref_drought_stats["mean_drought_duration"]]

        for ds_name in top10_names:
            n_drought_events.append(dfs_drought_stats[ds_name]["n_drought_events"])
            max_drought_duration.append(
                dfs_drought_stats[ds_name]["max_drought_duration"]
            )
            mean_drought_duration.append(
                dfs_drought_stats[ds_name]["mean_drought_duration"]
            )

        n_drought_events.append(dfs_drought_stats[median_ds_name]["n_drought_events"])
        max_drought_duration.append(
            dfs_drought_stats[median_ds_name]["max_drought_duration"]
        )
        mean_drought_duration.append(
            dfs_drought_stats[median_ds_name]["mean_drought_duration"]
        )

        fig, axs = plt.subplots(3, 1, figsize=(14, 12))
        x_pos = np.arange(len(labels))
        colors = ["black"] + ["green"] * len(top10_names) + ["blue"]

        axs[0].bar(x_pos, n_drought_events, color=colors, alpha=0.7)
        axs[0].axhline(
            n_drought_events[0],
            color="black",
            linestyle=":",
            linewidth=2,
            label="Reference",
        )
        axs[0].set_xticks(x_pos)
        axs[0].set_xticklabels("")
        axs[0].set_ylabel("Drought Events per Year")
        axs[0].set_title("Number of Drought Events per Year")
        axs[0].grid(axis="y", alpha=0.3)

        axs[1].bar(x_pos, max_drought_duration, color=colors, alpha=0.7)
        axs[1].axhline(
            max_drought_duration[0],
            color="black",
            linestyle=":",
            linewidth=2,
            label="Reference",
        )
        axs[1].set_xticks(x_pos)
        axs[1].set_xticklabels("")
        axs[1].set_ylabel("Max Drought Duration (days)")
        axs[1].set_title("Maximum Drought Duration")
        axs[1].grid(axis="y", alpha=0.3)

        axs[2].bar(x_pos, mean_drought_duration, color=colors, alpha=0.7)
        axs[2].axhline(
            mean_drought_duration[0],
            color="black",
            linestyle=":",
            linewidth=2,
            label="Reference",
        )
        axs[2].set_xticks(x_pos)
        axs[2].set_xticklabels(labels, rotation=45, ha="right")
        axs[2].set_ylabel("Mean Drought Duration (days)")
        axs[2].set_title("Mean Drought Duration")
        axs[2].grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Energy Drought Metrics - {country} (Reference=black, Top 10 Best=green, Median=blue)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        if save_dir:
            fig.savefig(
                os.path.join(save_dir, f"{country}_drought_metrics.png"),
                bbox_inches="tight",
                dpi=150,
            )
