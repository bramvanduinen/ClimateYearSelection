"""
NOTE: This module contains cost functions for optimization. Evaluation scores are in evaluation_scores.py
"""

from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ot import sliced_wasserstein_distance

from cluster_weather_years.constants import PROBS, SEASONS


# %% SA Scoring functions
def build_selection_mask(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    model_col: str = "model",
    year_col: str = "year",
) -> np.ndarray:
    """Return a boolean mask selecting rows whose (model, year) is in combos."""
    mi = pd.MultiIndex.from_arrays([df[model_col].to_numpy(), df[year_col].to_numpy()])
    return mi.isin(set(combos))  # .to_numpy()


def wasserstein_score(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "model",
    year_col: str = "year",
    n_projections: int = 50,
) -> float:
    """Wasserstein score: Wasserstein distance between full sample and selected years."""
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)

    # HACK: Hydro contains 0 instead of nans, so there have to be non-zero values.
    vars_non_nan = [var for var in variables if np.max(df[var]) > 0]

    sub_data = df.loc[idx_sel, vars_non_nan].to_numpy()
    full_data = df[vars_non_nan].to_numpy()

    # Replace NaNs with zeros to avoid issues in distance calculation
    # sub_data[np.isnan(sub_data)] = 0
    # full_data[np.isnan(full_data)] = 0

    # 50 projections is compromise between accuracy and speed. Better would be: 200
    return sliced_wasserstein_distance(
        sub_data, full_data, n_projections=n_projections, seed=42
    )


def wasserstein_seasonal_score(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "model",
    year_col: str = "year",
    n_projections: int = 50,
) -> float:
    """Wasserstein score: Wasserstein distance between full sample and selected years."""

    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    vars_non_nan = [var for var in variables if np.max(df[var]) > 0]

    wasserstein_seasonal = []
    for season, months in SEASONS.items():
        in_season = df["Date"].dt.month.isin(months).values
        # combine random draw boolean & seasonal boolean for subset
        sub_mask = idx_sel & in_season
        sub_data = df.loc[sub_mask, vars_non_nan].to_numpy()
        # only use seasonal boolean for full set
        full_data = df.loc[in_season, vars_non_nan].to_numpy()

        # Replace NaNs with zeros to avoid issues in distance calculation
        # sub_data[np.isnan(sub_data)] = 0
        # full_data[np.isnan(full_data)] = 0

        # 50 projections is compromise between accuracy and speed. Better would be: 200
        wasserstein_seasonal.append(
            sliced_wasserstein_distance(
                sub_data, full_data, n_projections=n_projections, seed=42
            )
        )
    # base score on worst-performing season
    return max(wasserstein_seasonal)


# %%


# TODO: df-based, might want to rewrite this later
# NOTE: Unused I think, so can probably be deleted.
def qq_univariate_rmse(df_daily, datasets, columns):
    """
    Compute QQ-based metrics for one or many datasets vs the full df_daily.

    Parameters
    - df_daily: reference dataframe (full climatology)
    - datasets: dict[str, pd.DataFrame], e.g. {"Centroid": df_centroids, "Random": df_random}
    - columns: list[str] variables to analyze

    Returns
    - pd.DataFrame with rows per (dataset, variable) and columns [MAE, RMSE, MaxAbs]
    """

    rows = []
    for col in columns:
        q_full = df_daily[col].quantile(PROBS)
        for name, df in datasets.items():
            q_sub = df[col].quantile(PROBS)
            diff = (q_sub - q_full).values
            rows.append(
                {
                    "dataset": name,
                    "variable": col,
                    "RMSE": float(np.sqrt(np.average(diff**2))),
                }
            )
    return pd.DataFrame(rows)


# %% Visualization functions
# NOTE: Unused currently, superseded by evaluation_plot_helpers.py


def ecdf_comparison(df_daily, datasets, columns):
    """Simple ECDF plots for visual inspection"""

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    for i, col in enumerate(columns):
        ax = axs.flat[i]
        for j, (name, df) in enumerate(datasets.items()):
            sns.ecdfplot(df, x=col, label=name, ax=ax)
        sns.ecdfplot(df_daily, x=col, label="Full climatology", ax=ax)
        ax.set_title(col)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")

    # Hide any unused axes (in case columns < 4)
    for j in range(len(columns), 4):
        axs.flat[j].axis("off")


def qq_plot(df_daily, datasets, columns):
    """
    Plot QQ curves and quantile differences for provided datasets vs full df.

    Parameters
    - df_daily: reference dataframe (full climatology)
    - datasets: dict[str, pd.DataFrame], e.g. {"Centroid": df_centroids, "Random": df_random}
    - columns: list[str]
    """
    fig_qq, axs_qq = plt.subplots(2, 2, figsize=(12, 10))
    fig_diff, axs_diff = plt.subplots(2, 2, figsize=(12, 10), sharey=False)

    for i, col in enumerate(columns):
        q_full = df_daily[col].quantile(PROBS)
        x = q_full.values

        # QQ plot
        axq = axs_qq.flat[i]
        for j, (name, df) in enumerate(datasets.items()):
            q_sub = df[col].quantile(PROBS)
            axq.plot(x, q_sub.values, label=name)
        lo, hi = np.nanmin(x), np.nanmax(x)
        axq.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        axq.set_title(f"QQ: {col}")
        axq.set_xlabel("Full climatology value")
        axq.set_ylabel("Subset value")
        axq.legend(loc="best")

        # Differences
        axd = axs_diff.flat[i]
        for j, (name, df) in enumerate(datasets.items()):
            q_sub = df[col].quantile(PROBS)
            diff = (q_sub - q_full).values
            axd.plot(PROBS, diff, color=f"C{j}", label=f"{name} - Full")
            axd.plot(PROBS, np.abs(diff), color=f"C{j}", ls="--")
            axd.axhline(np.mean(np.abs(diff)), color=f"C{j}")
        axd.axhline(0.0, color="k", lw=1, ls="--")
        axd.set_title(f"QQ-differences: {col}")
        axd.set_xlabel("Quantile")
        axd.set_ylabel("Δ value")
        axd.legend(loc="best")

    for j in range(len(columns), 4):
        axs_qq.flat[j].axis("off")
        axs_diff.flat[j].axis("off")

    fig_qq.tight_layout()
    fig_diff.tight_layout()

    return fig_qq, fig_diff
