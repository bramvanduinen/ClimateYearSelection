import warnings
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ot import sliced_wasserstein_distance
from scipy.stats import anderson_ksamp, ks_2samp

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

    vars_non_nan = [var for var in variables if not df[var].isna().all()]
    sub_data = df.loc[idx_sel, vars_non_nan].to_numpy()
    full_data = df[vars_non_nan].to_numpy()

    # Replace NaNs with zeros to avoid issues in distance calculation
    # sub_data[np.isnan(sub_data)] = 0
    # full_data[np.isnan(full_data)] = 0

    # 50 projections is compromise between accuracy and speed. Better would be: 200
    return sliced_wasserstein_distance(sub_data, full_data, n_projections=n_projections)


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
    vars_non_nan = [var for var in variables if not df[var].isna().all()]

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
                sub_data, full_data, n_projections=n_projections
            )
        )
    # base score on worst-performing season
    return max(wasserstein_seasonal)


# TODO: df-based, might want to rewrite this later
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


#### Below: unused, might be deleted later ####


def ks_total(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Unweighted Kolmogorov–Smirnov score (lower ⇒ closer)."""
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    sub = df.loc[idx_sel, variables]
    return float(
        sum(
            ks_2samp(df[v].to_numpy(), sub[v].to_numpy(), method="auto").statistic
            for v in variables
        )
    )


def _ad_stat(a: np.ndarray, b: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"p-value (?:capped|floored).*",
            category=UserWarning,
        )
        return anderson_ksamp([a, b]).statistic


def ad_total(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Anderson–Darling k-sample score (lower ⇒ closer; tail-weighted)."""
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    sub = df.loc[idx_sel, variables]
    full_disjoint = df.loc[~df.index.isin(sub.index), variables]
    return float(
        sum(_ad_stat(full_disjoint[v].to_numpy(), sub[v].to_numpy()) for v in variables)
    )


def ad_seasonal(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "model",
    year_col: str = "year",
    date_col: str = "Date",
) -> float:
    """
    Compute Anderson–Darling k-sample statistics for each season separately.

    Args:
        df: Input DataFrame with datetime column.
        combos: Tuples of (model, year) for selection.
        variables: Variables to evaluate.
        model_col: Column with model identifiers.
        year_col: Column with year identifiers.
        date_col: Column with datetime values.

    Returns:
        AD Scores summed over variable and seasons.
    """
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    sub = df.loc[idx_sel, [date_col] + variables]
    full_disjoint = df.loc[~df.index.isin(sub.index), [date_col] + variables]

    # Ensure datetime
    sub[date_col] = pd.to_datetime(sub[date_col])
    full_disjoint[date_col] = pd.to_datetime(full_disjoint[date_col])

    # Add seasons
    sub["season"] = sub[date_col].dt.month % 12 // 3 + 1
    full_disjoint["season"] = full_disjoint[date_col].dt.month % 12 // 3 + 1
    season_map = {1: "DJF", 2: "MAM", 3: "JJA", 4: "SON"}

    result = 0

    # Seasonal distributions
    for v in variables:
        for s_code, s_name in season_map.items():
            sub_s = sub.loc[sub["season"] == s_code, v].to_numpy()
            full_s = full_disjoint.loc[full_disjoint["season"] == s_code, v].to_numpy()
            if len(sub_s) > 0 and len(full_s) > 0:
                result += float(_ad_stat(full_s, sub_s))

    return result


# %% Visualization functions


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
