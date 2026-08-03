"""
NOTE: This module contains cost functions for optimization. Evaluation scores are in evaluation_scores.py
"""

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from ot import sliced_wasserstein_distance

from cluster_weather_years.constants import SEASONS


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

    # NOTE: hardcoded
    # set with fixed seed, so that projection is constant over optimisation
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

        # set with fixed seed, so that projection is constant over optimisation
        wasserstein_seasonal.append(
            sliced_wasserstein_distance(
                sub_data, full_data, n_projections=n_projections, seed=42
            )
        )
    # base score on worst-performing season
    return max(wasserstein_seasonal)
