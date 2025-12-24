# %%
import itertools
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from math import comb

import numpy as np
import pandas as pd
from ot import sliced_wasserstein_distance
from scipy.stats import energy_distance
from tqdm import tqdm

from cluster_weather_years.constants import (
    BASE_OUTPUT,
    HYDROLOGICAL_YEARS,
    MODELS,
    PROBS,
    SEASONS,
)
from cluster_weather_years.helper_functions import (
    build_df_from_modelyears,
    draw_random_modelyears,
)


def build_random_df(df_ref, model_years, num_to_select, rng, year_col):
    """Build a random dataframe from model years."""
    drawn_model_years = draw_random_modelyears(model_years, 1, num_to_select, rng)[0]
    df_random = build_df_from_modelyears(df_ref, drawn_model_years, mode=year_col)
    return df_random, drawn_model_years


# TODO: can probably all go in one dictionary
def reference_statistics(df_daily, columns):
    """Compute reference statistics for scoring."""
    ref = {}
    ref_mean = {}
    ref_quantiles = {}
    for var in columns:
        ref_arr = df_daily[var].to_numpy()
        ref[var] = ref_arr
        ref_mean[var] = ref_arr.mean()
        ref_quantiles[var] = np.quantile(ref_arr, PROBS)
    return ref, ref_mean, ref_quantiles


def score_tuning(
    df_ref,
    model_years,
    tune_samples,
    num_to_select,
    columns,
    rng,
    year_col,
    percentile_threshold=25,
):
    """Score a single dataset against reference data for tuning the thresholds to p10"""
    d_means, energy_distance_score = {}, {}
    ref, ref_mean, ref_quantiles = reference_statistics(df_ref, columns)
    for i in range(tune_samples):
        df_sub, _ = build_random_df(df_ref, model_years, num_to_select, rng, year_col)
        for var in columns:
            if d_means.get(var) is None:
                d_means[var] = []
            if energy_distance_score.get(var) is None:
                energy_distance_score[var] = []
            sub_arr = df_sub[var].to_numpy()
            # d_mean = float(abs((sub_arr.mean() - ref_mean[var]) / ref_mean[var]))
            d_mean = abs(sub_arr.mean() - ref_mean[var])
            energy_score = float(energy_distance(sub_arr, ref[var]))

            d_means[var].append(d_mean)
            energy_distance_score[var].append(energy_score)

    thresholds_mean = {
        var: np.percentile(d_means[var], percentile_threshold) for var in columns
    }
    thresholds_energy = {
        var: np.percentile(energy_distance_score[var], percentile_threshold)
        for var in columns
    }
    return thresholds_mean, thresholds_energy


def score_df(
    df_ref,
    df_sub,
    modelyears_sub,
    df_name,
    thresholds_mean,
    thresholds_energy,
    columns,
    skip_check=True,
):
    """
    Run cheap checks first (mean + energy). If passed, compute expensive scores.
    Returns:
        (passed: bool, record: dict)
    """
    d_means, energy_scores = [], []
    qq_mae, qq_mae_high, qq_mae_low = [], [], []

    ref, ref_mean, ref_quantiles = reference_statistics(df_ref, columns)

    # ---- cheap checks (early exit) ----
    for var in columns:
        sub_arr = df_sub[var].to_numpy()

        d_mean = abs(sub_arr.mean() - ref_mean[var])
        d_means.append(float(d_mean))

        if skip_check and d_mean > thresholds_mean[var]:
            return False, {
                "ds_name": df_name,
                "passed": False,
                "fail_check": "mean",
                "fail_var": var,
                "d_mean": float(d_mean),
                "threshold_mean": float(thresholds_mean[var]),
                "modelyears": modelyears_sub,
            }

        energy_score = float(energy_distance(sub_arr, ref[var]))
        energy_scores.append(float(energy_score))
        if skip_check and energy_score > thresholds_energy[var]:
            return False, {
                "ds_name": df_name,
                "passed": False,
                "fail_check": "energy",
                "fail_var": var,
                "d_mean": float(d_mean),
                "threshold_mean": float(thresholds_mean[var]),
                "energy_score": float(energy_score),
                "threshold_energy": float(thresholds_energy[var]),
                "modelyears": modelyears_sub,
            }

        # ---- expensive scores (only if cheap passed) ----

        q_sub = np.quantile(sub_arr, PROBS)
        q_ref = ref_quantiles[var]
        qq_diff = abs(q_sub - q_ref)

        qq_mae.append(float(qq_diff.mean()))
        qq_mae_high.append(float(qq_diff[:10].mean()))
        qq_mae_low.append(float(qq_diff[-10:].mean()))

    wasserstein = sliced_wasserstein_distance(
        df_sub[columns].to_numpy(),
        df_ref[columns].to_numpy(),
        n_projections=50,
    )

    return True, {
        "ds_name": df_name,
        "passed": True,
        "modelyears": modelyears_sub,
        "d_mean": d_means,
        "energy_distance_score": energy_scores,
        "qq_mae": qq_mae,
        "qq_mae_high": qq_mae_high,
        "qq_mae_low": qq_mae_low,
        "wasserstein": wasserstein,
    }
