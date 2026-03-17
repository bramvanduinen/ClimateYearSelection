# %%
import numpy as np
from ot import sliced_wasserstein_distance
from scipy.stats import energy_distance

from cluster_weather_years.constants import (
    PROBS,
    SEASONS,
)
from cluster_weather_years.cost_functions import wasserstein_seasonal_score
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


# NOTE: OPTIONAL, CURRENTLY NOT USED AND NOT EXTENSIVELY TESTED
def calibrate_percentile_threshold(
    d_means,
    e_scores,
    columns,
    target_rate,
    max_iter=30,
    tol=0.20,
):
    """
    Find a percentile p such that the joint pass-rate (all vars pass mean+energy)
    is ~ target_rate. Returns (p, thresholds_mean, thresholds_energy, est_rate).

    tol is relative tolerance on target_rate.
    """
    lo, hi = 0.0, 100.0
    best = None

    for _ in range(max_iter):
        p = 0.5 * (lo + hi)

        th_mean = {v: np.percentile(d_means[v], p) for v in columns}
        th_energy = {v: np.percentile(e_scores[v], p) for v in columns}

        passes = np.ones_like(d_means[columns[0]], dtype=bool)
        for v in columns:
            passes &= (d_means[v] <= th_mean[v]) & (e_scores[v] <= th_energy[v])

        est_rate = float(passes.mean())
        best = (p, th_mean, th_energy, est_rate)

        if est_rate > target_rate:
            hi = p
        else:
            lo = p

        if target_rate > 0 and abs(est_rate - target_rate) / target_rate <= tol:
            break

    return best


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
    """Score a datasets against reference data for tuning the thresholds to a given percentile."""
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
    # NOTE: currently return d_means and energy_distance score, for testing
    return d_means, energy_distance_score, thresholds_mean, thresholds_energy


def score_tuning_seasonal(
    df_ref,
    model_years,
    tune_samples,
    num_to_select,
    columns,
    rng,
    year_col,
    percentile_threshold=25,
):
    """Like score_tuning, but d_mean and energy_distance are computed per season;
    only the worst-season score is kept per variable per sample."""
    d_means, energy_distance_score = {}, {}

    for i in range(tune_samples):
        df_sub, _ = build_random_df(df_ref, model_years, num_to_select, rng, year_col)
        for var in columns:
            if d_means.get(var) is None:
                d_means[var] = []
            if energy_distance_score.get(var) is None:
                energy_distance_score[var] = []

            seasonal_d_means = []
            seasonal_energy_scores = []
            for season, months in SEASONS.items():
                in_season_ref = df_ref["Date"].dt.month.isin(months).values
                in_season_sub = df_sub["Date"].dt.month.isin(months).values
                ref_arr = df_ref.loc[in_season_ref, var].to_numpy()
                sub_arr = df_sub.loc[in_season_sub, var].to_numpy()
                if len(sub_arr) == 0 or len(ref_arr) == 0:
                    continue
                seasonal_d_means.append(abs(sub_arr.mean() - ref_arr.mean()))
                seasonal_energy_scores.append(float(energy_distance(sub_arr, ref_arr)))

            d_means[var].append(max(seasonal_d_means))
            energy_distance_score[var].append(max(seasonal_energy_scores))

    thresholds_mean = {
        var: np.percentile(d_means[var], percentile_threshold) for var in columns
    }
    thresholds_energy = {
        var: np.percentile(energy_distance_score[var], percentile_threshold)
        for var in columns
    }
    return d_means, energy_distance_score, thresholds_mean, thresholds_energy


def score_df(
    df_ref,
    df_sub,
    modelyears_sub,
    df_name,
    thresholds_mean,
    thresholds_energy,
    columns,
    n_projections=200,
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
                "model_years": modelyears_sub,
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
                "model_years": modelyears_sub,
            }

        # ---- expensive scores (only if cheap passed) ----

        q_sub = np.quantile(sub_arr, PROBS)
        q_ref = ref_quantiles[var]
        qq_diff = abs(q_sub - q_ref)

        qq_mae.append(float(qq_diff.mean()))
        qq_mae_high.append(float(qq_diff[:10].mean()))
        qq_mae_low.append(float(qq_diff[-10:].mean()))

    vars_non_nan = [var for var in columns if not df_ref[var].isna().all()]
    wasserstein = sliced_wasserstein_distance(
        df_sub[vars_non_nan].to_numpy(),
        df_ref[vars_non_nan].to_numpy(),
        n_projections=n_projections,
        seed=42,
    )

    # wasserstein_seasonal
    wasserstein_seasonal = wasserstein_seasonal_score(
        df_ref,
        modelyears_sub,
        vars_non_nan,
        model_col="model",
        year_col="Hydrological_year",
        n_projections=n_projections,
    )

    return True, {
        "ds_name": df_name,
        "passed": True,
        "model_years": modelyears_sub,
        "d_mean": d_means,
        "energy_distance_score": energy_scores,
        "qq_mae": qq_mae,
        "qq_mae_high": qq_mae_high,
        "qq_mae_low": qq_mae_low,
        "wasserstein": wasserstein,
        "wasserstein_seasonal": wasserstein_seasonal,
    }


def score_df_seasonal(
    df_ref,
    df_sub,
    modelyears_sub,
    df_name,
    thresholds_mean,
    thresholds_energy,
    columns,
    n_projections=200,
    skip_check=True,
):
    """
    Like score_df, but every metric is computed per season; only the worst-season
    score is kept per variable.  The output schema is identical to score_df.
    """
    d_means, energy_scores = [], []
    qq_mae, qq_mae_high, qq_mae_low = [], [], []

    # Pre-compute season masks once
    season_masks_ref = {
        season: df_ref["Date"].dt.month.isin(months).values
        for season, months in SEASONS.items()
    }
    season_masks_sub = {
        season: df_sub["Date"].dt.month.isin(months).values
        for season, months in SEASONS.items()
    }

    # ---- cheap checks (early exit) ----
    for var in columns:
        seasonal_d_means = []
        seasonal_energy_scores = []
        for season in SEASONS:
            ref_arr = df_ref.loc[season_masks_ref[season], var].to_numpy()
            sub_arr = df_sub.loc[season_masks_sub[season], var].to_numpy()
            if len(sub_arr) == 0 or len(ref_arr) == 0:
                continue
            seasonal_d_means.append(abs(sub_arr.mean() - ref_arr.mean()))
            seasonal_energy_scores.append(float(energy_distance(sub_arr, ref_arr)))

        d_mean = max(seasonal_d_means)
        d_means.append(float(d_mean))

        if skip_check and d_mean > thresholds_mean[var]:
            return False, {
                "ds_name": df_name,
                "passed": False,
                "fail_check": "mean",
                "fail_var": var,
                "d_mean": float(d_mean),
                "threshold_mean": float(thresholds_mean[var]),
                "model_years": modelyears_sub,
            }

        energy_score = max(seasonal_energy_scores)
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
                "model_years": modelyears_sub,
            }

        # ---- expensive scores (only if cheap checks passed) ----
        seasonal_qq_mae, seasonal_qq_mae_high, seasonal_qq_mae_low = [], [], []
        for season in SEASONS:
            ref_arr = df_ref.loc[season_masks_ref[season], var].to_numpy()
            sub_arr = df_sub.loc[season_masks_sub[season], var].to_numpy()
            if len(sub_arr) == 0 or len(ref_arr) == 0:
                continue
            q_sub = np.quantile(sub_arr, PROBS)
            q_ref = np.quantile(ref_arr, PROBS)
            qq_diff = abs(q_sub - q_ref)
            seasonal_qq_mae.append(float(qq_diff.mean()))
            seasonal_qq_mae_high.append(float(qq_diff[:10].mean()))
            seasonal_qq_mae_low.append(float(qq_diff[-10:].mean()))

        qq_mae.append(max(seasonal_qq_mae))
        qq_mae_high.append(max(seasonal_qq_mae_high))
        qq_mae_low.append(max(seasonal_qq_mae_low))

    vars_non_nan = [var for var in columns if not df_ref[var].isna().all()]

    # for wasserstein keep the normal score, to compare with wasserstein_seasonal!
    wasserstein = sliced_wasserstein_distance(
        df_sub[vars_non_nan].to_numpy(),
        df_ref[vars_non_nan].to_numpy(),
        n_projections=n_projections,
        seed=42,
    )

    wasserstein_seasonal = wasserstein_seasonal_score(
        df_ref,
        modelyears_sub,
        vars_non_nan,
        model_col="model",
        year_col="Hydrological_year",
        n_projections=n_projections,
    )

    return True, {
        "ds_name": df_name,
        "passed": True,
        "model_years": modelyears_sub,
        "d_mean_seasonal": d_means,
        "energy_distance_score_seasonal": energy_scores,
        "qq_mae_seasonal": qq_mae,
        "qq_mae_high_seasonal": qq_mae_high,
        "qq_mae_low_seasonal": qq_mae_low,
        "wasserstein": wasserstein,
        "wasserstein_seasonal": wasserstein_seasonal,
    }


# %% Functions for parallel execution

# ---- globals for worker processes (set once via initializer) ----
_G = {}


def init_worker(
    df_ref,
    model_years,
    thresholds_mean,
    thresholds_energy,
    num_to_select,
    year_col,
    columns,
    skip_check,
):
    """Store read-only config in worker-global dict (avoids passing every task)."""
    _G["df_ref"] = df_ref
    _G["model_years"] = model_years
    _G["thresholds_mean"] = thresholds_mean
    _G["thresholds_energy"] = thresholds_energy
    _G["num_to_select"] = num_to_select
    _G["year_col"] = year_col
    _G["columns"] = columns
    _G["skip_check"] = skip_check


def run_batch(batch_idx, batch_size, iterations, seed, seasonal=False):
    """Run one batch and return (pass_records, fail_records, pass_modelyears)."""

    rng_batch = np.random.default_rng(seed)
    start = batch_idx * batch_size
    stop = min(start + batch_size, iterations)

    pass_records = []
    fail_records = []
    pass_modelyears = []

    for i in range(start, stop):
        df_random, drawn_model_years = build_random_df(
            _G["df_ref"],
            _G["model_years"],
            _G["num_to_select"],
            rng_batch,
            _G["year_col"],
        )

        if seasonal:
            passed, record = score_df_seasonal(
                _G["df_ref"],
                df_random,
                drawn_model_years,
                i,
                _G["thresholds_mean"],
                _G["thresholds_energy"],
                columns=_G["columns"],
                skip_check=_G["skip_check"],
            )
        else:
            passed, record = score_df(
                _G["df_ref"],
                df_random,
                drawn_model_years,
                i,
                _G["thresholds_mean"],
                _G["thresholds_energy"],
                columns=_G["columns"],
                skip_check=_G["skip_check"],
            )

        if passed:
            pass_records.append(record)
            pass_modelyears.append(drawn_model_years)
        else:
            fail_records.append(record)

    return pass_records, fail_records, pass_modelyears
