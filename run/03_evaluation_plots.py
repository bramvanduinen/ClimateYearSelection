"""
Evaluation script for best subsets per method.
Detailed distribution, QQ, and correlation plots comparing each best subset to the reference dataset.
Also computes summary tables of all metrics.
Finally, computes the number of random draws needed to match the best subset's SWD score, as a rough "equivalent number of years" metric (optional, can be toggled off since it's a bit time-consuming).
"""

# %%
import os
from ast import literal_eval
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from climate_year_selection.constants import (
    BASE_OUTPUT,
    HYDROLOGICAL_YEARS,
    SEASONS,
)
from climate_year_selection.evaluation import (
    build_score_tables,
    compute_all_metrics,
    format_mean_std,
    sswd_seed_uncertainty,
)
from climate_year_selection.evaluation_plot_helpers import (
    _build_plot_series,
    _filter_season,
    _plot_correlation_panels,
    build_summary_config,
    make_summary_figure,
    plot_distributions,
    plot_ecdf,
    plot_equivalent_to_n_years,
    plot_qq_absolute,
    plot_qq_signed,
    plot_results_with_pareto,
    plot_seasonal_qq,
    pretty_var,
)
from climate_year_selection.helper_functions import build_df_from_modelyears

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these toggles
# ═══════════════════════════════════════════════════════════════════════
N_PROJECTIONS = 200  # number of random projections for SWD. Match this to the value used during optimization for consistency, or set higher for more accurate final scores (at the cost of longer computation time).
SEED = 42  # reproducibility seed for SWD
N_SEEDS_UNCERTAINTY = 25  # 25  # re-score best/median selections with seeds SEED..SEED+N-1 to quantify the projection noise on SSWD (set 0 to skip)
N_PROJECTIONS_UNCERTAINTY = N_PROJECTIONS  # projections for that re-scoring only. Keep equal to N_PROJECTIONS for final runs
INCLUDE_ERAA25 = True  # set False to exclude ERAA25 from all plots
MULTI_COUNTRY = (
    True  # True: panel (b) groups variables by type and sums QQ-MAE across countries
)
PLOT_EVERYTHING = False  # set False to skip distribution/QQ/correlation plots and only compute scores + summary tables (faster)
CALC_EQUIVALENT_TO_N_YEARS = False  # If True, compute how many random draws would be needed to match the best subset's SWD score (rough estimate of "equivalent number of years" for that method)
COMPARE_30_VS_36 = (
    False  # If True, compare 30yr vs 36yr random draw quality. Only had to be run once.
)
OLD_NAMING = True

RUNNAME_ = "RUNNAME"  # Enter the runname you want to evaluate here


SELECTED_YEARS = 30

# Plot layout
PLOT_ROWS, PLOT_COLS = 3, 2

SAVE_DIR = os.path.join(BASE_OUTPUT, RUNNAME_, "evaluation_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# PLOTTING STYLE
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.2,
    }
)

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def _to_modelyears(entry):
    return literal_eval(entry) if isinstance(entry, str) else entry


def _best_subset_by_source(df_results_sorted, df_ref, pattern):
    """Return (modelyears, df_subset) for the top-ranked row matching `pattern`."""
    col = df_results_sorted["selection_source"].astype(str).str.lower()
    mask = col.eq(pattern)
    if not mask.any():
        raise ValueError(f"No selection_source matching '{pattern}' in df_results")
    modelyears = _to_modelyears(df_results_sorted.loc[mask].iloc[0]["model_years"])
    return modelyears, build_df_from_modelyears(df_ref, modelyears)


def _active_subsets(all_subsets):
    """Filter ALL_SUBSETS by the INCLUDE_ERAA25 toggle."""
    if INCLUDE_ERAA25:
        return all_subsets
    return {k: v for k, v in all_subsets.items() if k != "ERAA25 Subset"}


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

# %% 1. Load data

fname_results = os.path.join(BASE_OUTPUT, RUNNAME_, "all_year_selections_scored.csv")
df_results = pd.read_csv(fname_results, index_col=0)

fname_settings = os.path.join(BASE_OUTPUT, RUNNAME_, "RUN_SETTINGS_of_run.py")
_run_settings = {}
with open(fname_settings) as f:
    exec(f.read(), _run_settings)
DF_PROD_DEMAND_FNAME = _run_settings["DF_PROD_DEMAND_FNAME"]
VARS_TO_MATCH = _run_settings["VARS_TO_MATCH"]
MATCH_COUNTRIES = _run_settings["MATCH_COUNTRIES"]

df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily[df_daily["Hydrological_year"].isin(HYDROLOGICAL_YEARS)].set_index(
    ["model", "Date"]
)

vars_non_nan = [v for v in VARS_TO_MATCH if np.max(df_daily[v]) > 0]
df_ref = df_daily[["Hydrological_year", "Calendar_year"] + vars_non_nan].reset_index()

PLOT_VARS = vars_non_nan[: PLOT_ROWS * PLOT_COLS]
PLOT_VAR_LABELS = [pretty_var(v) for v in PLOT_VARS]

# %% 2. Extract best subsets per method

df_results_sorted = df_results.sort_values("wasserstein_seasonal")

sel_my_sa, df_sub_sa = _best_subset_by_source(
    df_results_sorted, df_ref, "simulated_annealing"
)
sel_my_kmed, df_sub_kmed = _best_subset_by_source(
    df_results_sorted, df_ref, "k_medoids"
)

# HACK: for older runs where FRS was labeled "hbf" and random search was labeled "random_draws"
if OLD_NAMING:
    df_results["selection_source"] = df_results["selection_source"].replace(
        {"HBF": "frs", "random_draws": "random_search"}
    )
    df_results_sorted = df_results.sort_values("wasserstein_seasonal")

sel_my_random, df_sub_random = _best_subset_by_source(
    df_results_sorted, df_ref, "random_search"
)
sel_my_frs, df_sub_frs = _best_subset_by_source(df_results_sorted, df_ref, "frs")


eraa25_models = ["cmcc_cm2_sr5", "ec_earth3", "mpi_esm1_2_hr"]
eraa25_modelyears = tuple(product(eraa25_models, range(2025, 2037)))
df_sub_eraa25 = build_df_from_modelyears(df_ref, eraa25_modelyears)

ALL_SUBSETS = {
    "Best SA Subset": {"df": df_sub_sa, "modelyears": sel_my_sa},
    "Best K-Medoids Subset": {"df": df_sub_kmed, "modelyears": sel_my_kmed},
    "Best FRS Subset": {"df": df_sub_frs, "modelyears": sel_my_frs},
    "Best Random Search Subset": {"df": df_sub_random, "modelyears": sel_my_random},
    "ERAA25 Subset": {"df": df_sub_eraa25, "modelyears": eraa25_modelyears},
}
active_subsets = _active_subsets(ALL_SUBSETS)

# %% 3. SA run details + luck table

fname_sa = os.path.join(
    BASE_OUTPUT, RUNNAME_, "simulated_annealing/sa_year_selections.csv"
)
df_sa = pd.read_csv(fname_sa, index_col=0)
print(
    f"SA: {len(df_sa)} experiments, total iterations={df_sa['n_iter'].sum()}, "
    f"min={df_sa['n_iter'].min()}, max={df_sa['n_iter'].max()}"
)

fname_luck_df = os.path.join(BASE_OUTPUT, RUNNAME_, "evaluation/luck_analysis.csv")
df_luck = pd.read_csv(fname_luck_df)

if OLD_NAMING:
    df_luck["selection_source"] = df_luck["selection_source"].replace(
        {"HBF": "frs", "random_draws": "random_search"}
    )

df_luck.index = df_luck["selection_source"]

# Re-calculate with random_search median as normaliser (older runs used each method's own median)
_random_median = df_luck.loc["random_search", "median_score"]
df_luck["luck_factor"] = (
    df_luck["median_score"] - df_luck["best_score"]
) / _random_median
df_luck["misfortune_factor"] = (
    df_luck["worst_score"] - df_luck["median_score"]
) / _random_median

df_luck["rank_best"] = (
    df_results_sorted.groupby("selection_source")["wasserstein_seasonal"].idxmin() + 1
)
df_luck_summary = (
    df_luck[
        ["best_score", "rank_best", "median_score", "luck_factor", "misfortune_factor"]
    ]
    .round(2)
    .T
)
df_luck_summary.to_csv(os.path.join(SAVE_DIR, "summary_table.csv"))
print("Luck summary:\n", df_luck_summary)

# %% 3b. Seed uncertainty on the best and median SSWD score

if N_SEEDS_UNCERTAINTY:
    print(
        f"Re-scoring best/median selections with {N_SEEDS_UNCERTAINTY} seeds "
        f"(n_projections={N_PROJECTIONS_UNCERTAINTY})..."
    )
    df_sswd_unc, sswd_unc_raw = sswd_seed_uncertainty(
        df_results,
        df_ref,
        vars_non_nan,
        SEASONS,
        n_projections=N_PROJECTIONS_UNCERTAINTY,
        seed=SEED,
        n_seeds=N_SEEDS_UNCERTAINTY,
    )
    df_sswd_unc.to_csv(os.path.join(SAVE_DIR, "sswd_seed_uncertainty.csv"))
    print(
        f"SSWD over seeds {SEED}–{SEED + N_SEEDS_UNCERTAINTY - 1}, "
        f"n_projections={N_PROJECTIONS_UNCERTAINTY} (mean ± std, GWh/day):\n"
        f"{format_mean_std(df_sswd_unc)}"
    )

# %% 4. Compute all metrics (once)

metrics = compute_all_metrics(
    active_subsets,
    df_ref,
    vars_non_nan,
    SEASONS,
    n_projections=N_PROJECTIONS,
    seed=SEED,
)

# %% 5. Build & print score tables

seasonal_dfs, df_scores_summary = build_score_tables(metrics, active_subsets)
for season_name, df_s in seasonal_dfs.items():
    print(f"── {season_name} ──")
    print(df_s)
print("── Summary (worst season) ──")
print(df_scores_summary)

# %% 6. Pareto scatter

plot_results_with_pareto(
    df_results,
    "wasserstein_seasonal",
    "wasserstein",
    "SSWD (GWh/day)",
    "SWD (GWh/day)",
    SAVE_DIR,
)

# %% 7. Per-variable plots (optional)

if PLOT_EVERYTHING:
    plot_series = _build_plot_series(df_ref, active_subsets)
    plot_distributions(
        plot_series, PLOT_VARS, PLOT_VAR_LABELS, nrows=PLOT_ROWS, ncols=PLOT_COLS
    )
    plot_ecdf(plot_series, PLOT_VARS, PLOT_VAR_LABELS, nrows=PLOT_ROWS, ncols=PLOT_COLS)
    plot_qq_signed(
        plot_series,
        df_ref,
        PLOT_VARS,
        PLOT_VAR_LABELS,
        nrows=PLOT_ROWS,
        ncols=PLOT_COLS,
    )
    plot_qq_absolute(
        plot_series,
        df_ref,
        PLOT_VARS,
        PLOT_VAR_LABELS,
        nrows=PLOT_ROWS,
        ncols=PLOT_COLS,
    )
    plot_seasonal_qq(plot_series, df_ref, PLOT_VARS, PLOT_VAR_LABELS, SEASONS)

# %% 8. Correlation matrices (optional)

if PLOT_EVERYTHING and not MULTI_COUNTRY:
    corr_mats_annual = {"Reference": df_ref[vars_non_nan].corr()}
    for name, info in active_subsets.items():
        corr_mats_annual[name] = info["df"][vars_non_nan].corr()
    for plot_type in ("signed", "absolute"):
        _plot_correlation_panels(
            corr_mats_annual, vars_non_nan, title_prefix="(annual)", plot_type=plot_type
        )

    for season_name, months in SEASONS.items():
        corr_mats_s = {"Reference": _filter_season(df_ref, months)[vars_non_nan].corr()}
        for name, info in active_subsets.items():
            corr_mats_s[name] = _filter_season(info["df"], months)[vars_non_nan].corr()
        for plot_type in ("signed", "absolute"):
            _plot_correlation_panels(
                corr_mats_s,
                vars_non_nan,
                title_prefix=f"({season_name})",
                plot_type=plot_type,
            )

# %% 9. Summary figure

cfg = build_summary_config(include_eraa25=INCLUDE_ERAA25)
make_summary_figure(
    mode="seasonal",
    metrics=metrics,
    cfg=cfg,
    plot_vars=PLOT_VARS,
    save_dir=SAVE_DIR,
    multi_country=MULTI_COUNTRY,
    n_countries=len(MATCH_COUNTRIES),
)
# %% 10. Equivalent-to-N-years (optional)

if CALC_EQUIVALENT_TO_N_YEARS:
    num_years_range, all_scores = plot_equivalent_to_n_years(
        df_results, df_ref, vars_non_nan, SELECTED_YEARS, SAVE_DIR
    )
# %%
