"""
Evaluation script for best subsets per method.
Detailed distribution, QQ, and correlation plots comparing each best subset to the reference dataset.
Some previously scores (QQ-MAE, SSWD) are re-computed here, to get seasonal breakdowns. Optimizing was only done on worst-season.
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

from cluster_weather_years.constants import (
    BASE_OUTPUT,
    HYDROLOGICAL_YEARS,
    SEASONS,
)
from cluster_weather_years.evaluation import (
    build_score_tables,
    compute_all_metrics,
)
from cluster_weather_years.evaluation_plot_helpers import (
    _build_plot_series,
    _filter_season,
    _plot_correlation_panels,
    build_summary_config,
    compare_random_30_vs_36,
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
from cluster_weather_years.helper_functions import build_df_from_modelyears

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these toggles
# ═══════════════════════════════════════════════════════════════════════
N_PROJECTIONS = 200  # number of random projections for SWD. Match this to the value used during optimization for consistency, or set higher for more accurate final scores (at the cost of longer computation time).
SEED = 42  # reproducibility seed for SWD
INCLUDE_ERAA25 = True  # set False to exclude ERAA25 from all plots
MULTI_COUNTRY = (
    False  # True: panel (b) groups variables by type and sums QQ-MAE across countries
)
PLOT_EVERYTHING = False  # set False to skip distribution/QQ/correlation plots and only compute scores + summary tables (faster)
CALC_EQUIVALENT_TO_N_YEARS = False  # If True, compute how many random draws would be needed to match the best subset's SWD score (rough estimate of "equivalent number of years" for that method)
COMPARE_30_VS_36 = (
    False  # If True, compare 30yr vs 36yr random draw quality. Only had to be run once.
)
OLD_NAMING = True

# paper runs (initial submission)
# RUNNAME_ = "run_Europe_30y_hydro_20260306_1159"
# RUNNAME_ = "run_Europe_5y_hydro_20260323_1608"
RUNNAME_ = "run_NL_30y_hydro_20260319_1631"

# newer runs (LSTR)
# RUNNAME_ = "NL_30y_hydro_lstr_test_20260519_0818"
SELECTED_YEARS = 30
# HYDROLOGICAL_YEARS = range(2016, 2044)

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

# %% 9b. Summary figure (presentation version)
# NOTE: Temporary, not to be used for paper

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cluster_weather_years.evaluation_plot_helpers import (
    _plot_panel_b_qqmae,
    _plot_panel_c_corrdev,
    fontsize,
    fontsize_title,
)


def _plot_panel_a_swd_thin(ax, mode, metrics, cfg, bar_h=0.125):
    """Same as _plot_panel_a_swd but with configurable (thinner) bar height."""
    n_methods = len(cfg["labels"])
    y = np.arange(n_methods) * 0.66

    if mode == "seasonal":
        for i, (key, label) in enumerate(zip(cfg["keys"], cfg["labels"])):
            for j, (season, hatch) in enumerate(
                zip(metrics["season_list"], [".", "/"])
            ):
                val = metrics["swd_seasonal"][season][key]
                ax.barh(
                    y[i] + (j - 0.5) * bar_h,
                    val,
                    bar_h,
                    color=cfg["colors"][label],
                    hatch=hatch,
                    edgecolor="white",
                )
                ax.text(
                    val + 0.002,
                    y[i] + (j - 0.5) * bar_h,
                    f"{val:.2f}",
                    va="center",
                    fontsize=fontsize - 3,
                    color="0.3",
                )
            val_sswd = metrics["sswd"][key]
            ax.barh(
                y[i] + bar_h + bar_h / 2,
                val_sswd,
                bar_h,
                color=cfg["colors"][label],
                edgecolor="white",
            )
            ax.text(
                val_sswd + 0.002,
                y[i] + bar_h + bar_h / 2,
                f"{val_sswd:.2f}",
                va="center",
                fontsize=fontsize - 3,
                fontweight="bold",
                color="0.3",
            )

        if n_methods <= 4:
            ax.set_xlim(0, max(metrics["sswd"].values()) * 1.3)

        ax.legend(
            handles=[
                Patch(
                    facecolor="0.6",
                    edgecolor="white",
                    hatch="..",
                    label=metrics["season_list"][0],
                ),
                Patch(
                    facecolor="0.6",
                    edgecolor="white",
                    hatch="///",
                    label=metrics["season_list"][1],
                ),
                Patch(facecolor="0.6", edgecolor="white", label="SSWD (max)"),
            ],
            loc="upper right",
            fontsize=fontsize,
        )
        ax.set_title(
            "(a) Sliced Wasserstein distance, per season",
            fontweight="bold",
            fontsize=fontsize_title,
        )
    else:
        for i, (key, label) in enumerate(zip(cfg["keys"], cfg["labels"])):
            val = metrics["swd_annual"][key]
            ax.barh(y[i], val, bar_h, color=cfg["colors"][label], edgecolor="white")
            ax.text(
                val + 0.002,
                y[i],
                f"{val:.2f}",
                va="center",
                fontsize=fontsize - 3,
                fontweight="bold",
                color="0.3",
            )
        ax.set_title(
            "(a) Sliced Wasserstein distance — annual",
            fontweight="bold",
            fontsize=fontsize_title,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(cfg["labels"], fontsize=fontsize)
    ax.invert_yaxis()
    ax.set_xlabel("SWD (GWh/day)", fontsize=fontsize)


def make_summary_figure_presentation(
    mode: str,
    metrics: dict,
    cfg: dict,
    plot_vars: list,
    save_dir: str,
    multi_country: bool = False,
    n_countries: int = 1,
    methods_to_keep=("Best SA Subset", "ERAA25 Subset"),
):
    """Presentation variant of make_summary_figure.

    - Only plots the methods in ``methods_to_keep`` (order preserved from cfg).
    - Saves the PDF with a transparent background.
    """
    kept = [i for i, k in enumerate(cfg["keys"]) if k in methods_to_keep]
    cfg_filt = {
        "keys": [cfg["keys"][i] for i in kept],
        "labels": [cfg["labels"][i] for i in kept],
        "colors": {cfg["labels"][i]: cfg["colors"][cfg["labels"][i]] for i in kept},
    }

    fig = plt.figure(figsize=(13, 9.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.78, 1], hspace=0.38, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    _plot_panel_a_swd_thin(ax_a, mode, metrics, cfg_filt)
    _plot_panel_b_qqmae(
        ax_b, mode, metrics, cfg_filt, plot_vars, multi_country, n_countries
    )
    _plot_panel_c_corrdev(ax_c, mode, metrics, cfg_filt)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=cfg_filt["colors"][m])
        for m in cfg_filt["labels"]
    ]
    fig.legend(
        handles,
        cfg_filt["labels"],
        loc="lower center",
        ncol=len(cfg_filt["labels"]),
        fontsize=fontsize,
        bbox_to_anchor=(0.5, -0.075),
        frameon=True,
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"summary_figure_{mode}_presentation.png"),
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    return fig


fig = make_summary_figure_presentation(
    mode="seasonal",  # or "annual"
    metrics=metrics,
    cfg=cfg,
    plot_vars=PLOT_VARS,
    save_dir=SAVE_DIR,
    multi_country=True,
    n_countries=len(MATCH_COUNTRIES),
)
plt.show()

# %% 10. Equivalent-to-N-years (optional)

if CALC_EQUIVALENT_TO_N_YEARS:
    num_years_range, all_scores = plot_equivalent_to_n_years(
        df_results, df_ref, vars_non_nan, SELECTED_YEARS, SAVE_DIR
    )

if COMPARE_30_VS_36:
    all_scores = compare_random_30_vs_36(df_ref, vars_non_nan)

# %%
