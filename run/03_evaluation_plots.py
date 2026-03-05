# %%
import os
from ast import literal_eval
from itertools import product

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from ot import sliced_wasserstein_distance
from tqdm import tqdm

from cluster_weather_years.constants import (
    BASE_OUTPUT,
    HYDROLOGICAL_YEARS,
    MODELS,
    SEASONS,
)
from cluster_weather_years.helper_functions import build_df_from_modelyears
from cluster_weather_years.hierarchical_brute_force import build_random_df
from cluster_weather_years.scores import wasserstein_seasonal_score

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these toggles
# ═══════════════════════════════════════════════════════════════════════
N_PROJECTIONS = 200  # number of random projections for SWD
SEED = 42  # reproducibility seed for SWD
INCLUDE_ERAA25 = True  # set False to exclude ERAA25 from all plots
MULTI_COUNTRY = (
    False  # True: panel (b) groups variables by type and sums QQ-MAE across countries
)

# RUNNAME_ = "run_NL_30y_fair_20260225_1243"
RUNNAME_ = "run_NL_5y_fair_20260301_1105"
# RUNNAME_ = "run_Europe_30y_fair_20260227_1625"
SELECTED_YEARS = 5
# HYDROLOGICAL_YEARS = range(2016, 2044)

# Plot layout
PLOT_ROWS, PLOT_COLS = 3, 2
PLOT_QUANTILES = np.linspace(0.01, 0.99, 99)
SAVE_DIR = os.path.join(BASE_OUTPUT, RUNNAME_, "evaluation_plots")
os.makedirs(SAVE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# VARIABLE LABEL MAPPING
# ═══════════════════════════════════════════════════════════════════════
_VAR_SUFFIX_MAP = {
    "OFF_prod": "Offshore \n wind prod.",
    "ON_prod": "Onshore \n wind prod.",
    "SOLAR_prod": "Solar prod.",
    "Demand": "Demand",
    "Residual_load": "Residual load",
    "ROR_prod": "Run-of-river prod.",
    "RES_prod": "Reservoir prod.",
}


def pretty_var(raw_name: str) -> str:
    """Map e.g. 'NL_OFF_prod' -> 'NL: Offshore prod.'."""
    for suffix, label in _VAR_SUFFIX_MAP.items():
        if raw_name.endswith(suffix):
            country = raw_name[: -(len(suffix) + 1)]  # strip '_suffix'
            return f"{country}: {label}"
    return raw_name  # fallback: return as-is


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING STYLE
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.2,
    }
)

# Method display config: (internal_key, short_label, color)
# METHOD_CONFIG = [
#     ("Best SA Subset", "SA", "#2ca02c"),
#     ("Best K-Medoids Subset", "K-Medoids", "#d62728"),
#     ("Best HBF Subset", "HBF", "#9467bd"),
#     ("Best Random Subset", "Random", "#8c564b"),
#     ("ERAA25 Subset", "ERAA25 (36 yr)", "#1f77b4"),
# ]
METHOD_CONFIG = [
    ("Best SA Subset", "SA", "#4daf4a"),  # bluish-green
    ("Best K-Medoids Subset", "K-Medoids", "#984ea3"),  # purple
    ("Best HBF Subset", "HBF", "#ff7f00"),  # red
    ("Best Random Subset", "Random", "#377eb8"),  # blue
    ("ERAA25 Subset", "ERAA25 (36 yr)", "#e41a1c"),  # orange
]

# Substring patterns (lowercase) that identify each selection_source
_SOURCE_KEY_MAP = [
    ("simulated_annealing", "Best SA Subset"),
    ("k_medoids", "Best K-Medoids Subset"),
    ("hierarchical", "Best HBF Subset"),
    ("brute_force", "Best HBF Subset"),
    ("hbf", "Best HBF Subset"),
    ("random_draws", "Best Random Subset"),
    ("eraa", "ERAA25 Subset"),
]
_KEY_TO_LABEL = {key: label for key, label, _ in METHOD_CONFIG}
_KEY_TO_COLOR = {key: color for key, _, color in METHOD_CONFIG}


def _source_style(source):
    """Return (display_label, color) for a selection_source string."""
    s = str(source).lower()
    for substr, key in _SOURCE_KEY_MAP:
        if substr in s:
            return _KEY_TO_LABEL[key], _KEY_TO_COLOR[key]
    return source, "gray"  # fallback


def _active_methods():
    """Return METHOD_CONFIG filtered by INCLUDE_ERAA25 toggle."""
    if INCLUDE_ERAA25:
        return METHOD_CONFIG
    return [m for m in METHOD_CONFIG if m[0] != "ERAA25 Subset"]


# %%
# 1. Data loading

fname_results = os.path.join(BASE_OUTPUT, RUNNAME_, "all_year_selections_scored.csv")
df_results = pd.read_csv(fname_results, index_col=0)

variable_order = (
    pd.read_csv(os.path.join(BASE_OUTPUT, RUNNAME_, "variable_order.csv"))
    .values.flatten()
    .tolist()
)

# Load run settings (defines DF_PROD_DEMAND_FNAME, VARS_TO_MATCH, etc.)
fname_settings = os.path.join(BASE_OUTPUT, RUNNAME_, "RUN_SETTINGS_of_run.py")
with open(fname_settings) as f:
    exec(f.read())

# %%
# Reference dataset

df_daily = pd.read_parquet(DF_PROD_DEMAND_FNAME)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.query("Hydrological_year in @HYDROLOGICAL_YEARS")
df_daily = df_daily.set_index(["model", "Date"])

vars_non_nan = [v for v in VARS_TO_MATCH if not df_daily[v].isna().all()]
df_ref = df_daily[["Hydrological_year", "Calendar_year"] + VARS_TO_MATCH].reset_index()

PLOT_VARS = vars_non_nan[: PLOT_ROWS * PLOT_COLS]
PLOT_VAR_LABELS = [pretty_var(v) for v in PLOT_VARS]


# %%
def calc_pareto_mask(df, cols):
    """Mask for Pareto-optimal points in the dataframe based on specified columns."""
    X = df.loc[:, cols].to_numpy()
    n = len(df)
    m = np.ones(n, dtype=bool)
    for i in range(n):
        if not m[i]:
            continue
        # point j dominates i if j <= i in all and < in at least one (lower=better)
        dom_i = np.all(X <= X[i], axis=1) & np.any(X < X[i], axis=1)
        if dom_i.any():
            m[i] = False
    return m


def plot_results_with_pareto(
    df_results, metric_1, metric_2, label_m1, label_m2, save_dir=None
):
    """Scatter of metric_1 vs metric_2, with Pareto front highlighted."""
    # Map each row's selection_source to display label and color
    styles = df_results["selection_source"].map(_source_style)
    df_results = df_results.copy()
    df_results["_label"] = [s[0] for s in styles]
    df_results["_color"] = [s[1] for s in styles]

    pareto_mask = calc_pareto_mask(df_results, cols=(metric_1, metric_2))
    pareto_front = df_results.loc[pareto_mask]

    pareto_composition = pareto_front["_label"].value_counts()
    pareto_label = "Pareto front: " + ", ".join(
        f"{n}x {lbl}" for lbl, n in pareto_composition.items()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        pareto_front[metric_1],
        pareto_front[metric_2],
        facecolors="none",
        edgecolors="k",
        linewidths=1.2,
        label=pareto_label,
        s=30,
        zorder=5,
    )

    for label, grp in df_results.groupby("_label"):
        ax.scatter(
            grp[metric_1],
            grp[metric_2],
            color=grp["_color"].iloc[0],
            label=label,
            marker="o",
            s=14,
            alpha=0.8,
        )

    ax.set_xlabel(f"{label_m1} (lower is better)")
    ax.set_ylabel(f"{label_m2} (lower is better)")
    # ax.set_xlim(0, None)
    # ax.set_ylim(0, None)
    # Sort legend: Pareto ring first, then methods in METHOD_CONFIG order
    _method_order = [label for _, label, _ in METHOD_CONFIG]
    handles, labels = ax.get_legend_handles_labels()
    hl = {lbl: h for h, lbl in zip(handles, labels)}
    ordered_labels = [lbl for lbl in labels if lbl.startswith("Pareto")] + [
        lbl for lbl in _method_order if lbl in hl
    ]
    ax.legend([hl[lbl] for lbl in ordered_labels], ordered_labels)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/pareto_front_{metric_1}_{metric_2}.pdf")


plot_results_with_pareto(
    df_results, "wasserstein_seasonal", "wasserstein", "SSWD (-)", "SWD (-)", SAVE_DIR
)

# %%
# 2. Extract best subsets per method


def _to_modelyears(entry):
    return literal_eval(entry) if isinstance(entry, str) else entry


df_results_sorted = df_results.sort_values("wasserstein_seasonal")


def _best_subset_by_source(pattern, regex=False, exact=True):
    """Find the best-scoring model_years for a given selection_source."""
    col = df_results_sorted["selection_source"].astype(str).str.lower()
    if regex:
        mask = col.str.contains(pattern, regex=True)
    elif exact:
        mask = col.eq(pattern)
    else:
        mask = col.str.contains(pattern, regex=False)
    if not mask.any():
        raise ValueError(f"No selection_source matching '{pattern}' in df_results")
    # modelyears = _to_modelyears(
    #     mask.idxmax() and df_results_sorted.loc[mask].iloc[0]["model_years"]
    modelyears = _to_modelyears(df_results_sorted.loc[mask].iloc[0]["model_years"])
    return modelyears, build_df_from_modelyears(df_ref, modelyears)


sel_my_sa, df_sub_sa = _best_subset_by_source("simulated_annealing")
sel_my_kmed, df_sub_kmed = _best_subset_by_source("k_medoids", regex=False, exact=False)
sel_my_random, df_sub_random = _best_subset_by_source("random_draws")
sel_my_hbf, df_sub_hbf = _best_subset_by_source(
    r"(^hbf$|hierarchical|brute_force)", regex=True
)

# ERAA25 baseline
eraa25_models = ["cmcc_cm2_sr5", "ec_earth3", "mpi_esm_1_2_hr"]
eraa25_years = range(2025, 2037)
eraa25_modelyears = tuple(product(eraa25_models, eraa25_years))
df_sub_eraa25 = build_df_from_modelyears(df_ref, eraa25_modelyears)

# Central registry of all subsets
ALL_SUBSETS = {
    "Best SA Subset": {"df": df_sub_sa, "modelyears": sel_my_sa},
    "Best K-Medoids Subset": {"df": df_sub_kmed, "modelyears": sel_my_kmed},
    "Best HBF Subset": {"df": df_sub_hbf, "modelyears": sel_my_hbf},
    "Best Random Subset": {"df": df_sub_random, "modelyears": sel_my_random},
    "ERAA25 Subset": {"df": df_sub_eraa25, "modelyears": eraa25_modelyears},
}


def _active_subsets():
    """Return subset dict filtered by INCLUDE_ERAA25 toggle."""
    if INCLUDE_ERAA25:
        return ALL_SUBSETS
    return {k: v for k, v in ALL_SUBSETS.items() if k != "ERAA25 Subset"}


# Build the PLOT_SERIES list used by distribution/QQ/correlation plots
def _build_plot_series():
    series = [("Reference", df_ref, "black")]
    color_map = {k: c for k, _, c in METHOD_CONFIG}
    for name in _active_subsets():
        series.append((name, ALL_SUBSETS[name]["df"], color_map[name]))
    return series


# %%
# 3. Shared helper functions


def _ecdf(values):
    x = np.sort(np.asarray(values))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _qq_diff_and_stats(ref_values, sel_values, quantiles=PLOT_QUANTILES):
    q_ref = np.quantile(ref_values.dropna().values, quantiles)
    q_sel = np.quantile(sel_values.dropna().values, quantiles)
    qq_diff = q_sel - q_ref
    qq_mae = np.mean(np.abs(qq_diff))
    qq_rmse = np.sqrt(np.mean(qq_diff**2))
    return qq_diff, qq_mae, qq_rmse


def _corr_deviations(corr_sub, corr_ref):
    """Mean and max absolute correlation deviation (strict lower triangle)."""
    mask_lower = np.tril(np.ones(corr_ref.shape, dtype=bool), k=-1)
    abs_dev = np.abs((corr_sub - corr_ref).to_numpy()[mask_lower])
    return float(np.mean(abs_dev)), float(np.max(abs_dev))


def _filter_season(df, months):
    """Keep only rows whose Date month is in `months`."""
    return df[pd.to_datetime(df["Date"]).dt.month.isin(months)]


def _compute_swd(X_sub, X_ref):
    """Compute SWD with the global N_PROJECTIONS and SEED settings."""
    return float(
        sliced_wasserstein_distance(
            X_sub, X_ref, n_projections=N_PROJECTIONS, seed=SEED
        )
    )


def _new_axes(nrows=PLOT_ROWS, ncols=PLOT_COLS, figsize=(14, 12), sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    return fig, axes.flatten()


def _add_panel_labels(axes, n_panels):
    for i in range(n_panels):
        axes[i].text(
            0.0,
            1.08,
            f"({chr(97 + i)})",
            transform=axes[i].transAxes,
            va="bottom",
            ha="left",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )


def _apply_shared_axis_labels(axes, n_panels, ncols, xlabel, ylabel):
    for i in range(n_panels):
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    for i in range(n_panels):
        if i % ncols == 0:
            axes[i].set_ylabel(ylabel)
    for col in range(ncols):
        rows_in_col = [i // ncols for i in range(n_panels) if i % ncols == col]
        if not rows_in_col:
            continue
        idx = max(rows_in_col) * ncols + col
        if idx < n_panels:
            axes[idx].set_xlabel(xlabel)


def _hide_unused_axes(axes, n_used, n_total):
    for j in range(n_used, n_total):
        axes[j].axis("off")


# %%
# 4. Compute all metrics (once)

# 4a. Annual SWD per subset
print("Computing SWD scores...")
X_ref_all = df_ref[vars_non_nan].dropna().to_numpy()

swd_annual = {}
for name, info in _active_subsets().items():
    X_sub = info["df"][vars_non_nan].dropna().to_numpy()
    swd_annual[name] = _compute_swd(X_sub, X_ref_all)

# 4b. Seasonal SWD per subset
swd_seasonal = {}  # {season: {method: val}}
for season_name, months in SEASONS.items():
    df_ref_s = _filter_season(df_ref, months)
    X_ref_s = df_ref_s[vars_non_nan].dropna().to_numpy()
    swd_seasonal[season_name] = {}
    for name, info in _active_subsets().items():
        df_sub_s = _filter_season(info["df"], months)
        X_sub_s = df_sub_s[vars_non_nan].dropna().to_numpy()
        swd_seasonal[season_name][name] = _compute_swd(X_sub_s, X_ref_s)

# SSWD = max across seasons
sswd = {}
for name in _active_subsets():
    sswd[name] = max(swd_seasonal[s][name] for s in SEASONS)

# 4c. Annual QQ-MAE per variable per subset
qq_mae_annual = {}  # {method: {var: float}} over all vars_non_nan
for name, info in _active_subsets().items():
    qq_mae_annual[name] = {
        v: _qq_diff_and_stats(df_ref[v], info["df"][v])[1] for v in vars_non_nan
    }

# 4d. Seasonal QQ-MAE per variable per subset
qq_mae_seasonal = {}  # {season: {method: {var: float}}} over all vars_non_nan
for season_name, months in SEASONS.items():
    df_ref_s = _filter_season(df_ref, months)
    qq_mae_seasonal[season_name] = {}
    for name, info in _active_subsets().items():
        df_sub_s = _filter_season(info["df"], months)
        qq_mae_seasonal[season_name][name] = {
            v: _qq_diff_and_stats(df_ref_s[v], df_sub_s[v])[1] for v in vars_non_nan
        }

# 4e. Correlation deviations (annual + seasonal)
corr_dev = {}  # {method: {"Annual": val, season: val, ...}}
corr_ref_annual = df_ref[vars_non_nan].corr()
for name, info in _active_subsets().items():
    corr_sub = info["df"][vars_non_nan].corr()
    mean_acd_ann, _ = _corr_deviations(corr_sub, corr_ref_annual)
    corr_dev[name] = {"Annual": mean_acd_ann}
    for season_name, months in SEASONS.items():
        df_ref_s = _filter_season(df_ref, months)
        df_sub_s = _filter_season(info["df"], months)
        mean_acd_s, _ = _corr_deviations(
            df_sub_s[vars_non_nan].corr(), df_ref_s[vars_non_nan].corr()
        )
        corr_dev[name][season_name] = mean_acd_s

print(
    f"All metrics computed ✓  ({len(vars_non_nan)} vars, n_projections={N_PROJECTIONS}, seed={SEED})"
)


def _group_qq_mae_by_vartype(qq_mae_dict):
    """Sum QQ-MAE values in {var: float} across countries, grouped by variable type.

    Uses _VAR_SUFFIX_MAP to identify variable types; preserves its order.
    Returns an OrderedDict {vartype_label: summed_qq_mae}.
    """
    grouped = {}
    for suffix, label in _VAR_SUFFIX_MAP.items():
        matching = [v for v in qq_mae_dict if v.endswith(suffix)]
        if matching:
            grouped[label] = sum(qq_mae_dict[v] for v in matching)
    return grouped


# 4g. Build score tables
season_list = list(SEASONS.keys())

# Per-season detail tables
seasonal_dfs = {}
for season_name in SEASONS:
    rows = []
    for name in _active_subsets():
        rows.append(
            {
                "subset": name,
                "mean_qq_mae": float(
                    np.mean(list(qq_mae_seasonal[season_name][name].values()))
                ),
                "max_qq_mae": float(
                    np.max(list(qq_mae_seasonal[season_name][name].values()))
                ),
                "swd_seasonal": swd_seasonal[season_name][name],
                "mean_corr_dev": corr_dev[name][season_name],
            }
        )
    df_s = pd.DataFrame(rows).sort_values("swd_seasonal").reset_index(drop=True)
    df_s[["mean_qq_mae", "max_qq_mae", "swd_seasonal", "mean_corr_dev"]] = df_s[
        ["mean_qq_mae", "max_qq_mae", "swd_seasonal", "mean_corr_dev"]
    ].round(3)
    seasonal_dfs[season_name] = df_s

# Worst-season summary
worst_rows = []
for name in _active_subsets():
    worst_rows.append(
        {
            "subset": name,
            "swd_annual": swd_annual[name],
            "sswd": sswd[name],
            **{f"swd_{s}": swd_seasonal[s][name] for s in season_list},
            "worst_mean_qq_mae": max(
                np.mean(list(qq_mae_seasonal[s][name].values())) for s in season_list
            ),
            "worst_mean_corr_dev": max(corr_dev[name][s] for s in season_list),
        }
    )
df_scores_summary = pd.DataFrame(worst_rows).sort_values("sswd").reset_index(drop=True)
numeric_cols = df_scores_summary.select_dtypes(include="number").columns
df_scores_summary[numeric_cols] = df_scores_summary[numeric_cols].round(3)

# Display tables
for season_name, df_s in seasonal_dfs.items():
    print(f"── {season_name} ──")
    print(df_s)
print("── Summary (worst season) ──")
print(df_scores_summary)


# %%
# 5. Distribution plots

PLOT_SERIES = _build_plot_series()

fig, axes = _new_axes(sharey=False)
for i, var in enumerate(PLOT_VARS):
    ax = axes[i]
    values_all = np.concatenate([df[var].dropna().values for _, df, _ in PLOT_SERIES])
    bins = np.linspace(values_all.min(), values_all.max(), 51)

    for label, df_sel, color in PLOT_SERIES:
        if label == "Reference":
            ax.hist(
                df_sel[var].dropna().values,
                bins=bins,
                color="0.6",
                label=label,
                density=True,
                histtype="stepfilled",
                alpha=0.85,
                linewidth=0.8,
            )
        else:
            ax.hist(
                df_sel[var].dropna().values,
                bins=bins,
                color=color,
                label=label,
                density=True,
                histtype="step",
                linewidth=1.5,
            )

    x_pad = 0.03 * (values_all.max() - values_all.min())
    ax.set_xlim(values_all.min() - x_pad, values_all.max() + x_pad)
    ax.set_title(PLOT_VAR_LABELS[i])
    if i == 0:
        ax.legend(loc="best")

_hide_unused_axes(axes, len(PLOT_VARS), PLOT_ROWS * PLOT_COLS)
_apply_shared_axis_labels(
    axes, len(PLOT_VARS), PLOT_COLS, xlabel="Value (GWh/day)", ylabel="Density"
)
_add_panel_labels(axes, len(PLOT_VARS))
fig.tight_layout()
plt.show()


# %%
# 6. ECDF plots

fig, axes = _new_axes(sharey=True)
for i, var in enumerate(PLOT_VARS):
    ax = axes[i]
    x_all = []
    for label, df_sel, color in PLOT_SERIES:
        x, y = _ecdf(df_sel[var].dropna())
        x_all.append(x)
        ax.step(x, y, where="post", color=color, linewidth=1.5, label=label)

    x_concat = np.concatenate(x_all)
    x_pad = 0.03 * (x_concat.max() - x_concat.min())
    ax.set_xlim(x_concat.min() - x_pad, x_concat.max() + x_pad)
    ax.set_title(PLOT_VAR_LABELS[i])
    ax.set_ylim(0, 1)
    if i == 0:
        ax.legend(loc="lower right")

_hide_unused_axes(axes, len(PLOT_VARS), PLOT_ROWS * PLOT_COLS)
_apply_shared_axis_labels(
    axes,
    len(PLOT_VARS),
    PLOT_COLS,
    xlabel="Value (GWh/day)",
    ylabel="Cumulative probability",
)
_add_panel_labels(axes, len(PLOT_VARS))
fig.tight_layout()
plt.show()


# %%
# 7. QQ-difference plots

# Signed QQ-differences
fig, axes = _new_axes(sharey=False)
qq_ymax = 0.0
for i, var in enumerate(PLOT_VARS):
    ax = axes[i]
    for label, df_sel, color in PLOT_SERIES[1:]:
        qq_diff, _, _ = _qq_diff_and_stats(df_ref[var], df_sel[var])
        ax.plot(
            PLOT_QUANTILES, qq_diff, color=color, linewidth=1.5, label=f"{label} − Ref"
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    for q in (0.1, 0.5, 0.9):
        ax.axvline(q, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    qq_ymax = max(qq_ymax, np.max(np.abs(ax.get_ylim())))
    ax.set_title(PLOT_VAR_LABELS[i])
    ax.set_xlim(PLOT_QUANTILES.min(), PLOT_QUANTILES.max())
    if i == 0:
        ax.legend(loc="upper right")

_hide_unused_axes(axes, len(PLOT_VARS), PLOT_ROWS * PLOT_COLS)
for i in range(len(PLOT_VARS)):
    axes[i].set_ylim(-qq_ymax, qq_ymax)
_apply_shared_axis_labels(
    axes,
    len(PLOT_VARS),
    PLOT_COLS,
    xlabel="Quantile",
    ylabel=r"$Q_{sel} - Q_{ref}$ (GWh/day)",
)
_add_panel_labels(axes, len(PLOT_VARS))
fig.tight_layout()
plt.show()

# %%
# Absolute QQ-differences
fig, axes = _new_axes(sharey=False)
absqq_ymax = 0.0
for i, var in enumerate(PLOT_VARS):
    ax = axes[i]
    for label, df_sel, color in PLOT_SERIES[1:]:
        qq_diff, qq_mae, _ = _qq_diff_and_stats(df_ref[var], df_sel[var])
        ax.plot(
            PLOT_QUANTILES,
            np.abs(qq_diff),
            color=color,
            linewidth=1.5,
            label=f"|{label} − Ref|",
        )
        ax.axhline(qq_mae, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
    for q in (0.1, 0.5, 0.9):
        ax.axvline(q, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    absqq_ymax = max(absqq_ymax, ax.get_ylim()[1])
    ax.set_title(PLOT_VAR_LABELS[i])
    ax.set_xlim(PLOT_QUANTILES.min(), PLOT_QUANTILES.max())
    if i == 0:
        ax.legend(loc="upper right")

_hide_unused_axes(axes, len(PLOT_VARS), PLOT_ROWS * PLOT_COLS)
for i in range(len(PLOT_VARS)):
    axes[i].set_ylim(0, absqq_ymax)
_apply_shared_axis_labels(
    axes,
    len(PLOT_VARS),
    PLOT_COLS,
    xlabel="Quantile",
    ylabel=r"$|Q_{sel} - Q_{ref}|$ (GWh/day)",
)
_add_panel_labels(axes, len(PLOT_VARS))
fig.tight_layout()
plt.show()


# %%
# 8. Seasonal QQ-plots (absolute differences)

n_seasons = len(SEASONS)
nrows_s, ncols_s = len(PLOT_VARS), n_seasons

fig, axes = plt.subplots(
    nrows_s, ncols_s, figsize=(7 * ncols_s, 3.5 * nrows_s), sharey=False
)
axes = np.atleast_2d(axes)
row_ymaxes = np.zeros(nrows_s)

for col_idx, (season_name, months) in enumerate(SEASONS.items()):
    df_ref_s = _filter_season(df_ref, months)
    for row_idx, var in enumerate(PLOT_VARS):
        ax = axes[row_idx, col_idx]
        for label, df_sel, color in PLOT_SERIES[1:]:
            df_sel_s = _filter_season(df_sel, months)
            qq_diff, qq_mae, _ = _qq_diff_and_stats(df_ref_s[var], df_sel_s[var])
            ax.plot(
                PLOT_QUANTILES,
                np.abs(qq_diff),
                color=color,
                linewidth=1.5,
                label=f"|{label} − Ref|",
            )
            ax.axhline(qq_mae, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        for q in (0.1, 0.5, 0.9):
            ax.axvline(q, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlim(PLOT_QUANTILES.min(), PLOT_QUANTILES.max())
        if row_idx == 0:
            ax.set_title(season_name, fontsize=11, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(PLOT_VAR_LABELS[row_idx], fontsize=9)
        if row_idx == nrows_s - 1:
            ax.set_xlabel("Quantile")
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc="upper right", fontsize=7)
        row_ymaxes[row_idx] = max(row_ymaxes[row_idx], ax.get_ylim()[1])

for r in range(nrows_s):
    for c in range(ncols_s):
        axes[r, c].set_ylim(0, row_ymaxes[r])

for idx in range(nrows_s * ncols_s):
    r, c = divmod(idx, ncols_s)
    axes[r, c].text(
        0.0,
        1.05,
        f"({chr(97 + idx)})",
        transform=axes[r, c].transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )

fig.suptitle(
    r"Seasonal absolute QQ-differences: $|Q_{sel} - Q_{ref}|$ (GWh/day)",
    y=1.01,
    fontsize=11,
)
fig.tight_layout()
plt.show()


# %%
# 9. Correlation matrices


def _plot_correlation_panels(corr_mats, title_prefix="", plot_type="signed"):
    """Plot reference correlation + deviation panels for all active subsets."""
    ref_corr = corr_mats["Reference"]
    subset_names = [k for k in corr_mats if k != "Reference"]
    mask_upper = np.triu(np.ones_like(ref_corr, dtype=bool), k=1)

    n_panels = 1 + len(subset_names)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Pretty labels for axes
    pretty_labels = [pretty_var(v) for v in vars_non_nan]

    # Reference panel
    axes[0].set_facecolor("#d9d9d9")
    sns.heatmap(
        ref_corr,
        ax=axes[0],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        mask=mask_upper,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        cbar_kws={"label": "Correlation"},
        xticklabels=pretty_labels,
        yticklabels=pretty_labels,
    )
    axes[0].set_title(f"Reference {title_prefix}")

    for i, sname in enumerate(subset_names, start=1):
        diff = corr_mats[sname] - ref_corr
        axes[i].set_facecolor("#d9d9d9")
        if plot_type == "signed":
            data, cmap, vmin, vmax = diff, "RdBu_r", -0.1, 0.1
            cbar_label, title = "Δ correlation", f"{sname} − Ref"
        else:
            data, cmap, vmin, vmax = diff.abs(), "viridis", 0.0, 0.1
            cbar_label, title = "|Δ correlation|", f"|{sname} − Ref|"
        sns.heatmap(
            data,
            ax=axes[i],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            mask=mask_upper,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cbar_kws={"label": cbar_label},
            xticklabels=pretty_labels,
            yticklabels=pretty_labels,
        )
        axes[i].set_title(title)

    for k in range(n_panels, len(axes)):
        axes[k].axis("off")

    show_ticks = len(vars_non_nan) <= 20
    for k in range(n_panels):
        if show_ticks:
            axes[k].set_xticklabels(axes[k].get_xticklabels(), rotation=90, fontsize=7)
            axes[k].set_yticklabels(axes[k].get_yticklabels(), fontsize=7)
        else:
            axes[k].set_xticks([])
            axes[k].set_yticks([])

    fig.tight_layout()
    plt.show()


# Annual correlation
corr_mats_annual = {"Reference": df_ref[vars_non_nan].corr()}
for name, info in _active_subsets().items():
    corr_mats_annual[name] = info["df"][vars_non_nan].corr()

if not MULTI_COUNTRY:
    _plot_correlation_panels(
        corr_mats_annual, title_prefix="(annual)", plot_type="signed"
    )
    _plot_correlation_panels(
        corr_mats_annual, title_prefix="(annual)", plot_type="absolute"
    )

# %%
# Seasonal correlations
for season_name, months in SEASONS.items():
    corr_mats_s = {"Reference": _filter_season(df_ref, months)[vars_non_nan].corr()}
    for name, info in _active_subsets().items():
        corr_mats_s[name] = _filter_season(info["df"], months)[vars_non_nan].corr()

if not MULTI_COUNTRY:
    for pt in ("signed", "absolute"):
        _plot_correlation_panels(
            corr_mats_s, title_prefix=f"({season_name})", plot_type=pt
        )


# %%
# 10. Summary figures


# Helper: get short labels and colors for active methods
def _summary_config():
    methods = _active_methods()
    keys = [k for k, _, _ in methods]
    labels = [l for _, l, _ in methods]
    colors = {l: c for _, l, c in methods}
    key_to_label = {k: l for k, l, _ in methods}
    return keys, labels, colors, key_to_label


_sk, _sl, _sc, _k2l = _summary_config()


def _make_summary_figure(mode="seasonal"):
    """
    Create summary figure. mode='seasonal' uses SSWD with per-season bars;
    mode='annual' uses SWD with a single bar per method — but in the same
    visual layout as the seasonal figure for consistency.
    """
    fig = plt.figure(figsize=(13, 7.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.38, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    n_methods = len(_sl)
    bw = 0.15
    y = np.arange(n_methods)

    # ── (a) Horizontal bars: SWD / SSWD ──
    if mode == "seasonal":
        bar_h = 0.25
        for i, (key, label) in enumerate(zip(_sk, _sl)):
            vals = [swd_seasonal[s][key] for s in season_list]
            hatches = [".", "/"]
            for j, (val, hatch) in enumerate(zip(vals, hatches)):
                ax_a.barh(
                    y[i] + (j - 0.5) * bar_h,
                    val,
                    bar_h,
                    color=_sc[label],
                    hatch=hatch,
                    edgecolor="white",
                )
                ax_a.text(
                    val + 0.002,
                    y[i] + (j - 0.5) * bar_h,
                    f"{val:.2f}",
                    va="center",
                    fontsize=9,
                    color="0.3",
                )

            # SSWD bar (max of seasonal)
            val_sswd = sswd[key]
            ax_a.barh(
                y[i] + bar_h + bar_h / 2,
                val_sswd,
                bar_h,
                color=_sc[label],
                edgecolor="white",
            )
            ax_a.text(
                val_sswd + 0.002,
                y[i] + bar_h + bar_h / 2,
                f"{val_sswd:.2f}",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="0.3",
            )

        ax_a.legend(
            handles=[
                Patch(
                    facecolor="0.6", edgecolor="white", hatch="..", label=season_list[0]
                ),
                Patch(
                    facecolor="0.6",
                    edgecolor="white",
                    hatch="///",
                    label=season_list[1],
                ),
                Patch(facecolor="0.6", edgecolor="white", label="SSWD (max)"),
            ],
            loc="upper right",
            fontsize=10,
        )
        ax_a.set_title(
            "(a) Sliced Wasserstein distance, per season",
            fontweight="bold",
            fontsize=12,
        )
    else:  # annual
        bar_h = 0.25
        for i, (key, label) in enumerate(zip(_sk, _sl)):
            val = swd_annual[key]
            ax_a.barh(y[i], val, bar_h, color=_sc[label], edgecolor="white")
            ax_a.text(
                val + 0.002,
                y[i],
                f"{val:.2f}",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="0.3",
            )
        ax_a.set_title(
            "(a) Sliced Wasserstein distance — annual", fontweight="bold", fontsize=12
        )

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(_sl, fontsize=11)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("SWD (−)\n(lower = better)")

    # ── (b) QQ-MAE per variable ──
    if MULTI_COUNTRY:
        # Group variables by type, summing QQ-MAE across countries
        if mode == "seasonal":
            # Build grouped data: {season: {method: {vartype: float}}}
            _grp_seasonal = {
                sn: {
                    key: _group_qq_mae_by_vartype(qq_mae_seasonal[sn][key])
                    for key in _sk
                }
                for sn in season_list
            }
            vartypes = list(_grp_seasonal[season_list[0]][_sk[0]].keys())
            xv = np.arange(len(vartypes))
            marker_styles = {season_list[0]: "o", season_list[1]: "x"}
            for i, (key, label) in enumerate(zip(_sk, _sl)):
                off = (i - (n_methods - 1) / 2) * 0.13
                for sn, marker in marker_styles.items():
                    alpha = 0.55 if sn == season_list[0] else 0.35
                    size = 40 if sn == season_list[0] else 28
                    vals = [_grp_seasonal[sn][key][vt] for vt in vartypes]
                    ax_b.scatter(
                        xv + off,
                        vals,
                        color=_sc[label],
                        s=size,
                        zorder=3,
                        alpha=alpha,
                        marker=marker,
                    )
            season_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="0.5",
                    markersize=6,
                    linestyle="none",
                    label=season_list[0],
                ),
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="0.5",
                    markersize=5,
                    linestyle="none",
                    alpha=0.6,
                    label=season_list[1],
                ),
            ]
            ax_b.legend(handles=season_handles, loc="best", fontsize=7, framealpha=0.8)
        else:  # annual
            _grp_annual = {
                key: _group_qq_mae_by_vartype(qq_mae_annual[key]) for key in _sk
            }
            vartypes = list(next(iter(_grp_annual.values())).keys())
            xv = np.arange(len(vartypes))
            for i, (key, label) in enumerate(zip(_sk, _sl)):
                off = (i - (n_methods - 1) / 2) * 0.13
                vals = [_grp_annual[key][vt] for vt in vartypes]
                ax_b.scatter(
                    xv + off, vals, color=_sc[label], s=40, zorder=3, alpha=0.55
                )
                ax_b.plot(xv + off, vals, color=_sc[label], lw=0.7, alpha=0.3, zorder=2)
        ax_b.set_xticks(xv)
        ax_b.set_xticklabels(vartypes, fontsize=8, rotation=20, ha="right")
        ax_b.set_ylabel(
            f"QQ-MAE sum over {len(MATCH_COUNTRIES)} countries (GWh/day)\n(lower = better)",
            color="0.4",
        )
    else:
        xv = np.arange(len(PLOT_VARS))
        var_short = [pretty_var(v) for v in PLOT_VARS]
        # Wrap long labels
        var_short = [v.replace(": ", ":\n") for v in var_short]
        if mode == "seasonal":
            marker_styles = {season_list[0]: "o", season_list[1]: "x"}
            for i, (key, label) in enumerate(zip(_sk, _sl)):
                off = (i - (n_methods - 1) / 2) * 0.13
                for sn, marker in marker_styles.items():
                    alpha = 0.55 if sn == season_list[0] else 0.35
                    size = 40 if sn == season_list[0] else 28
                    ax_b.scatter(
                        xv + off,
                        [qq_mae_seasonal[sn][key][v] for v in PLOT_VARS],
                        color=_sc[label],
                        s=size,
                        zorder=3,
                        alpha=alpha,
                        marker=marker,
                    )
            season_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="0.5",
                    markersize=6,
                    linestyle="none",
                    label=season_list[0],
                ),
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="0.5",
                    markersize=5,
                    linestyle="none",
                    alpha=0.6,
                    label=season_list[1],
                ),
            ]
            ax_b.legend(handles=season_handles, loc="best", fontsize=7, framealpha=0.8)
        else:  # annual
            for i, (key, label) in enumerate(zip(_sk, _sl)):
                off = (i - (n_methods - 1) / 2) * 0.13
                ax_b.scatter(
                    xv + off,
                    [qq_mae_annual[key][v] for v in PLOT_VARS],
                    color=_sc[label],
                    s=40,
                    zorder=3,
                    alpha=0.55,
                )
                ax_b.plot(
                    xv + off,
                    [qq_mae_annual[key][v] for v in PLOT_VARS],
                    color=_sc[label],
                    lw=0.7,
                    alpha=0.3,
                    zorder=2,
                )
        ax_b.set_xticks(xv)
        ax_b.set_xticklabels(var_short, fontsize=8)
        ax_b.set_ylabel("QQ-MAE (GWh/day)\n(lower = better)", color="0.4")

    ax_b.tick_params(colors="0.4")
    ax_b.set_title(
        "(b) Deviations in marginal distributions",
        fontweight="bold",
        fontsize=10,
        color="0.4",
    )
    ax_b.text(
        0.5,
        1.0,
        "contributing to SWD via univariate projections",
        transform=ax_b.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.5",
        style="italic",
    )
    ax_b.set_ylim(bottom=0)

    # ── (c) Correlation deviation ──
    if mode == "seasonal":
        x_c = np.arange(len(season_list))
        for i, (key, label) in enumerate(zip(_sk, _sl)):
            vals = [corr_dev[key][sn] for sn in season_list]
            ax_c.bar(
                x_c + i * bw - (n_methods - 1) * bw / 2,
                vals,
                bw,
                color=_sc[label],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.5,
            )
        ax_c.set_xticks(x_c)
        ax_c.set_xticklabels(season_list, fontsize=9)
    else:  # annual
        x_c = np.arange(n_methods)
        for i, (key, label) in enumerate(zip(_sk, _sl)):
            ax_c.bar(
                x_c[i],
                corr_dev[key]["Annual"],
                0.6,
                color=_sc[label],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.5,
            )
        ax_c.set_xticks(x_c)
        ax_c.set_xticklabels(_sl, fontsize=9, rotation=30, ha="right")

    ax_c.set_ylabel("Mean |Δcorrelation| (−)\n(lower = better)", color="0.4")
    ax_c.tick_params(colors="0.4")
    ax_c.set_title(
        "(c) Deviations in correlation structure",
        fontweight="bold",
        fontsize=10,
        color="0.4",
    )
    ax_c.text(
        0.5,
        1.0,
        "contributing to SWD via cross-variable projections",
        transform=ax_c.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.5",
        style="italic",
    )

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=_sc[m]) for m in _sl]
    fig.legend(
        handles,
        _sl,
        loc="lower center",
        ncol=n_methods,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR, f"summary_figure_{mode}.pdf"),
        bbox_inches="tight",
        dpi=300,
    )


# %%
# _make_summary_figure(mode="annual")

_make_summary_figure(mode="seasonal")


# %%
# To how many random years is the best draw equivalent?

num_years_range = [5] + list(range(10, 161, 10)) + [168]
hydro_years = range(2016, 2044)
model_years = list(product(MODELS, hydro_years))

n_repeats = 25

# Store all scores for each num_years
all_scores = {num_years: [] for num_years in num_years_range}

for repeat in tqdm(range(n_repeats)):
    rng_repeat = np.random.default_rng(42 + repeat)
    for num_years in num_years_range:
        random_modelyears = build_random_df(
            df_ref, model_years, num_years, rng_repeat, "Hydrological_year"
        )[1]

        # score = wasserstein_score(
        #     df_ref,
        #     random_modelyears,
        #     vars_non_nan,
        #     year_col="Hydrological_year",
        #     n_projections=50,
        # )

        score = wasserstein_seasonal_score(
            df_ref,
            random_modelyears,
            vars_non_nan,
            year_col="Hydrological_year",
            n_projections=50,
        )
        all_scores[num_years].append(score)

# Calculate statistics
mean_scores = [np.mean(all_scores[ny]) for ny in num_years_range]
median_scores = [np.median(all_scores[ny]) for ny in num_years_range]
min_scores = [np.min(all_scores[ny]) for ny in num_years_range]
std_scores = [np.std(all_scores[ny]) for ny in num_years_range]

plt.figure(figsize=(10, 6))
plt.errorbar(
    list(num_years_range),
    mean_scores,
    yerr=std_scores,
    fmt="o",
    capsize=5,
    label="Random sampling (mean ± std)",
)
plt.scatter(
    list(num_years_range),
    min_scores,
    color="orange",
    label="Random sampling (min)",
)
plt.scatter(
    list(num_years_range),
    median_scores,
    marker="x",
    color="red",
    zorder=5,
    label="Random sampling (median)",
)
plt.plot(list(num_years_range), median_scores, color="red")
plt.axhline(
    df_results["wasserstein"].min(),
    color="green",
    label=f"Best score from year selection ({SELECTED_YEARS} years)",
)
plt.xlabel("Number of years")
plt.ylabel("SSWD (-)")
plt.ylim(0, None)

plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "equivalent_to_N_years.pdf"), bbox_inches="tight")

# %%
