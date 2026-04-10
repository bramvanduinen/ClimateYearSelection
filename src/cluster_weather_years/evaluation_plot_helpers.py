"""
Plotting helpers for climate year selection evaluation.
All used in 03_evaluation_plots.py

Provides:
  - Method display config (METHOD_CONFIG, _VAR_SUFFIX_MAP, pretty_var, ...)
  - Plot utility functions (_new_axes, _add_panel_labels, ...)
  - Per-variable plots (plot_distributions, plot_ecdf, plot_qq_signed, ...)
  - Correlation matrix plots (_plot_correlation_panels)
  - Three-panel summary figure (make_summary_figure and its sub-panels)
  - Pareto scatter (plot_results_with_pareto)
  - Equivalent-to-N-years analysis (plot_equivalent_to_n_years)
"""

import os
from itertools import product

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

from cluster_weather_years.constants import MODELS, PROBS, SEASONS
from cluster_weather_years.cost_functions import wasserstein_seasonal_score
from cluster_weather_years.evaluation import (
    _ecdf,
    _filter_season,
    _qq_diff_and_mae,
)
from cluster_weather_years.filtered_random_search import build_random_df

# ─────────────────────────────────────────────────────────────────────────────
# Method display config
# ─────────────────────────────────────────────────────────────────────────────

# (internal_key, short_label, color)
METHOD_CONFIG = [
    ("Best SA Subset", "SA", "#4daf4a"),  # green
    ("Best K-Medoids Subset", "K-Medoids", "#984ea3"),  # purple
    ("Best FRS Subset", "FRS", "#ff7f00"),  # orange
    ("Best Random Search Subset", "Random", "#377eb8"),  # blue
    ("ERAA25 Subset", "ERAA25 (36 yr)", "#e41a1c"),  # red
]

# Substring patterns (lowercase) that map selection_source → method key
_SOURCE_KEY_MAP = [
    ("simulated_annealing", "Best SA Subset"),
    ("k_medoids", "Best K-Medoids Subset"),
    ("frs", "Best FRS Subset"),
    ("random_search", "Best Random Search Subset"),
    ("eraa", "ERAA25 Subset"),
]
_KEY_TO_LABEL = {key: label for key, label, _ in METHOD_CONFIG}
_KEY_TO_COLOR = {key: color for key, _, color in METHOD_CONFIG}

_VAR_SUFFIX_MAP = {
    "OFF_prod": "Offshore \n wind prod.",
    "ON_prod": "Onshore \n wind prod.",
    "SOLAR_prod": "Solar prod.",
    "HRO_prod": "Run-of-river \n prod.",
    "HPO_prod": "Pondage \n run-of-river prod.",
    "HRG_prod": "Reservoir \n hydro prod.",
    "Demand": "Demand",
    "Residual_load": "Residual \n load",
}


def pretty_var(raw_name: str) -> str:
    """Map e.g. 'NL_OFF_prod' -> 'NL: Offshore\\nwind prod.'."""
    for suffix, label in _VAR_SUFFIX_MAP.items():
        if raw_name.endswith(suffix):
            country = raw_name[: -(len(suffix) + 1)]
            return f"{country}: {label}"
    return raw_name


def _source_style(source):
    """Return (display_label, color) for a selection_source string."""
    s = str(source).lower()
    for substr, key in _SOURCE_KEY_MAP:
        if substr in s:
            return _KEY_TO_LABEL[key], _KEY_TO_COLOR[key]
    return source, "gray"


def _active_methods(include_eraa25: bool = True):
    """Return METHOD_CONFIG filtered by the include_eraa25 flag."""
    if include_eraa25:
        return METHOD_CONFIG
    return [m for m in METHOD_CONFIG if m[0] != "ERAA25 Subset"]


# ─────────────────────────────────────────────────────────────────────────────
# Plot utility functions
# ─────────────────────────────────────────────────────────────────────────────


def _new_axes(nrows: int = 3, ncols: int = 2, figsize=(14, 12), sharey: bool = False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    return fig, axes.flatten()


def _add_panel_labels(axes, n_panels: int):
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


def _apply_shared_axis_labels(
    axes, n_panels: int, ncols: int, xlabel: str, ylabel: str
):
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


def _hide_unused_axes(axes, n_used: int, n_total: int):
    for j in range(n_used, n_total):
        axes[j].axis("off")


def _group_qq_mae_by_vartype(qq_mae_dict: dict) -> dict:
    """Sum QQ-MAE values across countries, grouped by variable type.

    Uses _VAR_SUFFIX_MAP to identify variable types; preserves its order.
    Useful when there are many variables across multiple countries.
    """
    grouped = {}
    for suffix, label in _VAR_SUFFIX_MAP.items():
        matching = [v for v in qq_mae_dict if v.endswith(suffix)]
        if matching:
            grouped[label] = sum(qq_mae_dict[v] for v in matching)
    return grouped


def _build_plot_series(df_ref: pd.DataFrame, active_subsets: dict) -> list:
    """Build [(label, df, color), ...] for distribution/QQ/ECDF plots."""
    color_map = {k: c for k, _, c in METHOD_CONFIG}
    series = [("Reference", df_ref, "black")]
    for name in active_subsets:
        series.append((name, active_subsets[name]["df"], color_map[name]))
    return series


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
    df_results, metric_1, metric_2, label_m1=None, label_m2=None, save_dir=None
):
    """Scatter of metric_1 vs metric_2, with Pareto front highlighted."""
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

    ax.set_xlabel(
        f"{label_m1} (lower is better)" if label_m1 else f"{metric_1} (lower is better)"
    )
    ax.set_ylabel(
        f"{label_m2} (lower is better)" if label_m2 else f"{metric_2} (lower is better)"
    )
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-variable plot scaffold and concrete plot functions
# ─────────────────────────────────────────────────────────────────────────────


def plot_per_variable(
    plot_vars: list,
    plot_var_labels: list,
    per_var_fn,
    xlabel: str,
    ylabel: str,
    nrows: int = 3,
    ncols: int = 2,
    sharey: bool = False,
    symmetric_ylim: bool = False,
    bottom_zero: bool = False,
):
    """Scaffold for multi-panel per-variable figures.

    Calls ``per_var_fn(ax, var, var_label, is_first_panel)`` for each variable,
    then applies shared labels, panel letters, and optional y-limit post-processing.

    Parameters
    ----------
    symmetric_ylim
        If True, sets all ylims to [-ymax, ymax] after drawing (signed QQ plots).
    bottom_zero
        If True, sets all ylims to [0, ymax] after drawing (absolute QQ plots).
    """
    n_panels = len(plot_vars)
    fig, axes = _new_axes(nrows=nrows, ncols=ncols, sharey=sharey)

    for i, (var, vlabel) in enumerate(zip(plot_vars, plot_var_labels)):
        per_var_fn(axes[i], var, vlabel, i == 0)

    _hide_unused_axes(axes, n_panels, nrows * ncols)
    _apply_shared_axis_labels(axes, n_panels, ncols, xlabel, ylabel)
    _add_panel_labels(axes, n_panels)

    if symmetric_ylim:
        ymax = max(np.max(np.abs(ax.get_ylim())) for ax in axes[:n_panels])
        for ax in axes[:n_panels]:
            ax.set_ylim(-ymax, ymax)
    elif bottom_zero:
        ymax = max(ax.get_ylim()[1] for ax in axes[:n_panels])
        for ax in axes[:n_panels]:
            ax.set_ylim(0, ymax)

    fig.tight_layout()
    plt.show()


def plot_distributions(
    plot_series: list,
    plot_vars: list,
    plot_var_labels: list,
    nrows: int = 3,
    ncols: int = 2,
):
    """Histogram distributions for each variable."""

    def _draw(ax, var, var_label, is_first):
        values_all = np.concatenate(
            [df[var].dropna().values for _, df, _ in plot_series]
        )
        bins = np.linspace(values_all.min(), values_all.max(), 51)
        for label, df_sel, color in plot_series:
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
        ax.set_title(var_label)
        if is_first:
            ax.legend(loc="best")

    plot_per_variable(
        plot_vars,
        plot_var_labels,
        _draw,
        xlabel="Value (GWh/day)",
        ylabel="Density",
        nrows=nrows,
        ncols=ncols,
    )


def plot_ecdf(
    plot_series: list,
    plot_vars: list,
    plot_var_labels: list,
    nrows: int = 3,
    ncols: int = 2,
):
    """Empirical CDF for each variable."""

    def _draw(ax, var, var_label, is_first):
        x_all = []
        for label, df_sel, color in plot_series:
            x, y = _ecdf(df_sel[var].dropna())
            x_all.append(x)
            ax.step(x, y, where="post", color=color, linewidth=1.5, label=label)
        x_concat = np.concatenate(x_all)
        x_pad = 0.03 * (x_concat.max() - x_concat.min())
        ax.set_xlim(x_concat.min() - x_pad, x_concat.max() + x_pad)
        ax.set_ylim(0, 1)
        ax.set_title(var_label)
        if is_first:
            ax.legend(loc="lower right")

    plot_per_variable(
        plot_vars,
        plot_var_labels,
        _draw,
        xlabel="Value (GWh/day)",
        ylabel="Cumulative probability",
        nrows=nrows,
        ncols=ncols,
        sharey=True,
    )


def plot_qq_signed(
    plot_series: list,
    df_ref: pd.DataFrame,
    plot_vars: list,
    plot_var_labels: list,
    nrows: int = 3,
    ncols: int = 2,
):
    """Signed QQ-differences (subset − reference) with shared symmetric y-axis."""

    def _draw(ax, var, var_label, is_first):
        for label, df_sel, color in plot_series[1:]:
            qq_diff, _ = _qq_diff_and_mae(df_ref[var], df_sel[var])
            ax.plot(PROBS, qq_diff, color=color, linewidth=1.5, label=f"{label} − Ref")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        for q in (0.1, 0.5, 0.9):
            ax.axvline(q, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlim(PROBS.min(), PROBS.max())
        ax.set_title(var_label)
        if is_first:
            ax.legend(loc="upper right")

    plot_per_variable(
        plot_vars,
        plot_var_labels,
        _draw,
        xlabel="Quantile",
        ylabel=r"$Q_{sel} - Q_{ref}$ (GWh/day)",
        nrows=nrows,
        ncols=ncols,
        symmetric_ylim=True,
    )


def plot_qq_absolute(
    plot_series: list,
    df_ref: pd.DataFrame,
    plot_vars: list,
    plot_var_labels: list,
    nrows: int = 3,
    ncols: int = 2,
):
    """Absolute QQ-differences with per-method MAE reference lines."""

    def _draw(ax, var, var_label, is_first):
        for label, df_sel, color in plot_series[1:]:
            qq_diff, qq_mae = _qq_diff_and_mae(df_ref[var], df_sel[var])
            ax.plot(
                PROBS,
                np.abs(qq_diff),
                color=color,
                linewidth=1.5,
                label=f"|{label} − Ref|",
            )
            ax.axhline(qq_mae, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        for q in (0.1, 0.5, 0.9):
            ax.axvline(q, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlim(PROBS.min(), PROBS.max())
        ax.set_title(var_label)
        if is_first:
            ax.legend(loc="upper right")

    plot_per_variable(
        plot_vars,
        plot_var_labels,
        _draw,
        xlabel="Quantile",
        ylabel=r"$|Q_{sel} - Q_{ref}|$ (GWh/day)",
        nrows=nrows,
        ncols=ncols,
        bottom_zero=True,
    )


def plot_seasonal_qq(
    plot_series: list,
    df_ref: pd.DataFrame,
    plot_vars: list,
    plot_var_labels: list,
    seasons: dict,
):
    """Seasonal absolute QQ-difference panels: rows = variables, cols = seasons."""
    nrows_s = len(plot_vars)
    ncols_s = len(seasons)
    fig, axes = plt.subplots(
        nrows_s, ncols_s, figsize=(7 * ncols_s, 3.5 * nrows_s), sharey=False
    )
    axes = np.atleast_2d(axes)
    row_ymaxes = np.zeros(nrows_s)

    for col_idx, (season_name, months) in enumerate(seasons.items()):
        df_ref_s = _filter_season(df_ref, months)
        for row_idx, var in enumerate(plot_vars):
            ax = axes[row_idx, col_idx]
            for label, df_sel, color in plot_series[1:]:
                qq_diff, qq_mae = _qq_diff_and_mae(
                    df_ref_s[var], _filter_season(df_sel, months)[var]
                )
                ax.plot(
                    PROBS,
                    np.abs(qq_diff),
                    color=color,
                    linewidth=1.5,
                    label=f"|{label} − Ref|",
                )
                ax.axhline(
                    qq_mae, color=color, linestyle="--", linewidth=1.1, alpha=0.9
                )
            for q in (0.1, 0.5, 0.9):
                ax.axvline(q, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.set_xlim(PROBS.min(), PROBS.max())
            if row_idx == 0:
                ax.set_title(season_name, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(plot_var_labels[row_idx], fontsize=9)
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


# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrix plots
# ─────────────────────────────────────────────────────────────────────────────


def _plot_correlation_panels(
    corr_mats: dict,
    vars_non_nan: list,
    title_prefix: str = "",
    plot_type: str = "signed",
):
    """Plot reference correlation + deviation heatmaps for all active subsets.

    Parameters
    ----------
    corr_mats : dict
        {"Reference": corr_df, <subset_name>: corr_df, ...}
    plot_type : "signed" | "absolute"
    """
    ref_corr = corr_mats["Reference"]
    subset_names = [k for k in corr_mats if k != "Reference"]
    mask_upper = np.triu(np.ones_like(ref_corr, dtype=bool), k=1)
    pretty_labels = [pretty_var(v) for v in vars_non_nan]

    n_panels = 1 + len(subset_names)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

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


# ─────────────────────────────────────────────────────────────────────────────
# Summary figure
# ─────────────────────────────────────────────────────────────────────────────


def build_summary_config(include_eraa25: bool = True) -> dict:
    methods = _active_methods(include_eraa25)
    return {
        "keys": [k for k, _, _ in methods],
        "labels": [l for _, l, _ in methods],
        "colors": {l: c for _, l, c in methods},
    }


def _season_legend_handles(season_list: list) -> list:
    """Two Line2D handles distinguishing the two seasons (circle vs cross marker)."""
    return [
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


def _plot_panel_a_swd(ax, mode: str, metrics: dict, cfg: dict):
    """(a) Horizontal bar chart: SWD per season + SSWD, or annual SWD."""
    n_methods = len(cfg["labels"])
    y = np.arange(n_methods)
    bar_h = 0.25

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
                    fontsize=9,
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
                fontsize=9,
                fontweight="bold",
                color="0.3",
            )

        # HACK: If ERAA25 not included (then n-methods = 4), set xlim to 1.3x max SSWD to fit legend
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
            fontsize=10,
        )

        ax.set_title(
            "(a) Sliced Wasserstein distance, per season",
            fontweight="bold",
            fontsize=12,
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
                fontsize=9,
                fontweight="bold",
                color="0.3",
            )
        ax.set_title(
            "(a) Sliced Wasserstein distance — annual", fontweight="bold", fontsize=12
        )

    ax.set_yticks(y)
    ax.set_yticklabels(cfg["labels"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("SWD (−)\n(lower = better)")


def _plot_panel_b_qqmae(
    ax,
    mode: str,
    metrics: dict,
    cfg: dict,
    plot_vars: list,
    multi_country: bool = False,
    n_countries: int = 1,
):
    """(b) Scatter of QQ-MAE per variable (or aggregated variable type) per method."""
    n_methods = len(cfg["labels"])
    marker_styles = dict(zip(metrics["season_list"], ["o", "x"]))
    alphas = dict(zip(metrics["season_list"], [0.55, 0.35]))
    sizes = dict(zip(metrics["season_list"], [40, 28]))

    if multi_country:
        if mode == "seasonal":
            grp = {
                sn: {
                    k: _group_qq_mae_by_vartype(metrics["qq_mae_seasonal"][sn][k])
                    for k in cfg["keys"]
                }
                for sn in metrics["season_list"]
            }
            vartypes = list(grp[metrics["season_list"][0]][cfg["keys"][0]].keys())
            xv = np.arange(len(vartypes))
            for i, (key, label) in enumerate(zip(cfg["keys"], cfg["labels"])):
                off = (i - (n_methods - 1) / 2) * 0.13
                for sn in metrics["season_list"]:
                    ax.scatter(
                        xv + off,
                        [grp[sn][key][vt] for vt in vartypes],
                        color=cfg["colors"][label],
                        s=sizes[sn],
                        zorder=3,
                        alpha=alphas[sn],
                        marker=marker_styles[sn],
                    )
            ax.legend(
                handles=_season_legend_handles(metrics["season_list"]),
                loc="best",
                fontsize=7,
                framealpha=0.8,
            )
        else:
            grp = {
                k: _group_qq_mae_by_vartype(metrics["qq_mae_annual"][k])
                for k in cfg["keys"]
            }
            vartypes = list(next(iter(grp.values())).keys())
            xv = np.arange(len(vartypes))
            for i, (key, label) in enumerate(zip(cfg["keys"], cfg["labels"])):
                off = (i - (n_methods - 1) / 2) * 0.13
                vals = [grp[key][vt] for vt in vartypes]
                ax.scatter(
                    xv + off,
                    vals,
                    color=cfg["colors"][label],
                    s=40,
                    zorder=3,
                    alpha=0.55,
                )
                ax.plot(
                    xv + off,
                    vals,
                    color=cfg["colors"][label],
                    lw=0.7,
                    alpha=0.3,
                    zorder=2,
                )
        ax.set_xticks(xv)
        ax.set_xticklabels(vartypes, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel(
            f"QQ-MAE sum over {n_countries} countries (GWh/day)\n(lower = better)",
            color="0.4",
        )
    else:
        var_labels = [pretty_var(v).replace(": ", ":\n") for v in plot_vars]
        xv = np.arange(len(plot_vars))
        if mode == "seasonal":
            for i, (key, label) in enumerate(zip(cfg["keys"], cfg["labels"])):
                off = (i - (n_methods - 1) / 2) * 0.13
                for sn in metrics["season_list"]:
                    ax.scatter(
                        xv + off,
                        [metrics["qq_mae_seasonal"][sn][key][v] for v in plot_vars],
                        color=cfg["colors"][label],
                        s=sizes[sn],
                        zorder=3,
                        alpha=alphas[sn],
                        marker=marker_styles[sn],
                    )
            ax.legend(
                handles=_season_legend_handles(metrics["season_list"]),
                loc="best",
                fontsize=7,
                framealpha=0.8,
            )
        else:
            for i, (key, label) in enumerate(zip(cfg["keys"], cfg["labels"])):
                off = (i - (n_methods - 1) / 2) * 0.13
                vals = [metrics["qq_mae_annual"][key][v] for v in plot_vars]
                ax.scatter(
                    xv + off,
                    vals,
                    color=cfg["colors"][label],
                    s=40,
                    zorder=3,
                    alpha=0.55,
                )
                ax.plot(
                    xv + off,
                    vals,
                    color=cfg["colors"][label],
                    lw=0.7,
                    alpha=0.3,
                    zorder=2,
                )
        ax.set_xticks(xv)
        ax.set_xticklabels(var_labels, fontsize=8)
        ax.set_ylabel("QQ-MAE (GWh/day)\n(lower = better)", color="0.4")

    ax.tick_params(colors="0.4")
    ax.set_title(
        "(b) Deviations in marginal distributions",
        fontweight="bold",
        fontsize=10,
        color="0.4",
    )
    ax.text(
        0.5,
        1.0,
        "contributing to SWD via univariate projections",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.5",
        style="italic",
    )
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)


def _plot_panel_c_corrdev(ax, mode: str, metrics: dict, cfg: dict):
    """(c) Grouped bar chart: mean absolute correlation deviation per season or method."""
    n_methods = len(cfg["labels"])
    bw = 0.15

    if mode == "seasonal":
        x_c = np.arange(len(metrics["season_list"]))
        for i, (key, label) in enumerate(zip(cfg["keys"], cfg["labels"])):
            vals = [metrics["corr_dev"][key][sn] for sn in metrics["season_list"]]
            ax.bar(
                x_c + i * bw - (n_methods - 1) * bw / 2,
                vals,
                bw,
                color=cfg["colors"][label],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xticks(x_c)
        ax.set_xticklabels(metrics["season_list"], fontsize=9)
    else:
        x_c = np.arange(n_methods)
        for i, (key, label) in enumerate(zip(cfg["keys"], cfg["labels"])):
            ax.bar(
                x_c[i],
                metrics["corr_dev"][key]["Annual"],
                0.6,
                color=cfg["colors"][label],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xticks(x_c)
        ax.set_xticklabels(cfg["labels"], fontsize=9, rotation=30, ha="right")

    ax.set_ylabel("Mean |Δcorrelation| (−)\n(lower = better)", color="0.4")
    ax.tick_params(colors="0.4")
    ax.set_title(
        "(c) Deviations in correlation structure",
        fontweight="bold",
        fontsize=10,
        color="0.4",
    )
    ax.text(
        0.5,
        1.0,
        "contributing to SWD via cross-variable projections",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.5",
        style="italic",
    )


def make_summary_figure(
    mode: str,
    metrics: dict,
    cfg: dict,
    plot_vars: list,
    save_dir: str,
    multi_country: bool = False,
    n_countries: int = 1,
):
    """Three-panel summary figure: (a) SWD bars, (b) QQ-MAE scatter, (c) correlation deviation.

    Parameters
    ----------
    mode : "seasonal" | "annual"
    """
    fig = plt.figure(figsize=(13, 7.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.38, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    _plot_panel_a_swd(ax_a, mode, metrics, cfg)
    _plot_panel_b_qqmae(ax_b, mode, metrics, cfg, plot_vars, multi_country, n_countries)
    _plot_panel_c_corrdev(ax_c, mode, metrics, cfg)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=cfg["colors"][m]) for m in cfg["labels"]]
    fig.legend(
        handles,
        cfg["labels"],
        loc="lower center",
        ncol=len(cfg["labels"]),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"summary_figure_{mode}.pdf"),
        bbox_inches="tight",
        dpi=300,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Equivalent-to-N-years
# ─────────────────────────────────────────────────────────────────────────────


def plot_equivalent_to_n_years(
    df_results: pd.DataFrame,
    df_ref: pd.DataFrame,
    vars_non_nan: list,
    selected_years: int,
    save_dir: str,
):
    """Estimate how many random draws are needed to match the best selection score.

    Parameters
    ----------
    selected_years : int
        Number of years in the best subset (used only for the figure label).
    """
    num_years_range = [1, 3, 5] + list(range(10, 181, 10))
    hydro_years = range(2016, 2046)
    model_years = list(product(MODELS, hydro_years))
    n_repeats = 25

    all_scores = {ny: [] for ny in num_years_range}
    for repeat in tqdm(range(n_repeats)):
        rng = np.random.default_rng(42 + repeat)
        for num_years in num_years_range:
            random_my = build_random_df(
                df_ref, model_years, num_years, rng, "Hydrological_year"
            )[1]
            score = wasserstein_seasonal_score(
                df_ref,
                random_my,
                vars_non_nan,
                year_col="Hydrological_year",
                n_projections=50,
            )
            all_scores[num_years].append(score)

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
        list(num_years_range), min_scores, color="orange", label="Random sampling (min)"
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
        label=f"Best score from year selection ({selected_years} years)",
    )
    plt.xlabel("Number of years")
    plt.ylabel("SSWD (-)")
    plt.ylim(0, None)
    plt.xlim(0, None)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(0, 181, 10))
    plt.savefig(
        os.path.join(save_dir, "equivalent_to_N_years.pdf"), bbox_inches="tight"
    )

    return num_years_range, all_scores
