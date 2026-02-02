import os
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np

from cluster_weather_years.constants import BASE_OUTPUT


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


def plot_results_with_pareto(df_results, metric_1, metric_2, save_dir=None):
    """Plot results with Pareto front highlighted."""
    pareto_mask_2d = calc_pareto_mask(df_results, cols=(metric_1, metric_2))
    pareto_front_2d = df_results.loc[pareto_mask_2d]

    # Count composition of Pareto front
    pareto_composition = pareto_front_2d["selection_source"].value_counts().to_dict()
    pareto_label_parts = [
        f"{count}x {source}" for source, count in pareto_composition.items()
    ]
    pareto_label = "Pareto front: " + ", ".join(pareto_label_parts)

    # Create scatter plot with proper labels per selection source
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.scatter(
        pareto_front_2d[metric_1], pareto_front_2d[metric_2], c="k", label=pareto_label
    )

    for source in df_results["selection_source"].unique():
        mask = df_results["selection_source"] == source
        if source == "k_medoids":
            marker = "x"
            size = 30
        else:
            marker = "o"
            size = 10
        ax.scatter(
            df_results.loc[mask, metric_1],
            df_results.loc[mask, metric_2],
            label=source,
            marker=marker,
            s=size,
            alpha=0.6,
        )

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    # ax.set_xlabel("SWD Worst Season (lower is better) [-]")
    # ax.set_ylabel("SWD Overall (lower is better) [-]")
    ax.set_xlabel(f"{metric_1} (lower is better)")
    ax.set_ylabel(f"{metric_2} (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/pareto_front_{metric_1}_{metric_2}.png", dpi=150)


def plot_lollipop_by_country(
    df_results, df_country_results, countries, metric, metric_label, save_dir=None
):
    """Plot lollipop chart of results by country."""
    # take 5 best SA selections
    idx_best = df_results.sort_values(metric).index[:5]
    df_best = df_results.loc[idx_best]

    # Color palette for SA markers
    sa_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    y = np.arange(len(countries))
    fig, ax = plt.subplots(figsize=(4, 6))

    for i, country in enumerate(countries):
        # df_country = df_country_results.query("country == @country")
        # df_country["model_years"] = df_country["model_years"].apply(lambda x: str(x))
        df_country = df_country_results.query("country == @country").copy()
        df_country["model_years"] = df_country["model_years"].astype(str)
        wass_country_best = df_best.merge(
            df_country[["model_years", metric]], on="model_years"
        )[f"{metric}_y"].to_numpy()

        rnd = df_country.query("selection_source == 'random_draws'")[f"{metric}"].mean()

        k_med = df_country.query("selection_source == 'k_medoids'")[f"{metric}"].values[
            0
        ]

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
