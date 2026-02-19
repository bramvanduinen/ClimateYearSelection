import os
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# fmt: off
# pylint: disable=import-error
from cluster_weather_years.constants import (
    BASE_OUTPUT,
    DROUGHT_MIN_DURATION,
    DROUGHT_PERCENTILE,
    # MATCH_COUNTRIES,
    # VARS_TO_MATCH,
)
from cluster_weather_years.helper_functions import (
    build_df_from_modelyears,
)

# pylint: enable=import-error
# fmt: on
### Statistical Evaluation ###


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
        # HACK: Some messing around to ensure model_years are strings for merging
        # TODO: implement proper handling of model_years as tuples everywhere
        df_country = df_country_results.query("country == @country").copy()
        df_country["model_years"] = df_country["model_years"].apply(str)

        # Also convert df_best model_years to string for consistent merging
        df_best_str = df_best.copy()
        df_best_str["model_years"] = df_best_str["model_years"].apply(str)

        wass_country_best = df_best_str.merge(
            df_country[["model_years", metric]], on="model_years"
        )[f"{metric}_y"].to_numpy()

        rnd = df_country.query("selection_source == 'random_draws'")[f"{metric}"].mean()

        k_med = df_country.query("selection_source == 'k_medoids'")[f"{metric}"].values[
            0
        ]

        # Safety check: skip if no matching data
        if len(wass_country_best) == 0:
            print(f"Warning: No matching model_years found for country {country}")
            continue

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


### Physical Evaluation ###
def compute_monthly_mean(df, vars):
    """Compute monthly means for given variables."""
    df = df.set_index("Date")
    monthly_means = df[vars].groupby(df.index.month).mean()
    monthly_means.index.name = "Month"
    return monthly_means


def compute_correlation_matrix(df, vars):
    """Compute correlation matrix for variables."""
    return df[vars].corr()


def compute_drought_statistics(df, country, ref_percentiles=None):
    """Analyze renewable droughts in the dataframe."""
    # --- Renewable droughts ---
    percentiles = {}
    stats = {}

    n_days = len(df)

    if f"{country}_total_renewable" in df.columns:
        renewables = df[f"{country}_total_renewable"].dropna()

        if ref_percentiles is None:
            p_drought = renewables.quantile(DROUGHT_PERCENTILE / 100)
            percentiles["renewables_drought_threshold"] = p_drought
        else:
            p_drought = ref_percentiles["renewables_drought_threshold"]

        # Find drought periods (consecutive days below threshold)
        below_threshold = (renewables < p_drought).astype(int)

        # Count drought events and their durations
        drought_events = []
        current_drought = 0

        for val in below_threshold:
            if val == 1:
                current_drought += 1
            else:
                if current_drought >= DROUGHT_MIN_DURATION:
                    drought_events.append(current_drought)
                current_drought = 0
        # Don't forget last event
        if current_drought >= DROUGHT_MIN_DURATION:
            drought_events.append(current_drought)

        stats["n_drought_events"] = len(drought_events) / (n_days / 365)  # Per year
        stats["mean_drought_duration"] = (
            np.mean(drought_events) if drought_events else 0
        )
        stats["max_drought_duration"] = np.max(drought_events) if drought_events else 0
        stats["total_drought_days_freq"] = (
            sum(drought_events) / n_days if drought_events else 0
        )
    return percentiles, stats


def perform_physical_evaluation(df_results, df_ref, countries, vars, save_dir=None):
    """
    Perform comprehensive physical evaluation for all countries in countries.
    Creates all plots and analyses for seasonal patterns, correlations, and drought metrics.
    """

    for country in countries:
        print(f"\n{'=' * 60}")
        print(f"Analyzing country: {country}")
        print(f"{'=' * 60}\n")

        # --- Setup country-specific variables ---
        country_vars = [var for var in vars if var.startswith(country + "_")]
        country_prod_vars = [var for var in country_vars if "prod" in var]
        country_vars.append(f"{country}_total_renewable")
        df_ref[f"{country}_total_renewable"] = df_ref[
            [var for var in country_prod_vars if var != f"{country}_total_renewable"]
        ].sum(axis=1)

        # Compute reference statistics
        ref_monthly_mean = compute_monthly_mean(df_ref, country_vars)
        ref_correlation_matrix = compute_correlation_matrix(df_ref, country_vars)
        ref_percentiles, ref_drought_stats = compute_drought_statistics(
            df_ref[country_vars], country
        )

        # Compute statistics for all subsets
        dfs_monthly_means = {}
        dfs_correlation_matrices = {}
        dfs_drought_stats = {}

        for i, row in df_results.iterrows():
            if type(row["model_years"]) == str:
                sel_modelyears = literal_eval(row["model_years"])
            else:
                sel_modelyears = row["model_years"]

            ds_name = row["selection_source"] + "_" + str(i + 1)
            df_sub = build_df_from_modelyears(df_ref, sel_modelyears)
            df_sub[f"{country}_total_renewable"] = df_sub[
                [
                    var
                    for var in country_prod_vars
                    if var != f"{country}_total_renewable"
                ]
            ].sum(axis=1)

            dfs_monthly_means[ds_name] = compute_monthly_mean(df_sub, country_vars)
            dfs_correlation_matrices[ds_name] = compute_correlation_matrix(
                df_sub, country_vars
            )
            dfs_drought_stats[ds_name] = compute_drought_statistics(
                df_sub[country_vars], country, ref_percentiles
            )[1]

        # --- 1. Monthly means analysis ---
        df_subsets_monthly_means_list = list(dfs_monthly_means.values())
        mean_across_subsets = (
            pd.concat(df_subsets_monthly_means_list).groupby(level=0).mean()
        )
        std_across_subsets = (
            pd.concat(df_subsets_monthly_means_list).groupby(level=0).std()
        )

        best_idx = df_results.sort_values("wasserstein_seasonal").index[0]
        # TODO: implement proper handling of model_years as tuples everywhere to avoid this hack
        if type(df_results.loc[best_idx, "model_years"]) == str:
            modelyears_best = literal_eval(df_results.loc[best_idx, "model_years"])
        else:
            modelyears_best = df_results.loc[best_idx, "model_years"]

        best_ds_name = (
            df_results.loc[best_idx, "selection_source"] + "_" + str(best_idx + 1)
        )
        df_best_monthly_means = dfs_monthly_means[best_ds_name]

        sorted_indices = df_results.sort_values("wasserstein_seasonal").index
        median_idx = sorted_indices[len(sorted_indices) // 2]

        # TODO: implement proper handling of model_years as tuples everywhere to avoid this hack
        if type(df_results.loc[median_idx, "model_years"]) == str:
            modelyears_median = literal_eval(df_results.loc[median_idx, "model_years"])
        else:
            modelyears_median = df_results.loc[median_idx, "model_years"]

        median_ds_name = (
            df_results.loc[median_idx, "selection_source"] + "_" + str(median_idx + 1)
        )
        df_median_monthly_means = dfs_monthly_means[median_ds_name]

        # Plot monthly means
        plot_vars = country_vars
        n_vars = len(plot_vars)
        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()
        month_positions = np.arange(len(ref_monthly_mean))

        for idx, var in enumerate(plot_vars):
            ax = axes[idx]
            ref_mean = ref_monthly_mean[var]
            subset_mean = mean_across_subsets[var]
            subset_std = std_across_subsets[var]
            best_mean = df_best_monthly_means[var]
            median_mean = df_median_monthly_means[var]

            ax.plot(
                month_positions,
                ref_mean.values,
                label="Reference",
                linewidth=2.5,
                color="black",
                marker="o",
                markersize=6,
            )
            ax.fill_between(
                month_positions,
                subset_mean - subset_std,
                subset_mean + subset_std,
                alpha=0.4,
                color="gray",
                label="Subset mean ± std",
            )
            ax.plot(
                month_positions,
                subset_mean.values,
                linewidth=2,
                color="gray",
                linestyle="--",
            )
            ax.plot(
                month_positions,
                median_mean.values,
                label="Median subset",
                linewidth=2.5,
                color="C1",
                marker="^",
                markersize=6,
            )
            ax.plot(
                month_positions,
                best_mean.values,
                label="Best subset",
                linewidth=2.5,
                color="C0",
                marker="s",
                markersize=6,
            )

            ax.set_xticks(month_positions)
            ax.set_xticklabels(ref_monthly_mean.index, rotation=45)
            ax.set_xlabel("Month")
            ax.set_ylabel(var)
            ax.set_title(f"{var}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        for idx in range(n_vars, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            f"Monthly Mean Comparison - {country}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

        if save_dir:
            fig.savefig(
                os.path.join(save_dir, f"{country}_monthly_means.png"),
                bbox_inches="tight",
                dpi=150,
            )

        # --- 2. Correlation analysis ---
        ref_corr_country = compute_correlation_matrix(df_ref, country_vars)
        sub_best_corr_country = compute_correlation_matrix(
            build_df_from_modelyears(df_ref, modelyears_best),
            country_vars,
        )
        sub_median_corr_country = compute_correlation_matrix(
            build_df_from_modelyears(df_ref, modelyears_median),
            country_vars,
        )

        diff_best_country = sub_best_corr_country - ref_corr_country
        diff_median_country = sub_median_corr_country - ref_corr_country

        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        sns.heatmap(
            ref_corr_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[0, 0],
            cbar=False,
        )
        axs[0, 0].set_title("Reference")

        sns.heatmap(
            sub_best_corr_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[0, 1],
            cbar=False,
        )
        axs[0, 1].set_title("Best Subset")

        sns.heatmap(
            diff_best_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Difference"},
            vmin=-0.1,
            vmax=0.1,
            ax=axs[0, 2],
            cbar=False,
        )
        axs[0, 2].set_title("Difference (Best - Ref)")

        sns.heatmap(
            ref_corr_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[1, 0],
            cbar=False,
        )
        axs[1, 0].set_title("Reference")

        sns.heatmap(
            sub_median_corr_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[1, 1],
            cbar=False,
        )
        axs[1, 1].set_title("Median Subset")

        sns.heatmap(
            diff_median_country,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Difference"},
            vmin=-0.1,
            vmax=0.1,
            ax=axs[1, 2],
            cbar=False,
        )
        axs[1, 2].set_title("Difference (Median - Ref)")

        plt.suptitle(
            f"Correlation Matrices - {country}", fontsize=14, fontweight="bold", y=0.995
        )
        plt.tight_layout()
        plt.show()
        if save_dir:
            fig.savefig(
                os.path.join(save_dir, f"{country}_correlation_matrices.png"),
                bbox_inches="tight",
                dpi=150,
            )

        # Mean and std of correlations
        corr_matrices_list = list(dfs_correlation_matrices.values())
        corr_array = np.array([df.values for df in corr_matrices_list])
        mean_corr = pd.DataFrame(
            corr_array.mean(axis=0),
            index=corr_matrices_list[0].index,
            columns=corr_matrices_list[0].columns,
        )
        std_corr = pd.DataFrame(
            corr_array.std(axis=0),
            index=corr_matrices_list[0].index,
            columns=corr_matrices_list[0].columns,
        )

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(
            mean_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={"label": "Mean Correlation"},
            vmin=-1,
            vmax=1,
            ax=axs[0],
            cbar=False,
        )
        axs[0].set_title("Mean Correlation Across All Subsets")

        sns.heatmap(
            std_corr,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar_kws={"label": "Std of Correlation"},
            ax=axs[1],
            cbar=False,
        )
        axs[1].set_title("Std Dev of Correlation Across All Subsets")

        plt.suptitle(
            f"Correlation Variability - {country}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()
        if save_dir:
            fig.savefig(
                os.path.join(save_dir, f"{country}_correlation_variability.png"),
                bbox_inches="tight",
                dpi=150,
            )

        # --- 3. Energy drought metrics ---
        top10_indices = df_results.sort_values("wasserstein_seasonal").head(10).index
        top10_names = [
            df_results.loc[idx, "selection_source"] + "_" + str(idx + 1)
            for idx in top10_indices
        ]

        labels = ["Reference"] + top10_names + [median_ds_name]
        n_drought_events = [ref_drought_stats["n_drought_events"]]
        max_drought_duration = [ref_drought_stats["max_drought_duration"]]
        mean_drought_duration = [ref_drought_stats["mean_drought_duration"]]

        for ds_name in top10_names:
            n_drought_events.append(dfs_drought_stats[ds_name]["n_drought_events"])
            max_drought_duration.append(
                dfs_drought_stats[ds_name]["max_drought_duration"]
            )
            mean_drought_duration.append(
                dfs_drought_stats[ds_name]["mean_drought_duration"]
            )

        n_drought_events.append(dfs_drought_stats[median_ds_name]["n_drought_events"])
        max_drought_duration.append(
            dfs_drought_stats[median_ds_name]["max_drought_duration"]
        )
        mean_drought_duration.append(
            dfs_drought_stats[median_ds_name]["mean_drought_duration"]
        )

        fig, axs = plt.subplots(3, 1, figsize=(14, 12))
        x_pos = np.arange(len(labels))
        colors = ["black"] + ["green"] * len(top10_names) + ["blue"]

        axs[0].bar(x_pos, n_drought_events, color=colors, alpha=0.7)
        axs[0].axhline(
            n_drought_events[0],
            color="black",
            linestyle=":",
            linewidth=2,
            label="Reference",
        )
        axs[0].set_xticks(x_pos)
        axs[0].set_xticklabels("")
        axs[0].set_ylabel("Drought Events per Year")
        axs[0].set_title("Number of Drought Events per Year")
        axs[0].grid(axis="y", alpha=0.3)

        axs[1].bar(x_pos, max_drought_duration, color=colors, alpha=0.7)
        axs[1].axhline(
            max_drought_duration[0],
            color="black",
            linestyle=":",
            linewidth=2,
            label="Reference",
        )
        axs[1].set_xticks(x_pos)
        axs[1].set_xticklabels("")
        axs[1].set_ylabel("Max Drought Duration (days)")
        axs[1].set_title("Maximum Drought Duration")
        axs[1].grid(axis="y", alpha=0.3)

        axs[2].bar(x_pos, mean_drought_duration, color=colors, alpha=0.7)
        axs[2].axhline(
            mean_drought_duration[0],
            color="black",
            linestyle=":",
            linewidth=2,
            label="Reference",
        )
        axs[2].set_xticks(x_pos)
        axs[2].set_xticklabels(labels, rotation=45, ha="right")
        axs[2].set_ylabel("Mean Drought Duration (days)")
        axs[2].set_title("Mean Drought Duration")
        axs[2].grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Energy Drought Metrics - {country} (Reference=black, Top 10 Best=green, Median=blue)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        if save_dir:
            fig.savefig(
                os.path.join(save_dir, f"{country}_drought_metrics.png"),
                bbox_inches="tight",
                dpi=150,
            )
