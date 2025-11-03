# %%
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cluster_weather_years.constants import (
    COLORS,
    COUNTRIES,
    MIN_WINDOW_ROLLING,
    MODELS,
    MODELS_FNAME,
    OUTPUT_DIR_SOLAR,
    OUTPUT_DIR_TEMP,
    OUTPUT_DIR_WIND,
    THRES_CDD,
    THRES_HDD,
    WINDOW_ROLLING,
    YEAR_COL,
    YEARS,
)


# %% building dataframes of subsets
def build_df_centroids(df_daily, selected, mode="Hydrological_year"):
    """
    Build df with daily data of selected years
    """
    frames_centroids = []
    for i, (model, year) in enumerate(selected):
        # model_centroid, year_centroid = row["model"], row["Date"]
        if mode == "Hydrological_year":
            df_centroid = df_daily.query(
                f"model == '{model}' and Hydrological_year == {year}"
            )
        elif mode == "Calendar_year":
            df_centroid = df_daily.query(
                f"model == '{model}' and Calendar_year == {year}"
            )
        else:
            raise ValueError("Mode not recognized.")

        frames_centroids.append(df_centroid)

    return pd.concat(frames_centroids, ignore_index=True)


def build_df_random(
    df_idx: pd.DataFrame,
    keys: np.ndarray,
    rng: np.random.Generator,
    num_random: int,
) -> pd.DataFrame:
    """Build dataframe with a randomly drawn subset of model-years"""
    sel = rng.choice(len(keys), size=num_random, replace=False)
    return df_idx.loc[keys[sel]].reset_index()


# %%
def build_df():
    """Build dataframe for COUNTRIES based on all raw CSV files downloaded from PECD."""
    frames_hourly = []

    for model_fname, model_name in zip(MODELS_FNAME, MODELS):
        for year in YEARS:
            file_wind_off = f"P_CMI6_{model_fname}_WOF_NA---_Pecd_NUT0_S{year}01010000_E{year}12312300_CFR_TIM_01h_NA-_noc_org_20_SP245_ReGrB_PhM04_PECD4.2_fv1.csv"
            file_wind_on = f"P_CMI6_{model_fname}_WON_NA---_Pecd_NUT0_S{year}01010000_E{year}12312300_CFR_TIM_01h_NA-_noc_org_30_SP245_ReGrB_PhM04_PECD4.2_fv1.csv"
            file_solar = f"P_CMI6_{model_fname}_SPV_0000m_Pecd_NUT0_S{year}01010000_E{year}12312300_CFR_TIM_01h_NA-_noc_org_61_SP245_NA---_PhM03_PECD4.2_fv1.csv"
            file_temp = f"P_CMI6_{model_fname}_TA-_0002m_Pecd_NUT0_S{year}01010000_E{year}12312300_INS_TIM_01h_NA-_cdf_org_NA_SP245_NA---_NA---_PECD4.2_fv1.csv"

            df_wind_off = pd.read_csv(
                os.path.join(OUTPUT_DIR_WIND, file_wind_off),
                skiprows=52,
            )
            df_wind_on = pd.read_csv(
                os.path.join(OUTPUT_DIR_WIND, file_wind_on),
                skiprows=52,
            )

            df_solar = pd.read_csv(
                os.path.join(OUTPUT_DIR_SOLAR, file_solar),
                skiprows=52,
            )

            df_temp = pd.read_csv(
                os.path.join(OUTPUT_DIR_TEMP, file_temp),
                skiprows=52,
            )

            df_year = pd.DataFrame(
                {
                    "Date": df_wind_off["Date"],
                    **{
                        f"{country}_OFF": df_wind_off[f"{country}_OFF"]
                        for country in COUNTRIES
                    },
                    **{
                        f"{country}_ON": df_wind_on[f"{country}"]
                        for country in COUNTRIES
                    },
                    **{
                        f"{country}_SOLAR": df_solar[f"{country}"]
                        for country in COUNTRIES
                    },
                    **{
                        f"{country}_TEMP": df_temp[f"{country}"]
                        for country in COUNTRIES
                    },
                    "model": model_name,
                }
            )

            frames_hourly.append(df_year)

    df_hourly = pd.concat(frames_hourly, ignore_index=True)
    df_hourly["Date"] = pd.to_datetime(df_hourly["Date"])

    df_daily = (
        df_hourly.groupby(["model", pd.Grouper(key="Date", freq="D")])
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["model", "Date"])
    )
    df_daily["Hydrological_year"] = (
        df_daily["Date"] - pd.DateOffset(months=3)
    ).dt.year  # start hydrological year 1st of April
    df_daily["Calendar_year"] = df_daily["Date"].dt.year

    # compute HDD and CDD

    CDD = df_daily["NL_TEMP"] - THRES_CDD
    CDD = CDD.where(CDD > 0, 0)
    CDD_val = CDD.sum()
    df_daily["CDD"] = CDD

    HDD = THRES_HDD - df_daily["NL_TEMP"]
    HDD = HDD.where(HDD > 0, 0)
    HDD_val = HDD.sum()
    df_daily["HDD"] = HDD

    num_cols = df_daily.select_dtypes(include="number").columns

    # perform 30-day rolling mean
    df_nl_rolling = (
        df_daily.groupby("model")[num_cols]
        .rolling(window=WINDOW_ROLLING, min_periods=MIN_WINDOW_ROLLING, center=False)
        .mean()
    ).reset_index()

    df_nl_rolling.pop("level_1")
    df_nl_rolling["Date"] = df_daily["Date"]

    df_nl_rolling = df_nl_rolling.set_index(["model", "Date"])
    df_daily = df_daily.set_index(["model", "Date"])

    return df_daily, df_nl_rolling


def standardize_df(
    df: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    """
    Standardise `variables` to zero mean and unit variance (z-score).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    variables : Sequence[str]
        Columns to standardise.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with standardised variables.
    """
    df_std = df.copy()
    df_std[variables] = (df_std[variables] - df_std[variables].mean()) / df_std[
        variables
    ].std()
    return df_std


def calc_clim_and_year_data_multi(df: pd.DataFrame, column: str):
    """
    Build a (model, year) × day-of-year matrix and per-model climatology.

    Parameters
    ----------
    df : pandas.DataFrame
        MultiIndex on ("model", "Date") with daily rows.
    column : str
        Column to analyze.

    Returns
    -------
    nl_matrix : numpy.ndarray
        Shape (n_model_years, 366).
    row_index : pandas.MultiIndex
        (model, year) index for nl_matrix rows.
    clim_mean : pandas.DataFrame
        Per-model mean climatology; index=model, columns=1..366.
    clim_std : pandas.DataFrame
        Per-model std climatology (ddof=0); index=model, columns=1..366.
    days : numpy.ndarray
        Day-of-year array 1..366.
    """
    idx = df.index
    models = idx.get_level_values("model")
    dates = idx.get_level_values("Date")

    tmp = pd.DataFrame(
        {
            "model": models,
            "year": dates.year,
            "Hydrological_year": df["Hydrological_year"].values,
            "doy": dates.dayofyear,
            "val": df[column].to_numpy(),
        }
    )

    days = np.arange(1, 367)
    mat_df = (
        tmp.pivot_table(
            index=["model", YEAR_COL], columns="doy", values="val", aggfunc="first"
        )
        .reindex(columns=days)
        .sort_index()
    )

    nl_matrix = mat_df.to_numpy()
    clim_mean = mat_df.mean(numeric_only=True)
    clim_std = mat_df.std(ddof=0, numeric_only=True)

    return nl_matrix, mat_df.index, clim_mean, clim_std, days


def calc_clim_per_cluster(df_nl_rolling, yearly_stats, column, num_clusters):
    nl_matrix, row_index, clim_mean, clim_std, days = calc_clim_and_year_data_multi(
        df_nl_rolling, column=column
    )
    cluster_vals = yearly_stats["cluster_med"].values
    clim_per_cluster = np.empty((num_clusters, 366))
    for i in range(num_clusters):
        cluster_mask = cluster_vals == i
        cluster_data = nl_matrix[cluster_mask]
        clim_per_cluster[i] = np.nanmean(cluster_data, axis=0)
    return nl_matrix, row_index, clim_per_cluster, clim_mean, clim_std, days


# PLOTTING FUNCTIONS


def plot_clim_and_cum_anom(df_nl_rolling, column):
    # nl_matrix, row_index, clim_mean, clim_std, days = calc_clim_and_year_data(
    #     df_nl_rolling, column
    # )
    nl_matrix, row_index, clim_mean, clim_std, days = calc_clim_and_year_data_multi(
        df_nl_rolling, column=column
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top plot: time series and climatology
    # for i, year in enumerate(YEARS):
    for i, (model, year) in enumerate(row_index):
        axes[0].plot(days, nl_matrix[i], label=f"{model}, {year}", zorder=1)
    axes[0].plot(days, clim_mean, color="k", label="Clim.", linewidth=2, zorder=5)
    axes[0].fill_between(
        days,
        clim_mean - clim_std,
        clim_mean + clim_std,
        color="grey",
        alpha=0.8,
        label="±1 std",
        zorder=3,
    )
    axes[0].set_xlim(days[0], days[-1])
    if column in ["NL_ON", "NL_OFF"]:
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel(f"{column} cf (-)")
        axes[1].set_ylabel("Cumulative anomaly cf (-)")
    elif column == "NL_SOLAR":
        axes[0].set_ylim(0, 0.3)
        axes[0].set_ylabel(f"{column} cf (-)")
        axes[1].set_ylabel("Cumulative anomaly cf (-)")
    else:
        axes[0].set_ylabel(f"{column}")
        axes[1].set_ylabel("Cumulative anomal")

    axes[0].set_title(
        f"{column} by Day of Year for Each Year with Climatology +- 1 std"
    )
    # axes[0].legend(loc="upper right")

    # Bottom plot: cumulative anomalies

    for i, (model, year) in enumerate(row_index):
        axes[1].plot(
            days,
            np.nancumsum(nl_matrix[i] - clim_mean),
            label=f"{model}, {year}",
            zorder=1,
        )
    axes[1].axhline(y=0, color="k", linestyle="--")
    axes[1].set_xlim(days[0], days[-1])
    axes[1].set_xlabel("Day of Year")
    axes[1].set_title(f"{column} cumulative anomaly")
    # axes[1].legend(loc="lower left")


def plot_clim_per_cluster(
    df_nl_rolling, yearly_stats, centroid_years, column, num_clusters
):
    nl_matrix, row_index, clim_per_cluster, clim_mean, clim_std, days = (
        calc_clim_per_cluster(df_nl_rolling, yearly_stats, column, num_clusters)
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top plot: time series and climatology
    # for i, year in enumerate(YEARS):
    for i, (model, year) in enumerate(row_index):
        color = COLORS[yearly_stats["cluster_med"][row_index[i]]]
        axes[0].plot(
            days,
            nl_matrix[i],
            color=color,
            label=f"{model}, {year}",
            zorder=1,
            alpha=0.5,
        )

    for i in range(num_clusters):
        color = COLORS[i]
        axes[0].plot(
            days, clim_per_cluster[i], color=color, label=i, linewidth=5, zorder=5
        )

    axes[0].plot(days, clim_mean, color="k", label="Clim.", linewidth=2, zorder=5)
    axes[0].fill_between(
        days,
        clim_mean - clim_std,
        clim_mean + clim_std,
        color="grey",
        alpha=0.8,
        label="±1 std",
        zorder=3,
    )
    axes[0].set_xlim(days[0], days[-1])
    if column in ["NL_ON", "NL_OFF"]:
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel(f"{column} cf (-)")
        axes[1].set_ylabel("Cumulative anomaly cf (-)")
    elif column == "NL_SOLAR":
        axes[0].set_ylim(0, 0.3)
        axes[0].set_ylabel(f"{column} cf (-)")
        axes[1].set_ylabel("Cumulative anomaly cf (-)")
    else:
        axes[0].set_ylabel(f"{column}")
        axes[1].set_ylabel("Cumulative anomaly")

    axes[0].set_title(
        f"{column} by Day of Year for Each Year with Climatology +- 1 std"
    )
    # axes[0].legend(loc="upper right")

    # Bottom plot: cumulative anomalies

    used_labels = set()
    for i, (model, year) in enumerate(row_index):
        cluster_id = yearly_stats["cluster_med"][row_index[i]]
        color = COLORS[cluster_id]
        if (model, year) in centroid_years.index:
            # label = f"Centroid year {cluster_id}"
            linewidth = 6
            zorder = 5
            alpha = 1
        else:
            label = f"Cluster {cluster_id}" if cluster_id not in used_labels else None
            linewidth = 1
            zorder = 1
            alpha = 0.5

        axes[1].plot(
            days,
            np.nancumsum(nl_matrix[i] - clim_mean),
            color=color,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
            alpha=alpha,
        )
        used_labels.add(cluster_id)

    axes[1].axhline(y=0, color="k", linestyle="--")
    axes[1].set_xlim(days[0], days[-1])
    axes[1].set_xlabel("Day of Year")
    axes[1].set_title(f"{column} cumulative anomaly")
    # axes[1].legend(loc="lower left")
    # handles, labels = axes[1].get_legend_handles_labels()

    # # Sort by cluster number (assumes labels look like "Cluster X")
    # sorted_handles_labels = sorted(
    #     zip(handles, labels),
    #     key=lambda hl: int(hl[1].split()[-1]) if "Cluster" in hl[1] else -1,
    # )
    # handles, labels = zip(*sorted_handles_labels)
    # for h in handles:
    #     h.set_linewidth(6)
    #     h.set_alpha(1)
    # axes[1].legend(handles, labels, loc="lower left")

    # Build proxy handles in desired order (Cluster 0 … Cluster 4), thick for legend only
    legend_labels = [f"Cluster {i}" for i in range(5)]
    legend_handles = [
        Line2D([0], [0], color=COLORS[i], linewidth=6, alpha=1, label=legend_labels[i])
        for i in range(5)
    ]

    axes[1].legend(legend_handles, legend_labels, loc="lower left")

    # Add histogram of cumulative anomalies to the right of the second subplot

    # Compute final cumulative anomalies per year and their clusters
    final_cum_anomalies = np.nansum(nl_matrix - clim_mean.to_numpy(), axis=1)
    clusters = np.array([yearly_stats["cluster_med"][idx] for idx in row_index])

    # Create a single stacked histogram axis
    left, bottom, width, height = axes[1].get_position().bounds
    hist_ax = fig.add_axes([left + width + 0.03, bottom, 0.12, height], sharey=axes[1])

    # Prepare cluster-wise cumulative anomaly values
    cluster_vals_list = []
    cluster_labels = []
    cluster_colors = []

    for cluster_id in range(5):
        vals = final_cum_anomalies[clusters == cluster_id]
        if len(vals) == 0:
            continue
        cluster_vals_list.append(vals)
        cluster_labels.append(f"Cluster {cluster_id}")
        cluster_colors.append(COLORS[cluster_id])

    # Define common bins
    bins = np.histogram_bin_edges(final_cum_anomalies, bins=20)

    # Plot stacked histogram
    hist_ax.hist(
        cluster_vals_list,
        bins=bins,
        orientation="horizontal",
        stacked=True,
        color=cluster_colors,
        label=cluster_labels,
        edgecolor="black",
    )

    hist_ax.set_xlabel("Count")
    hist_ax.yaxis.set_tick_params(labelleft=False)


def calc_clim_and_year_data(df_nl_rolling, column):
    """OLD!!!"""
    # Prepare a matrix: rows = years, columns = dayofyear (1..366)
    years = sorted(df_nl_rolling.index.year.unique())
    max_days = 366  # to handle leap years
    nl_matrix = np.full((len(years), max_days), np.nan)

    for i, year in enumerate(years):
        group = df_nl_rolling.loc[df_nl_rolling.index.year == year, column]
        # Align by dayofyear
        for doy, value in zip(group.index.dayofyear, group.values):
            nl_matrix[i, doy - 1] = value

    # Compute climatology (mean and std, ignoring NaNs)
    clim_mean = np.nanmean(nl_matrix, axis=0)
    clim_std = np.nanstd(nl_matrix, axis=0)
    days = np.arange(1, max_days + 1)
    return nl_matrix, clim_mean, clim_std, days
