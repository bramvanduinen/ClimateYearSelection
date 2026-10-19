# %%
import glob
import os

import numpy as np
import pandas as pd

from climate_year_selection.constants import (
    COUNTRIES,
    HYDRO_VARS,
    INPUT_DIR_HYDRO,
    INPUT_DIR_HYDRO_RAW,
    INPUT_DIR_SOLAR,
    INPUT_DIR_TEMP,
    INPUT_DIR_WIND,
    MODELS,
    MODELS_FNAME,
    YEARS,
)


# %% building dataframes of subsets
def build_df_from_modelyears(df_daily, modelyears, mode="Hydrological_year"):
    """
    Build df with daily data of selected modelyears
    """
    frames_centroids = []
    for i, (model, year) in enumerate(modelyears):
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


def draw_random_modelyears(model_years, num_jobs, num_to_select, rng):
    """
    Draw random subsets of model-years, saved to a list-of-tuples.
    """
    random_draws_list = []

    for _ in range(num_jobs):
        selected_idx = rng.choice(
            len(model_years),
            size=num_to_select,
            replace=False,
        )
        random_draws_list.append(tuple(model_years[i] for i in selected_idx))
    return random_draws_list


def build_df_random(
    df_idx: pd.DataFrame,
    keys: np.ndarray,
    rng: np.random.Generator,
    num_random: int,
) -> pd.DataFrame:
    """Build dataframe with a randomly drawn subset of model-years"""
    sel = rng.choice(len(keys), size=num_random, replace=False)
    return df_idx.loc[keys[sel]].reset_index()


# %% Functions for building prod/demand dataframes from raw CSV files


def build_df():
    """Build dataframe for COUNTRIES based on all raw CSV files downloaded from PECD."""
    frames_hourly = []

    for model_fname, model_name in zip(MODELS_FNAME, MODELS):
        for year in YEARS:
            if (
                model_fname == "BCC-_BCCS"
            ):  # use fv2 files for this model, only those corrected for wind
                file_wind_off = f"P_CMI6_{model_fname}_WOF_NA---_Pecd_PEOF_S{year}01010000_E{year}12312300_CFR_TIM_01h_NA-_noc_org_20_SP245_ReGrB_PhM04_PECD4.2_fv2_NUTS0_level.csv"
                file_wind_on = f"P_CMI6_{model_fname}_WON_NA---_Pecd_PEON_S{year}01010000_E{year}12312300_CFR_TIM_01h_NA-_noc_org_30_SP245_ReGrB_PhM04_PECD4.2_fv2_NUTS0_level.csv"
            else:
                file_wind_off = f"P_CMI6_{model_fname}_WOF_NA---_Pecd_PEOF_S{year}01010000_E{year}12312300_CFR_TIM_01h_NA-_noc_org_20_SP245_ReGrB_PhM04_PECD4.2_fv1_NUTS0_level.csv"
                file_wind_on = f"P_CMI6_{model_fname}_WON_NA---_Pecd_PEON_S{year}01010000_E{year}12312300_CFR_TIM_01h_NA-_noc_org_30_SP245_ReGrB_PhM04_PECD4.2_fv1_NUTS0_level.csv"
            file_solar = f"P_CMI6_{model_fname}_SPV_0000m_Pecd_NUT0_S{year}01010000_E{year}12312300_CFR_TIM_01h_NA-_noc_org_61_SP245_NA---_PhM03_PECD4.2_fv1.csv"
            file_temp = f"P_CMI6_{model_fname}_TA-_0002m_Pecd_NUT0_S{year}01010000_E{year}12312300_INS_TIM_01h_NA-_cdf_org_NA_SP245_NA---_NA---_PECD4.2_fv1.csv"
            # file_hydro = f"P_CMI6_{model_fname}_HRO_0000m_Pecd_SZON_S203712280000_E203812200000_NRG_TIM_07d_NA-_noc_org_NA_SP245_NA---_StRnF_PECD4.2_fv1.csv"

            df_wind_off = pd.read_csv(
                os.path.join(INPUT_DIR_WIND, file_wind_off),
                # skiprows=52,
            )
            df_wind_on = pd.read_csv(
                os.path.join(INPUT_DIR_WIND, file_wind_on),
                # skiprows=52,
            )

            df_solar = pd.read_csv(
                os.path.join(INPUT_DIR_SOLAR, file_solar),
                skiprows=52,
            )

            df_temp = pd.read_csv(
                os.path.join(INPUT_DIR_TEMP, file_temp),
                skiprows=52,
            )

            df_year = pd.DataFrame(
                {
                    "Date": df_solar["Date"],
                    **{
                        f"{country}_OFF": df_wind_off.get(
                            f"{country}_OFF", pd.Series(np.nan, index=df_wind_off.index)
                        )
                        for country in COUNTRIES
                    },
                    **{
                        f"{country}_ON": df_wind_on.get(
                            f"{country}", pd.Series(np.nan, index=df_wind_on.index)
                        )
                        for country in COUNTRIES
                    },
                    **{
                        f"{country}_SOLAR": df_solar.get(
                            f"{country}", pd.Series(np.nan, index=df_solar.index)
                        )
                        for country in COUNTRIES
                    },
                    **{
                        f"{country}_TEMP": df_temp.get(
                            f"{country}", pd.Series(np.nan, index=df_temp.index)
                        )
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

    df_hourly["Hydrological_year"] = (
        df_hourly["Date"] - pd.DateOffset(months=3)
    ).dt.year  # start hydrological year 1st of April
    df_hourly["Calendar_year"] = df_hourly["Date"].dt.year

    df_daily["Hydrological_year"] = (
        df_daily["Date"] - pd.DateOffset(months=3)
    ).dt.year  # start hydrological year 1st of April
    df_daily["Calendar_year"] = df_daily["Date"].dt.year

    df_daily = df_daily.set_index(["model", "Date"])
    df_hourly = df_hourly.set_index(["model", "Date"])
    return df_hourly, df_daily


# functions for hydro processing
HOURS_PER_WEEK = 7 * 24
DAYS_PER_WEEK = 7


def convert_to_hourly(df_hydro):
    """
    Convert weekly MWh data to hourly GWh using forward fill,
    simply averaging weekly values over all the hours in that week."""
    # Convert weekly MWh → hourly GWh
    df_per_hour = df_hydro / (HOURS_PER_WEEK * 1000)

    # Extend by 7 days so the final weekly value can fill into its full week
    start = df_per_hour.index.min()
    end = df_per_hour.index.max() + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
    hourly_index = pd.date_range(start, end, freq="h")

    df_hydro_hourly = df_per_hour.reindex(hourly_index).ffill(limit=HOURS_PER_WEEK - 1)

    df_hydro_hourly.index.name = "Date"
    return df_hydro_hourly


def sum_by_country(df_hydro: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hydro generation by country using the first two characters
    of each column name.

    Countries not present in the input data are returned as NaN.
    """

    country_map = {}
    for col in df_hydro.columns:
        cc = col[:2]
        if cc == "GR":
            country_map[col] = "EL"
        elif cc in COUNTRIES:
            country_map[col] = cc

    df_country = df_hydro.rename(columns=country_map).T.groupby(level=0).sum().T

    return df_country.reindex(columns=COUNTRIES, fill_value=np.nan)


def process_hydro_file(df_hydro: pd.DataFrame) -> pd.DataFrame:
    """Process hydro data to get hourly values, aggregated per country."""
    df_hydro_country = sum_by_country(df_hydro)
    df_hydro_country_per_hour = convert_to_hourly(df_hydro_country)
    return df_hydro_country_per_hour


def process_all_hydro_files():
    """Build hydro generation dataframe for COUNTRIES based on all raw CSV files downloaded from PECD."""

    for model_fname, _ in zip(MODELS_FNAME, MODELS):
        for hydro_var in HYDRO_VARS:
            filename_hydro = f"P_CMI6_{model_fname}_{hydro_var}*"
            fname_out = (
                f"P_CMI6_{model_fname}_{hydro_var}_{YEARS[0]}_{YEARS[-1]}_NUTS0.csv"
            )
            files_hydro = glob.glob(os.path.join(INPUT_DIR_HYDRO_RAW, filename_hydro))
            df_hydro_raw = pd.concat(
                (pd.read_csv(f, comment="#") for f in files_hydro), ignore_index=True
            )
            df_hydro_raw["Date"] = pd.to_datetime(df_hydro_raw["Date"])
            df_hydro_raw = df_hydro_raw.set_index("Date").sort_index()
            df_hydro_var = process_hydro_file(df_hydro_raw).add_suffix(
                f"_{hydro_var}_prod"
            )
            df_hydro_var.to_csv(os.path.join(INPUT_DIR_HYDRO, fname_out))


# %%
