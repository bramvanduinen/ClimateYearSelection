# overview of settings for SA runs. Make changes here.
import os

from cluster_weather_years.constants import BASE_OUTPUT
from cluster_weather_years.sa import wasserstein_seasonal_score

### INPUT/OUTPUT SETTINGS ###
RUNNAME = "testrun_refactor"
# filenames to be used as input
# TODO: Add hydro

# capacity factors and hydro production (from PECD, downloaded in 00_download_pecd_data.ipynb)
DF_DAILY_FNAME = os.path.join(BASE_OUTPUT, "dataframes/df_daily_all_countries_v3.csv")
DF_HOURLY_FNAME = os.path.join(BASE_OUTPUT, "dataframes/df_hourly_all_countries_v3.csv")
DF_HYDRO_FNAME = os.path.join(
    BASE_OUTPUT, "dataframes/df_hydro_prod_all_countries_v1.parquet"
)

# combined production and demand dataframe (produced in 01_build_dataset.ipynb)
DF_PROD_DEMAND_FNAME = os.path.join(
    BASE_OUTPUT,
    "dataframes/df_prod_demand_daily_all_countries_with_hydro_scaled_demand.parquet",
)
DF_PROD_DEMAND_FNAME_REGION = os.path.join(
    BASE_OUTPUT,
    "dataframes/df_prod_demand_daily_all_regions_with_hydro_scaled_demand.parquet",
)

### SELECTION SETTINGS ###
# fmt: off
# list of countries to consider for matching, can be all countries or a subset
MATCH_COUNTRIES = ["NL", "BE"] # for quick testing
# MATCH_COUNTRIES = [
#     "AL", "AT", "BE", "BA", "BG", "HR", "CZ", "DK", "EE",
#     "EL", "FI", "FR", "DE", "HU", "IE", "IT", "LV", "LT",
#     "LU", "MT", "ME", "MK", "NL", "NO", "PL", "PT", "RO",
#     "RS", "SK", "SI", "ES", "SE", "CH", "UK", "TR",
# ] # full list of ENTSO-E countries (note: EL is Greece in Eurostat)
# fmt: on

# TODO: Add hydro
VARS_TO_MATCH = (
    [f"{country}_OFF_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_ON_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_SOLAR_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_Demand" for country in MATCH_COUNTRIES]
    + [f"{country}_Residual_load" for country in MATCH_COUNTRIES]
)

NUM_TO_SELECT = 5  # number of years to select
YEAR_COL = "Hydrological_year"  # Default year column to operate on: "Hydrological_year" or "Calendar_year"

### SA SETTINGS ###
# parameter grid for different SA runs
PARAM_GRID = {
    "metric": [
        wasserstein_seasonal_score
    ],  # [ad_seasonal, wasserstein_score, wasserstein_seasonal_score, combined_score, ad_total],
    "temp_start": [2.0],
    "cooling_rate": [0.99, 0.995, 0.9975],
    # "cooling_rate": [0.9, 0.95, 0.97],  # for quick testing
    "max_iter": [5_000],
}
SEEDS = range(15)
