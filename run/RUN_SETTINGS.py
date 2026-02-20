# overview of settings for SA runs. Make changes here.
import os

from cluster_weather_years.constants import BASE_INPUT
from cluster_weather_years.sa import wasserstein_score, wasserstein_seasonal_score

### INPUT/OUTPUT SETTINGS ###
RUNNAME = "testrun_NL_5y"
HYDRO = False  # whether to include hydro in the matching

RUN_SA = True  # whether to run simulated annealing
RUNNAME_SA_FALLBACK = "run_Europe_30y_20260127_1559"  # fallback run to copy SA results from if RUN_SA is False
RUN_HBF = True  # whether to run hierarchical brute-force

# NOTE: For fair comparison, seasonal should match wasserstein_seasonal or wasserstein regular
SEASONAL = True  # whether to use seasonal scoring (True) or regular scoring (False) HBF

### SELECTION SETTINGS ###
# fmt: off
# list of countries to consider for matching, can be all countries or a subset
MATCH_COUNTRIES = ["NL"]
# MATCH_COUNTRIES = ["NL", "BE"]  # for quick testing
#MATCH_COUNTRIES = ["NL", "BE", "FR", "DK", "DE", "UK"] # for semi-quick testing
# MATCH_COUNTRIES = [
#     "AL", "AT", "BE", "BA", "BG", "HR", "CZ", "DK", "EE",
#     "EL", "FI", "FR", "DE", "HU", "IE", "IT", "LV", "LT",
#     "LU", "MT", "ME", "MK", "NL", "NO", "PL", "PT", "RO",
#     "RS", "SK", "SI", "ES", "SE", "CH", "UK", "TR",
# ] # full list of ENTSO-E countries (note: EL is Greece in Eurostat)
# fmt: on

NUM_TO_SELECT = 5  # number of years to select
YEAR_COL = "Hydrological_year"  # Default year column to operate on: "Hydrological_year" or "Calendar_year"

### SA SETTINGS ###

# parameter grid for different SA runs
PARAM_GRID = {
    "metric": [
        wasserstein_seasonal_score
        # wasserstein_score
    ],  # [ad_seasonal, wasserstein_score, wasserstein_seasonal_score, combined_score, ad_total],
    "temp_start": [2.0],
    # "cooling_rate": [0.99, 0.995, 0.9975],
    "cooling_rate": [
        0.9,
    ],  # for quick testing
    "max_iter": [5000],
}
SEEDS = range(3)

### Hierarchical Brute-Force (HBF) Settings ###
P_THRES = 20  # percentile threshold for univariates (25 for one country, 50 for a few countries, 75 for many countries, for 30 years!)
TUNING_ITERATIONS = int(1e2)
HBF_ITERATIONS = int(1e4)
BATCH_SIZE = 100

### DATA SETTINGS ###

# filenames to be used as input

# capacity factors and hydro production (from PECD, downloaded in 00_download_pecd_data.ipynb)
DF_DAILY_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_daily_all_countries_v3.csv"
)
DF_HOURLY_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_hourly_all_countries_v3.csv"
)
DF_HYDRO_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_hydro_prod_all_countries_v1.parquet"
)

# combined production and demand dataframe (produced in 01_build_dataset.ipynb)
DF_PROD_DEMAND_FNAME = os.path.join(
    BASE_INPUT,
    "preprocessed_dataframes/df_prod_demand_daily_all_countries_with_hydro_scaled_demand.parquet",
)
DF_PROD_DEMAND_FNAME_REGION = os.path.join(
    BASE_INPUT,
    "preprocessed_dataframes/df_prod_demand_daily_all_regions_with_hydro_scaled_demand.parquet",
)

if HYDRO:
    VARS_TO_MATCH = (
        [f"{country}_OFF_prod" for country in MATCH_COUNTRIES]
        + [f"{country}_ON_prod" for country in MATCH_COUNTRIES]
        + [f"{country}_SOLAR_prod" for country in MATCH_COUNTRIES]
        + [f"{country}_Demand" for country in MATCH_COUNTRIES]
        + [f"{country}_Residual_load" for country in MATCH_COUNTRIES]
        + [f"{country}_HRO_prod" for country in MATCH_COUNTRIES]  # hydro run-of-river
        + [f"{country}_HRG_prod" for country in MATCH_COUNTRIES]  # hydro reservoir
        + [f"{country}_HPO_prod" for country in MATCH_COUNTRIES]  # hydro ROR + pondage
    )
else:
    VARS_TO_MATCH = (
        [f"{country}_OFF_prod" for country in MATCH_COUNTRIES]
        + [f"{country}_ON_prod" for country in MATCH_COUNTRIES]
        + [f"{country}_SOLAR_prod" for country in MATCH_COUNTRIES]
        + [f"{country}_Demand" for country in MATCH_COUNTRIES]
        + [f"{country}_Residual_load" for country in MATCH_COUNTRIES]
    )
