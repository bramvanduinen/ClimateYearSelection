# overview of settings for SA runs. Make changes here.
import os

from cluster_weather_years.constants import BASE_OUTPUT
from cluster_weather_years.sa import wasserstein_seasonal_score

# filenames to be used as input
# updated version, all countries with scaled demand. Note: demand scaling is WRONG here!
# TODO: Add hydro
DF_DAILY_FNAME = os.path.join(BASE_OUTPUT, "dataframes/df_daily_all_countries_v1.csv")
DF_PROD_DEMAND_FNAME = os.path.join(
    BASE_OUTPUT, "dataframes/df_prod_demand_daily_all_countries_WRONG_DEMAND.csv"
)

# previous version (NL, FR, BE, DE only all with NL capacities)
# DF_DAILY_FNAME = os.path.join(BASE_OUTPUT, "dataframes/df_daily_multi_country_v1.csv")
# DF_PROD_DEMAND_FNAME = os.path.join(
#     BASE_OUTPUT, "dataframes/df_prod_demand_daily_multi_country_WRONG_CAPACITIES.csv"
# )

# Variable matching mode: "full" or "residual" or "full_multi_country"
MATCH_VARS = "full_multi_country"

# MATCH_COUNTRIES = ["NL", "BE", "FR", "DE"]
# MATCH_COUNTRIES = ["NL"]
# fmt: off
MATCH_COUNTRIES = [
    "AL", "AT", "BE", "BA", "BG", "HR", "CZ", "DK", "EE",
    "EL", "FI", "FR", "DE", "HU", "IE", "IT", "LV", "LT",
    "LU", "MT", "ME", "MK", "NL", "NO", "PL", "PT", "RO",
    "RS", "SK", "SI", "ES", "SE", "CH", "UK", "TR",
] # full list of ENTSO-E countries (note: EL is Greece in Eurostat)
# fmt: on
RUNNAME = "all_countries_full_vars"

NUM_CLUSTERS = 5  # number of clusters to be used by K-Medoids/K-Means

# Default year column to operate on: "Hydrological_year" or "Calendar_year"
YEAR_COL = "Hydrological_year"

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
