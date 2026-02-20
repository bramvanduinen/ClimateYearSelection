# Run settings for the weather year selection pipeline.
# Make all configuration changes here.
import os

from cluster_weather_years.constants import BASE_INPUT
from cluster_weather_years.sa import wasserstein_score, wasserstein_seasonal_score

# =============================================================================
# GENERAL RUN SETTINGS
# =============================================================================
RUNNAME = "run_NL_5y_fair"
HYDRO = False  # whether to include hydro variables in matching
SEASONAL = True  # True: seasonal scoring/matching  |  False: annual
NUM_TO_SELECT = 5  # number of years to select
YEAR_COL = "Hydrological_year"  # "Hydrological_year" or "Calendar_year"

# =============================================================================
# COUNTRIES TO MATCH
# =============================================================================
# fmt: off
MATCH_COUNTRIES = ["NL"]

# MATCH_COUNTRIES = ["NL", "BE"]                       # quick testing
# MATCH_COUNTRIES = ["NL", "BE", "FR", "DK", "DE", "UK"]  # semi-quick testing
# MATCH_COUNTRIES = [
#     "AL", "AT", "BE", "BA", "BG", "HR", "CZ", "DK", "EE",
#     "EL", "FI", "FR", "DE", "HU", "IE", "IT", "LV", "LT",
#     "LU", "MT", "ME", "MK", "NL", "NO", "PL", "PT", "RO",
#     "RS", "SK", "SI", "ES", "SE", "CH", "UK", "TR",
# ]  # full ENTSO-E country list (EL = Greece in Eurostat notation)
# fmt: on

# =============================================================================
# SIMULATED ANNEALING (SA) SETTINGS
# =============================================================================
RUN_SA = True  # set False to skip SA and load results from RUNNAME_SA_FALLBACK instead
RUNNAME_SA_FALLBACK = "run_Europe_30y_20260127_1559"

SEEDS = range(15)

# Parameter grid — one SA run is launched per combination of (seed × param set)
# NOTE: metric should match SEASONAL setting above
PARAM_GRID = {
    "metric": [
        wasserstein_seasonal_score,
        # wasserstein_score,
    ],
    "temp_start": [2.0],
    "cooling_rate": [
        [
            0.99,
            0.995,
            0.9975,
        ],  # 0.9 for quick testing; use [0.99, 0.995, 0.9975] for full runs
    ],
    "max_iter": [5000],
}

# =============================================================================
# HIERARCHICAL BRUTE-FORCE (HBF) SETTINGS
# =============================================================================
RUN_HBF = True  # set False to skip HBF

P_THRES = 20  # percentile threshold for univariate pre-filters
# (guideline: ~25 for 1 country, ~50 for a few, ~75 for many, with 30 years)
TUNING_ITERATIONS = int(1e3)  # iterations used to calibrate thresholds
HBF_ITERATIONS = int(1e6)  # total random draws in HBF search
BATCH_SIZE = 1000  # draws per parallel batch

# =============================================================================
# DATA SETTINGS (usually don't need to change these)
# =============================================================================

# Raw input files
DF_DAILY_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_daily_all_countries_v3.csv"
)
DF_HOURLY_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_hourly_all_countries_v3.csv"
)
DF_HYDRO_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_hydro_prod_all_countries_v1.parquet"
)

# Combined production + demand dataframe (built in 01_build_dataset.ipynb)
DF_PROD_DEMAND_FNAME = os.path.join(
    BASE_INPUT,
    "preprocessed_dataframes/df_prod_demand_daily_all_countries_with_hydro_scaled_demand.parquet",
)
DF_PROD_DEMAND_FNAME_REGION = os.path.join(
    BASE_INPUT,
    "preprocessed_dataframes/df_prod_demand_daily_all_regions_with_hydro_scaled_demand.parquet",
)

# Variables to match — derived from MATCH_COUNTRIES and HYDRO flag
_base_vars = (
    [f"{c}_OFF_prod" for c in MATCH_COUNTRIES]
    + [f"{c}_ON_prod" for c in MATCH_COUNTRIES]
    + [f"{c}_SOLAR_prod" for c in MATCH_COUNTRIES]
    + [f"{c}_Demand" for c in MATCH_COUNTRIES]
    + [f"{c}_Residual_load" for c in MATCH_COUNTRIES]
)
_hydro_vars = (
    [f"{c}_HRO_prod" for c in MATCH_COUNTRIES]  # run-of-river
    + [f"{c}_HRG_prod" for c in MATCH_COUNTRIES]  # reservoir
    + [f"{c}_HPO_prod" for c in MATCH_COUNTRIES]  # ROR + pondage
)
VARS_TO_MATCH = _base_vars + (_hydro_vars if HYDRO else [])
