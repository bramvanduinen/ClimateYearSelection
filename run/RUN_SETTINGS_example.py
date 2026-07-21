# Run settings for the weather year selection pipeline.
# Make all configuration changes here.
import os

from cluster_weather_years.constants import BASE_INPUT
from cluster_weather_years.sa import wasserstein_score, wasserstein_seasonal_score

# =============================================================================
# GENERAL RUN SETTINGS
# =============================================================================
RUNNAME = "RUNNAME"  # name for this run, used in output folder
HYDRO = True  # whether to include hydro variables in matching
SEASONAL = True  # True: seasonal scoring/matching  |  False: annual
NUM_TO_SELECT = 5  # number of years to select
YEAR_COL = "Hydrological_year"  # "Hydrological_year" or "Calendar_year"

# =============================================================================
# COUNTRIES TO MATCH
# =============================================================================
# fmt: off
MATCH_COUNTRIES = ["NL"]

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
RUNNAME_SA_FALLBACK = "FALLBACK_RUNNAME"

# Total number of SA/Random Search/FRS experiments (each gets a unique seed).
# Cooling rates are distributed as evenly as possible across experiments:
# e.g. 45 experiments with 3 cooling rates → 15 experiments per rate.
NUM_EXPERIMENTS = 25

# Parameter grid — cooling_rate is distributed fairly across NUM_EXPERIMENTS seeds.
# All other grid values must be single-element lists (only first value is used).

if SEASONAL:
    score_metric = wasserstein_seasonal_score
else:
    score_metric = wasserstein_score

PARAM_GRID = {
    "metric": [
        score_metric,
    ],
    "temp_start": [2.0],
    "cooling_rate": [
        0.9975,
    ],
    "max_iter": [5000],
}

# =============================================================================
# FILTERED RANDOM SEARCH (FRS) SETTINGS
# =============================================================================
RUN_FRS = True  # set False to skip FRS

P_THRES = 25  # percentile threshold for univariate pre-filters
# (guideline: ~25 for 1 country, ~50 for a few, ~75 for many, with 30 years)
TUNING_ITERATIONS = int(1e3)  # iterations used to calibrate thresholds
# FRS runs FRS_ITER_MULTIPLIER × (sum of SA n_iter) total draws.
# E.g. with NUM_EXPERIMENTS=45 and sa_iter=5000 → 45*5000*10 = 2,250,000 draws.
FRS_ITER_MULTIPLIER = 10

# =============================================================================
# EXTREMES COST FUNCTION SETTINGS (run/02_run_selections_extremes.py only)
# =============================================================================
# Alternative optimisation target focused on extremes in residual load.
# Series = {c}_Residual_load for each c in MATCH_COUNTRIES, plus an optional
# European aggregate (sum over those countries). Ignored by the standard pipeline.
EXTREMES_Q_TAIL_MIN = 0.90  # lower bound of the upper-tail quantile grid
EXTREMES_N_Q = 101  # number of points in the quantile grids
EXTREMES_EVENT_THRESHOLD_QUANTILE = 0.95  # quantile defining the event threshold u_c
EXTREMES_BETA = 1.0  # weight of the event component vs the tail component
EXTREMES_GAMMA = 1.0  # weight of the event-frequency penalty within the event term
EXTREMES_INCLUDE_EUROPEAN_AGGREGATE = True  # add aggregate series (if >1 country)
EXTREMES_M_NORM = 1000  # random draws used for the normalisation constants
EXTREMES_NORM_SEED = 0  # seed for the normalisation draws

# =============================================================================
# DATA SETTINGS (usually don't need to change these)
# =============================================================================

# Raw input files
DF_DAILY_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_daily_all_countries.csv"
)
DF_HOURLY_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_hourly_all_countries.csv"
)
DF_HYDRO_FNAME = os.path.join(
    BASE_INPUT, "preprocessed_dataframes/df_hydro_prod_all_countries.parquet"
)

# Combined production + demand dataframe (built in 01_build_dataset.ipynb)
DF_PROD_DEMAND_FNAME = os.path.join(
    BASE_INPUT,
    "preprocessed_dataframes/df_prod_demand_daily.parquet",
)
DF_PROD_DEMAND_FNAME_REGION = os.path.join(
    BASE_INPUT,
    "preprocessed_dataframes/df_prod_demand_daily_regional.parquet",
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
