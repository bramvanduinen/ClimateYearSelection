# list of constants
import os

import numpy as np
from matplotlib import cm

# YEARS = range(2015, 2045)
YEARS = range(
    2016, 2047
)  # updated because hydro data only starts in July 2015, so incomplete hydro year.
HYDROLOGICAL_YEARS = range(2016, 2046)

# quantiles for QQ-comparison
dense_low = np.linspace(0.005, 0.05, 9, endpoint=False)
coarse = np.linspace(0.05, 0.95, 91)
dense_high = np.linspace(0.955, 0.995, 9)
PROBS = np.concatenate([dense_low, coarse, dense_high])

# extended winter/summer
SEASONS = {"Winter": [10, 11, 12, 1, 2, 3], "Summer": [4, 5, 6, 7, 8, 9]}

COLORS = cm.get_cmap("tab10").colors

# TODO: move these to RUN_SETTINGS.py. Doesn't make sense to set paths in method.
# Maybe the whole constants.py should be moved there actually, since it's mostly run settings and paths.
BASE_INPUT = "/net/pc230050/nobackup/users/duinen/04_weather_year_selection/inputs/"
BASE_OUTPUT = "/net/pc230050/nobackup/users/duinen/04_weather_year_selection/outputs/"

INPUT_DIR_WIND = os.path.join(BASE_INPUT, "pecd_data/energy/wind")
INPUT_DIR_SOLAR = os.path.join(BASE_INPUT, "pecd_data/energy/solar")
INPUT_DIR_TEMP = os.path.join(BASE_INPUT, "pecd_data/energy/temp")
INPUT_DIR_HYDRO_RAW = os.path.join(BASE_INPUT, "pecd_data/energy/hydro_weekly")
INPUT_DIR_HYDRO = os.path.join(BASE_INPUT, "pecd_data/energy/hydro")

MODELS = [
    "cmcc_cm2_sr5",
    "ec_earth3",
    "mpi_esm1_2_hr",
    "awi_cm_1_1_mr",
    "bcc_csm2_mr",
    "mri_esm2_0",
]

MODELS_FNAME = [
    "CMCC_CMR5",
    "ECEC_ECE3",
    "MPI-_MEHR",
    "AWI-_AWCM",
    "BCC-_BCCS",
    "MRI-_MRM2",
]
# COUNTRIES = ["NL", "DE", "BE", "FR"]  # for now
# fmt: off
COUNTRIES = [
    "AL", "AT", "BE", "BA", "BG", "HR", "CZ", "DK", "EE",
    "EL", "FI", "FR", "DE", "HU", "IE", "IT", "LV", "LT",
    "LU", "MT", "ME", "MK", "NL", "NO", "PL", "PT", "RO",
    "RS", "SK", "SI", "ES", "SE", "CH", "UK", "TR",
] # full list of ENTSO-E countries
# fmt: on

HYDRO_VARS = ["HRO", "HRG", "HPO"]


# Renewable drought settings (only used for physical evaluation)
DROUGHT_PERCENTILE = 5  # Below P5 = low renewable day
DROUGHT_MIN_DURATION = 3  # Minimum consecutive days to count as drought
