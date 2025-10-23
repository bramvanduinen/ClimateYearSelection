# list of constants
import os

from matplotlib import cm

YEARS = range(2015, 2045)
HYDROLOGICAL_YEARS = range(2015, 2044)

YEAR_COL = "Hydrological_year"

THRES_HDD = 12.3  # values for Austria, taken from Katharina Gruber
THRES_CDD = 25.5  # values for Austria, taken from Katharina Gruber

WINDOW_ROLLING = 30  # days rolling mean for climatology
MIN_WINDOW_ROLLING = 7  # minimum amount of days for rolling mean to be non-nan.

COLORS = cm.get_cmap("tab10").colors

BASE = "/net/pc230050/nobackup/users/duinen/04_weather_year_selection/pecd_data"
BASE_OUTPUT = "/net/pc230050/nobackup/users/duinen/04_weather_year_selection/outputs/"

OUTPUT_DIR_WIND = os.path.join(BASE, "wind")
OUTPUT_DIR_SOLAR = os.path.join(BASE, "solar")
OUTPUT_DIR_TEMP = os.path.join(BASE, "temp")

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
COUNTRIES = ["NL", "DE", "BE", "FR"]  # for now


OUTPUT_DIRS = [OUTPUT_DIR_WIND, OUTPUT_DIR_SOLAR, OUTPUT_DIR_TEMP]

# Derived paths for cached dataframes and outputs used by SA notebook

SA_OUTPUT_DIR = os.path.join(BASE_OUTPUT, "simulated_annealing")
