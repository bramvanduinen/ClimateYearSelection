# Constants used in the calculations. User settings (paths, countries, reference period,
# algorithm settings) live in run/RUN_SETTINGS.py instead.
import numpy as np

# =============================================================================
# PECD 4.2 DATASET DEFINITION
# Tied to the downloaded PECD files and their names (see 00a_download_pecd_energy.py)
# =============================================================================

# Calendar years that are downloaded. Hydro data only starts in July 2015, so 2015 is
# left out to avoid an incomplete hydrological year.
YEARS = range(2016, 2047)

MODELS = [
    "cmcc_cm2_sr5",
    "ec_earth3",
    "mpi_esm1_2_hr",
    "awi_cm_1_1_mr",
    "bcc_csm2_mr",
    "mri_esm2_0",
]

# Model names as they appear in the PECD file names (same order as MODELS)
MODELS_FNAME = [
    "CMCC_CMR5",
    "ECEC_ECE3",
    "MPI-_MEHR",
    "AWI-_AWCM",
    "BCC-_BCCS",
    "MRI-_MRM2",
]

# fmt: off
COUNTRIES = [
    "AL", "AT", "BE", "BA", "BG", "HR", "CZ", "DK", "EE",
    "EL", "FI", "FR", "DE", "HU", "IE", "IT", "LV", "LT",
    "LU", "MT", "ME", "MK", "NL", "NO", "PL", "PT", "RO",
    "RS", "SK", "SI", "ES", "SE", "CH", "UK", "TR",
] # full list of ENTSO-E countries (EL = Greece in Eurostat notation)
# fmt: on

# Hydro technologies: run-of-river, reservoir, run-of-river with pondage
HYDRO_VARS = ["HRO", "HRG", "HPO"]

# =============================================================================
# METHOD AND EVALUATION CONVENTIONS
# =============================================================================

# Seasons used for the seasonal score (SSWD): extended winter/summer
SEASONS = {"Winter": [10, 11, 12, 1, 2, 3], "Summer": [4, 5, 6, 7, 8, 9]}

# Quantiles for QQ-comparison, denser in the tails
dense_low = np.linspace(0.005, 0.05, 9, endpoint=False)
coarse = np.linspace(0.05, 0.95, 91)
dense_high = np.linspace(0.955, 0.995, 9)
PROBS = np.concatenate([dense_low, coarse, dense_high])

# Renewable drought settings (only used for physical evaluation)
DROUGHT_PERCENTILE = 5  # Below P5 = low renewable day
DROUGHT_MIN_DURATION = 3  # Minimum consecutive days to count as drought
