"""
Download the PECD 4.2 climate and energy data from the Copernicus Climate Data Store.

Downloads wind and solar capacity factors, 2 m temperature and hydropower generation
for all MODELS and YEARS (see constants.py), SSP2-4.5 future projections. Files are
unzipped into the INPUT_DIR_* folders set in RUN_SETTINGS.py.

Requires a CDS account and API key, see https://cds.climate.copernicus.eu/how-to-api
Only needs to be run once; downloading all files can take a long time.
"""

# %%
import shutil
from pathlib import Path

import cdsapi

# pylint: disable=import-error
from RUN_SETTINGS import (
    INPUT_DIR_HYDRO_RAW,
    INPUT_DIR_SOLAR,
    INPUT_DIR_TEMP,
    INPUT_DIR_WIND_RAW,
)

from climate_year_selection.constants import MODELS, YEARS

# pylint: enable=import-error

# %%
# build requests and output dirs
dataset = "sis-energy-pecd"

request_wind = {
    "energy_scenario": ["resource_grade_b"],
    "variable": [
        "wind_power_offshore_capacity_factor",
        "wind_power_onshore_capacity_factor",
    ],
    "technology": ["20", "30"],  # current technologies
    "spatial_resolution": ["peof", "peon"],
}

request_temp = {
    "variable": ["2m_temperature", "nuts_0_region_mask"],
    "spatial_resolution": ["nuts_0"],
}

request_solar = {
    "variable": ["solar_photovoltaic_generation_capacity_factor", "nuts_0_region_mask"],
    "technology": [
        "61",  # residential
    ],
    "spatial_resolution": ["nuts_0"],
}

request_hydro = {
    "variable": [
        "hydropower_reservoir_generation",
        "hydropower_run_of_river_generation",
        "hydropower_run_of_river_with_pondage_generation",
    ],
    "spatial_resolution": ["szon"],
}

output_dir_wind = Path(INPUT_DIR_WIND_RAW)
output_dir_wind.mkdir(parents=True, exist_ok=True)
output_dir_solar = Path(INPUT_DIR_SOLAR)
output_dir_solar.mkdir(parents=True, exist_ok=True)
output_dir_temp = Path(INPUT_DIR_TEMP)
output_dir_temp.mkdir(parents=True, exist_ok=True)
output_dir_hydro = Path(INPUT_DIR_HYDRO_RAW)
output_dir_hydro.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

requests = [request_wind, request_solar, request_hydro, request_temp]
output_dirs = [output_dir_wind, output_dir_solar, output_dir_hydro, output_dir_temp]

# %%
# download data
for model in MODELS:
    print("Retrieving model:", model)
    for year in map(str, YEARS):
        print("Retrieving year:", year)
        for request, output_dir in zip(requests, output_dirs):
            if (
                model == "bcc_csm2_mr" and request == request_wind
            ):  # use fv2 for this model and only for wind
                request_y = request | {
                    "pecd_version": "pecd4_2",
                    "temporal_period": ["future_projections"],
                    "emission_scenario": ["ssp2_4_5"],
                    "year": [year],
                    "origin": [model],
                    "file_version": "fv2",
                }
            else:
                request_y = request | {
                    "pecd_version": "pecd4_2",
                    "temporal_period": ["future_projections"],
                    "emission_scenario": ["ssp2_4_5"],
                    "year": [year],
                    "origin": [model],
                    "file_version": "fv1",
                }

            output_file = output_dir / f"pecd_{model}_{year}.zip"
            client.retrieve(dataset, request_y, output_file)
            # unzip
            shutil.unpack_archive(output_file, output_dir)
