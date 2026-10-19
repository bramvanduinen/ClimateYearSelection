# %%
import shutil
from pathlib import Path

import cdsapi

from climate_year_selection.constants import BASE_INPUT, MODELS, YEARS

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

output_dir_wind = Path(f"{BASE_INPUT}/pecd_data/energy/wind_peof_peon")
output_dir_wind.mkdir(parents=True, exist_ok=True)
output_dir_solar = Path(f"{BASE_INPUT}/pecd_data/energy/solar")
output_dir_solar.mkdir(parents=True, exist_ok=True)
output_dir_temp = Path(f"{BASE_INPUT}/pecd_data/energy/temp")
output_dir_temp.mkdir(parents=True, exist_ok=True)
output_dir_hydro = Path(f"{BASE_INPUT}/pecd_data/energy/hydro_weekly")
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
