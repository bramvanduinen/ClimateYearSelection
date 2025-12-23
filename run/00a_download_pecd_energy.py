# %%
import shutil
from pathlib import Path

import cdsapi

from cluster_weather_years.constants import BASE_INPUT

# %%
dataset = "sis-energy-pecd"
models = [
    "cmcc_cm2_sr5",
    "ec_earth3",
    "mpi_esm1_2_hr",
    "awi_cm_1_1_mr",
    "bcc_csm2_mr",
    "mri_esm2_0",
]

request_wind = {
    "pecd_version": "pecd4_2",
    "temporal_period": ["future_projections"],
    "emission_scenario": ["ssp2_4_5"],
    "energy_scenario": ["resource_grade_b"],
    "variable": [
        "wind_power_offshore_capacity_factor",
        "wind_power_onshore_capacity_factor",
    ],
    "technology": ["20", "30"],
    "spatial_resolution": ["peof", "peon"],
}

request_temp = {
    "pecd_version": "pecd4_2",
    "temporal_period": ["future_projections"],
    "emission_scenario": ["ssp2_4_5"],
    "variable": ["2m_temperature", "nuts_0_region_mask"],
    "spatial_resolution": ["nuts_0"],
}

request_solar = {
    "pecd_version": "pecd4_2",
    "temporal_period": ["future_projections"],
    "emission_scenario": ["ssp2_4_5"],
    # "year": ["2015", "2016"],
    "variable": ["solar_photovoltaic_generation_capacity_factor", "nuts_0_region_mask"],
    "technology": [
        "61",  # residential
    ],
    "spatial_resolution": ["nuts_0"],
}

request_hydro = {
    "pecd_version": "pecd4_2",
    "temporal_period": ["future_projections"],
    "emission_scenario": ["ssp2_4_5"],
    "variable": [
        "hydropower_reservoir_generation",
        "hydropower_run_of_river_generation",
        "hydropower_run_of_river_with_pondage_generation",
    ],
    "spatial_resolution": ["szon"],
}

# %%
output_dir_wind = Path(f"{BASE_INPUT}/energy/wind_peof_peon")
output_dir_wind.mkdir(exist_ok=True)
output_dir_solar = Path(f"{BASE_INPUT}/energy/solar")
output_dir_solar.mkdir(exist_ok=True)
output_dir_temp = Path(f"{BASE_INPUT}/energy/temp")
output_dir_temp.mkdir(exist_ok=True)
output_dir_hydro = Path(f"{BASE_INPUT}/energy/hydro")
output_dir_hydro.mkdir(exist_ok=True)

client = cdsapi.Client()

# requests = [request_wind, request_solar, request_temp]
# output_dirs = [output_dir_wind, output_dir_solar, output_dir_temp]
requests = [request_hydro]  # only hydro for now
output_dirs = [output_dir_hydro]

for model in models:
    print("Retrieving model:", model)
    yearrange = range(2015, 2045)
    for year in map(str, yearrange):
        print("Retrieving year:", year)
        for request, output_dir in zip(requests, output_dirs):
            if (
                model == "bcc_csm2_mr" and request == request_wind
            ):  # use fv2 for this model and only for wind
                request_y = request | {
                    "year": [year],
                    "origin": [model],
                    "file_version": "fv2",
                }
            else:
                request_y = request | {
                    "year": [year],
                    "origin": [model],
                    "file_version": "fv1",
                }

            output_file = output_dir / f"pecd_{model}_{year}.zip"
            # client.retrieve(dataset, request_y, target_file.as_posix())
            client.retrieve(dataset, request_y, output_file)
            # unzip
            shutil.unpack_archive(output_file, output_dir)
