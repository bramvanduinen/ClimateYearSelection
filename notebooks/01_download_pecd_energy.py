# %%
import shutil
from pathlib import Path

import cdsapi

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
    # "origin": ["ec_earth3"],
    "emission_scenario": ["ssp2_4_5"],
    "energy_scenario": ["resource_grade_b"],
    "variable": [
        "wind_power_offshore_capacity_factor",
        "wind_power_onshore_capacity_factor",
        "nuts_0_region_mask",
    ],
    "technology": ["20", "30"],
    "spatial_resolution": ["nuts_0"],
}

request_temp = {
    "pecd_version": "pecd4_2",
    "temporal_period": ["future_projections"],
    # "origin": ["ec_earth3"],
    "emission_scenario": ["ssp2_4_5"],
    # "year": ["2015", "2016", "2017"],
    "variable": ["2m_temperature", "nuts_0_region_mask"],
    "spatial_resolution": ["nuts_0"],
}

request_solar = {
    "pecd_version": "pecd4_2",
    "temporal_period": ["future_projections"],
    # "origin": ["ec_earth3"],
    "emission_scenario": ["ssp2_4_5"],
    # "year": ["2015", "2016"],
    "variable": ["solar_photovoltaic_generation_capacity_factor", "nuts_0_region_mask"],
    "technology": [
        "61",  # residential
    ],
    "spatial_resolution": ["nuts_0"],
}


# %%
output_dir_wind = Path("pecd_data/wind")
output_dir_wind.mkdir(exist_ok=True)
output_dir_solar = Path("pecd_data/solar")
output_dir_solar.mkdir(exist_ok=True)
output_dir_temp = Path("pecd_data/temp")
output_dir_temp.mkdir(exist_ok=True)

client = cdsapi.Client()

requests = [request_wind, request_solar, request_temp]
output_dirs = [output_dir_wind, output_dir_solar, output_dir_temp]

for model in models:
    print("Retrieving model:", model)
    if model == "ec_earth3":
        yearrange = range(2025, 2045)  # we already have 2015-2024 for ec-earth3.
    else:
        yearrange = range(2015, 2045)
    for year in map(str, yearrange):
        print("Retrieving year:", year)
        for request, output_dir in zip(requests, output_dirs):
            request_y = request_y = request | {"year": [year], "origin": [model]}

            output_file = output_dir / f"pecd_{model}_{year}.zip"
            # client.retrieve(dataset, request_y, target_file.as_posix())
            client.retrieve(dataset, request_y, output_file)
            # unzip
            shutil.unpack_archive(output_file, output_dir)
