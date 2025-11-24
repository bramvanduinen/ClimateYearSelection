# %%
# NOTE: NOT USED YET!!


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
    "variable": ["100m_wind_speed"],
    "spatial_resolution": ["nuts_0"],
    "file_version": ["fv2"],
}

# request_temp = {
#     "pecd_version": "pecd4_2",
#     "temporal_period": ["future_projections"],
#     # "origin": ["ec_earth3"],
#     "emission_scenario": ["ssp2_4_5"],
#     # "year": ["2015", "2016", "2017"],
#     "variable": ["2m_temperature", "nuts_0_region_mask"],
#     "spatial_resolution": ["nuts_0"],
#     "file_version": ["fv2"],
# }

request_solar = {
    "pecd_version": "pecd4_2",
    "temporal_period": ["future_projections"],
    # "origin": ["ec_earth3"],
    "emission_scenario": ["ssp2_4_5"],
    # "year": ["2015", "2016"],
    "variable": ["surface_solar_radiation_downwards"],
    "spatial_resolution": ["nuts_0"],
    "file_version": ["fv2"],
}


# %%
output_dir_wind = Path("pecd_data/wind")
output_dir_wind.mkdir(exist_ok=True)
output_dir_solar = Path("pecd_data/solar")
output_dir_solar.mkdir(exist_ok=True)
output_dir_temp = Path("pecd_data/temp")
output_dir_temp.mkdir(exist_ok=True)
mlz_models = ["cmcc_cm2_sr5", "mpi_esm1_2_hr", "ec_earth3"]

client = cdsapi.Client()

requests = [request_wind, request_solar]
output_dirs = [output_dir_wind, output_dir_solar]

for model in models:
    print("Retrieving model:", model)
    if model in mlz_models:
        yearrange = range(2015, 2025) + range(
            2037, 2045
        )  # we already have 2015-2024 for ec-earth3.
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
