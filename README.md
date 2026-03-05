# Climate Year Selection

> **Note:** This project is a work in progress.

## Description

Method to select a small subset of representative climate years from a large multi-model climate simulation ensemble, for use in energy system impact studies. It matches distributional properties of key energy variables (wind, solar, hydro, demand and residual load) across one or more countries using several selection algorithms: Simulated Annealing (SA), K-Medoids, Hierarchical Brute-Force (HBF), and random draws.

## Contact

Bram van Duinen — bram.van.duinen@knmi.nl

## Installation

The package requires Python >= 3.10. It is recommended to use a virtual environment.

```bash
python -m venv climate_year_selection
source climate_year_selection/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -r requirements.txt` installs all required dependencies with pinned versions.

`pip install -e .` installs the `cluster_weather_years` library from `src/` in editable mode, meaning it links directly to the source files rather than copying them, so any local changes to the library are picked up immediately without reinstalling.

> **Note:** `cluster_weather_years` is not an appropriate name anymore. Will rename to `climate_year_selection` in the future.


## Configuration

Before running the pipeline, two configuration files need to be set up:

**1. `src/cluster_weather_years/constants.py`**

Set the base input and output paths:

```python
BASE_INPUT  = "/path/to/inputs/"
BASE_OUTPUT = "/path/to/outputs/"
```

Also verify that `HYDROLOGICAL_YEARS` and `MODELS` match your dataset.

**2. `run/RUN_SETTINGS.py`**

All run-specific settings live here. The main ones to configure:

| Setting | Description |
|---|---|
| `RUNNAME` | Name for the output directory of this run |
| `MATCH_COUNTRIES` | List of country codes for which you want to <br> find representative years (e.g. `["NL", "DE"]`) |
| `NUM_TO_SELECT` | Number of representative years to select |
| `HYDRO` | Whether to include hydro variables in matching |
| `SEASONAL` | Use seasonal scoring instead of annual |
| `RUN_SA` / `RUN_HBF` | Enable/disable individual algorithms |
| `NUM_EXPERIMENTS` | Number of independent experiments per algorithm |
| `PARAM_GRID` | SA hyperparameters (cooling rate, etc.) |

## Workflow

All scripts are in the `run/` directory. Steps 00a/00b only need to be run once to obtain the raw data. Step 01 only has to be run once to build the reference dataset.

### 00a. Download PECD weather/energy data *(optional)*
`run/00a_download_pecd_energy.py`

Download raw PECD climate and energy data via cdsapi, if not already available locally.

### 00b. Aggregate wind to NUTS0 *(optional)*
`run/00b_aggregate_wind_to_nuts0.ipynb`

Aggregate raw wind capacity factors from PEOF/PEON zones to country (NUTS0) level. Notebook adapted from C3S.

### 01. Build reference dataset
`run/01_build_dataset.ipynb`

Combines wind, solar, demand and hydro into a single daily
production-demand dataframe (`df_prod_demand_daily.parquet`) used by all
downstream steps. Here, enter any changes in installed capacity or demand-temperature assumptions you want to make.

### 02. Run selection algorithms
`run/02_run_selections.py`

Runs all configured algorithms (SA, K-Medoids, HBF, random draws) and produces automatic evaluation plots. Results are written to `BASE_OUTPUT/<RUNNAME>_<timestamp>/`.

```bash
cd run
python 02_run_selections.py
```

### 03. Evaluation plots
`run/03_evaluation_plots.py`

Generates additional evaluation figures from a completed run.

## Project structure

```
src/cluster_weather_years/   # core library
run/                         # pipeline scripts and notebooks
    RUN_SETTINGS.py          # all run configuration
    01_build_dataset.ipynb
    02_run_selections.py
    03_evaluation_plots.py
    ...
```

