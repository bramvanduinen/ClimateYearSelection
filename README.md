# Climate Year Selection


> **Note:** This repository will be further cleaned up in the near future to ensure user friendliness.

## Description

Method to select a small subset of representative climate years from a large multi-model climate simulation ensemble, for use in energy system impact studies. It matches distributional properties of key energy-impact variables (wind, solar, hydro, demand and residual load) across one or more countries using several selection algorithms: Simulated Annealing (SA), K-Medoids, Filtered Random Search (FRS), and Random Search (RS).

Selections are scored with the **Sliced Seasonal Wasserstein Distance (SSWD)**: the multivariate Sliced Wasserstein Distance between the selected subset and the full reference ensemble is computed per season, and the worst-performing season sets the score. Lower is better. In annual mode (`SEASONAL = False`) the plain Sliced Wasserstein Distance (SWD) over the whole year is used instead.

## Reference

This code accompanies the paper:

> Bram van Duinen, Karin van der Wiel, Jean Thorey, Laurens Stoop.
> *Bridging the climate to energy data gap: simulated annealing for representative climate year selection.*
> In review at **Applied Energy**.
> Preprint: https://doi.org/10.48550/arXiv.2605.15958

If you use this code, please cite the paper above.

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

`pip install -e .` installs the `climate_year_selection` library from `src/` in editable mode, meaning it links directly to the source files rather than copying them, so any local changes to the library are picked up immediately without reinstalling.



## Configuration

Before running the pipeline, two configuration files need to be set up:

**1. `src/climate_year_selection/constants.py`**

Set the base input and output paths:

```python
BASE_INPUT  = "/path/to/inputs/"
BASE_OUTPUT = "/path/to/outputs/"
```

Also verify that `HYDROLOGICAL_YEARS` and `MODELS` match your dataset.

**2. `run/RUN_SETTINGS_example.py`**

To run, change name to RUN_SETTINGS.py. All run-specific settings live here. The main ones to configure:

| Setting | Description |
|---|---|
| `RUNNAME` | Name for the output directory of this run |
| `MATCH_COUNTRIES` | List of country codes for which you want to <br> find representative years (e.g. `["NL", "DE"]`) |
| `NUM_TO_SELECT` | Number of representative years to select |
| `HYDRO` | Whether to include hydro variables in matching |
| `SEASONAL` | Use seasonal scoring (SSWD) instead of annual (SWD) |
| `NUM_EXPERIMENTS` | Number of independent experiments per algorithm, <br> each with its own seed |
| `SEED_OFFSET` | Added to every experiment seed. Change it to get a <br> different stochastic outcome without re-tuning anything else |
| `PARAM_GRID` | SA hyperparameters: `temp_start`, `cooling_rate`, `max_iter`. <br> Cooling rates are spread evenly across `NUM_EXPERIMENTS`, <br> all other entries must be single-element lists |


Scoring cost/accuracy trade-off. The SSWD is estimated from random projections, so more projections means a more accurate but slower score:

| Setting | Description |
|---|---|
| `PERFORM_TWO_STEP` | If `True`, score all candidates cheaply with `N_PROJECTIONS_INIT`, <br> then re-score the top 10% with `N_PROJECTIONS_FULL` |
| `N_PROJECTIONS_INIT` | Projections used for the cheap screening pass (default 50) |
| `N_PROJECTIONS_FULL` | Projections used for the refinement pass (default 200) |

Filtered Random Search settings:

| Setting | Description |
|---|---|
| `P_THRES` | Percentile threshold for the univariate pre-filters <br> (guideline: ~25 for 1 country, ~50 for a few, ~85 for many, with 5 or 30 years) |
| `TUNING_ITERATIONS` | Iterations used to calibrate those thresholds |
| `FRS_ITER_MULTIPLIER` | FRS draws = `FRS_ITER_MULTIPLIER` × (total SA iterations) |

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

Runs all four selection algorithms and produces automatic evaluation plots. Results are written to `BASE_OUTPUT/<RUNNAME>_<timestamp>/`, with one subdirectory per method (`simulated_annealing/`, `random_search/`, `k_medoids/`, `filtered_random_search/`) plus `evaluation/` (and `evaluation/by_country/`). A copy of `RUN_SETTINGS.py` is saved alongside the results as `RUN_SETTINGS_of_run.py` for record keeping.

The methods run in this order, with Random Search and K-Medoids matched to SA's compute budget:

1. **Simulated Annealing (SA)** 
2. **Random Search (RS)**
3. **K-Medoids**
4. **Filtered Random Search (FRS)**

```bash
cd run
python 02_run_selections.py
```

### 03. Evaluation plots
`run/03_evaluation_plots.py`

Generates additional evaluation figures from a completed run.

## Project structure

```
src/climate_year_selection/       # core library
    constants.py                 # paths, models, hydrological years, seasons
    cost_functions.py            # (S)SWD scoring functions used by the search
    sa.py                        # simulated annealing
    k_medoids_stochastic.py      # k-medoids, stochastic variant
    k_medoids_deterministic.py   # k-medoids, deterministic variant (unused)
    filtered_random_search.py    # FRS: univariate pre-filters + random sampling
    evaluation.py                # scoring and seed-uncertainty analysis
    evaluation_plot_helpers.py   # figure generation
    helper_functions.py

run/                             # pipeline scripts and notebooks
    RUN_SETTINGS.py              # all run configuration (gitignored; copy from example)
    RUN_SETTINGS_example.py      # template — rename to RUN_SETTINGS.py
    00a_download_pecd_energy.py
    00b_aggregate_wind_to_nuts0.ipynb
    01_build_dataset.ipynb
    02_run_selections.py
    03_evaluation_plots.py
    illustrations/               # standalone scripts for paper figures
    weights/                     # PECD capacity weights for wind aggregation
```

`RUN_SETTINGS.py` is gitignored so that local run configuration never ends up in version control — edit `RUN_SETTINGS_example.py` only when the *set* of available settings changes.

