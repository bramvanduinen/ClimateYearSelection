# Climate Year Selection

## Description

Method to select a small subset of representative climate years from a large multi-model climate simulation ensemble, for use in energy system impact studies. It matches distributional properties of key energy-impact variables (wind, solar, hydro, demand and residual load) across one or more countries. The paper compares four selection algorithms: Simulated Annealing (SA), K-Medoids, Filtered Random Search (FRS), and Random Search (RS). SA performed best and is the recommended method.

Selections are scored with the **Sliced Seasonal Wasserstein Distance (SSWD)**: the multivariate Sliced Wasserstein Distance between the selected subset and the full reference ensemble is computed per season, and the worst-performing season sets the score. Lower is better. In annual mode (`SEASONAL = False`) the plain Sliced Wasserstein Distance (SWD) over the whole year is used instead.

The result is a list of `(climate model, year)` pairs, e.g. 5 or 30 years, that can be used as weather years in an energy system model instead of the full ensemble.

## Reference

This code accompanies the paper:

> Bram van Duinen, Karin van der Wiel, Jean Thorey, Laurens Stoop.
> *Bridging the climate to energy data gap: simulated annealing for representative climate year selection.*
> Accepted for publication in **Applied Energy**.
> Preprint: https://doi.org/10.48550/arXiv.2605.15958

If you use this code, please cite the paper above (see also `CITATION.cff`).

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

## Quick start

1. Install (see above).
2. Copy `run/RUN_SETTINGS_example.py` to `run/RUN_SETTINGS.py` and set at least `BASE_INPUT`, `BASE_OUTPUT` and `MATCH_COUNTRIES`.
3. Get the input data (see [Input data](#input-data)): run `00a` and `00b` once, and put the ERAA2024 files in `BASE_INPUT`.
4. Run `01_build_dataset.ipynb` once to build the reference dataset.
5. Select years. There are two options:
   - **Simulated Annealing only** (recommended for new applications):
     ```bash
     cd run
     python 02b_run_simulated_annealing.py
     ```
     The selected years are in `BASE_OUTPUT/<RUNNAME>_<timestamp>/best_selection.csv`. All SA experiments, ranked by score, are in `BASE_OUTPUT/<RUNNAME>_<timestamp>/simulated_annealing/sa_year_selections_scored.csv`.
   - **Full paper pipeline** (all four methods, to replicate the paper's comparison):
     ```bash
     cd run
     python 02a_run_all_selection_methods.py
     ```
     All selections, ranked by score, are in `BASE_OUTPUT/<RUNNAME>_<timestamp>/all_year_selections_scored.csv`. Afterwards, `03_evaluation_plots.py` makes the evaluation figures.

## Configuration

All settings are in **`run/RUN_SETTINGS.py`**. Copy it from `run/RUN_SETTINGS_example.py`. `RUN_SETTINGS.py` is gitignored, so local paths and run choices never end up in version control. Edit `RUN_SETTINGS_example.py` only when the *set* of available settings changes.

The main settings:

| Setting | Description |
|---|---|
| `BASE_INPUT`, `BASE_OUTPUT` | Input data folder and output folder. All other paths <br> (`INPUT_DIR_*`, `DF_*_FNAME`, ERAA files) are derived from these |
| `RUNNAME` | Name for the output directory of this run |
| `MATCH_COUNTRIES` | List of country codes for which you want to <br> find representative years (e.g. `["NL", "DE"]`) |
| `NUM_TO_SELECT` | Number of representative years to select |
| `HYDRO` | Whether to include hydro variables in the optimisation |
| `SEASONAL` | Use seasonal scoring (SSWD) instead of annual (SWD). Defaults to True. |
| `HYDROLOGICAL_YEARS` | Reference period: the years (of every model) that <br> together form the reference ensemble |
| `NUM_EXPERIMENTS` | Number of independent experiments per algorithm, <br> each with its own seed |
| `SEED_OFFSET` | Added to every experiment seed. Change it to get a <br> different stochastic outcome without re-tuning anything else. Defaults to 0. |
| `PARAM_GRID` | SA hyperparameters: `temp_start`, `cooling_rate`, `max_iter`. <br> Cooling rates are spread evenly across `NUM_EXPERIMENTS`, <br> all other entries must be single-element lists |

Scoring cost/accuracy trade-off. The SSWD is estimated from random projections, so more projections means a more accurate but slower score:

| Setting | Description |
|---|---|
| `PERFORM_TWO_STEP` | If `True`, score all candidates cheaply with `N_PROJECTIONS_INIT`, <br> then re-score the top 10% with `N_PROJECTIONS_FULL` |
| `N_PROJECTIONS_INIT` | Projections used for the cheap screening pass (default 50) |
| `N_PROJECTIONS_FULL` | Projections used for the refinement pass (default 200) |

Filtered Random Search settings (only used by `02a`):

| Setting | Description |
|---|---|
| `P_THRES` | Percentile threshold for the univariate pre-filters <br> (guideline: ~25 for 1 country, ~50 for a few, ~85 for many, with 5 or 30 years) |
| `TUNING_ITERATIONS` | Iterations used to calibrate those thresholds |
| `FRS_ITER_MULTIPLIER` | FRS draws = `FRS_ITER_MULTIPLIER` × (total SA iterations). Defaults to 10. |

`src/climate_year_selection/constants.py` holds the definition of the PECD 4.2 dataset (climate models, years, countries) and the conventions used in the calculations. Normally it does not need to be changed. The one setting you might want to change is `SEASONS`, the season definition used by the SSWD (default: extended winter October–March and summer April–September).

## Input data

The pipeline needs:

- **PECD 4.2** climate-energy data (C3S, [`sis-energy-pecd`](https://cds.climate.copernicus.eu/datasets/sis-energy-pecd)): wind and solar capacity factors, 2 m temperature and hydropower generation, for 6 CMIP6 models under SSP2-4.5. Downloaded by `00a` (requires a CDS account and API key).
- **ERAA2024** installed generation capacities and annual demand (ENTSO-E). Download "Dashboard - Raw data" (ZIP) under Input Data on the [ERAA 2024 modelling data page](https://www.entsoe.eu/eraa/2024/modelling-data/#Input%20Data), and save `GenerationCapacities.xlsx` and `Aggregated_Demand.xlsx` from it as `ERAA2024_GenerationCapacities.xlsx` and `ERAA2024_Aggregated_Demand.xlsx` in `BASE_INPUT` (or point `ERAA_CAPACITIES_FNAME` and `ERAA_DEMAND_FNAME` in `RUN_SETTINGS.py` to them). Used in `01` to convert capacity factors to production and to scale demand per country.
- **PECD shapefiles** (`NUT0_ADM0_PECD.zip` and `PECD42_v2_final.zip`) in `run/shapefiles/`, only needed for the maps in `00b`. They can be found via the [C3S aggregation notebook](https://github.com/ecmwf-training/dss-notebooks/blob/main/datasets/sis-energy-pecd/aggregation-pecd-regions-into-nut0/aggregation-pecd-regions-into-nut0.ipynb). The wind aggregation weights are included in `run/weights/`.

Expected layout of `BASE_INPUT` (folder names can be changed in `RUN_SETTINGS.py`):

```
BASE_INPUT/
    ERAA2024_GenerationCapacities.xlsx
    ERAA2024_Aggregated_Demand.xlsx
    pecd_data/energy/
        wind_peof_peon/      # raw wind per PECD zone (00a)
        wind/                # wind aggregated to NUTS0 (00b)
        solar/               # (00a)
        temp/                # (00a)
        hydro_weekly/        # raw weekly hydro (00a)
        hydro/               # hourly hydro per country (01)
    preprocessed_dataframes/ # dataframes built in 01
```

## Workflow

All scripts are in the `run/` directory and should be run from there (they import `RUN_SETTINGS.py` from the current directory). Steps 00a/00b only need to be run once to obtain the raw data. Step 01 only has to be run once to build the reference dataset.

### 00a. Download PECD weather/energy data *(optional)*
`run/00a_download_pecd_energy.py`

Download raw PECD climate and energy data via cdsapi, if not already available locally.

### 00b. Aggregate wind to NUTS0 *(optional)*
`run/00b_aggregate_wind_to_nuts0.ipynb`

Aggregate raw wind capacity factors from PEOF/PEON zones to country (NUTS0) level. Notebook adapted from the [C3S aggregation notebook](https://github.com/ecmwf-training/dss-notebooks/blob/main/datasets/sis-energy-pecd/aggregation-pecd-regions-into-nut0/aggregation-pecd-regions-into-nut0.ipynb). Run it once for `tech_type = "Onshore"` and once for `"Offshore"`.

### 01. Build reference dataset
`run/01_build_dataset.ipynb`

Combines wind, solar, demand and hydro into a single daily
production-demand dataframe (`df_prod_demand_daily.parquet`) used by all
downstream steps. Here, enter any changes in installed capacity or demand-temperature assumptions you want to make.

### 02. Select representative years

There are two options. For a new application, use **02b**. Use **02a** only to replicate the method comparison from the paper.

#### 02b. Simulated Annealing only *(recommended)*
`run/02b_run_simulated_annealing.py`

Runs only Simulated Annealing, the best-performing method in the paper. It runs `NUM_EXPERIMENTS` independent SA experiments (one seed each), re-scores them and keeps the best. Results are written to `BASE_OUTPUT/<RUNNAME>_<timestamp>/`:
- `best_selection.csv`: the selected years, one `(model, year)` pair per row.
- `simulated_annealing/sa_year_selections_scored.csv`: all SA experiments with their scores, best first.
- `simulated_annealing/plots/`: a convergence plot per experiment.

Uses the general and SA settings from `RUN_SETTINGS.py`; the FRS settings are ignored. Set `RUN_PHYSICAL_EVALUATION = True` at the top of the script for extra diagnostic plots per country.

```bash
cd run
python 02b_run_simulated_annealing.py
```

#### 02a. All selection methods *(paper workflow)*
`run/02a_run_all_selection_methods.py`

Runs all four selection algorithms and produces automatic evaluation plots. Results are written to `BASE_OUTPUT/<RUNNAME>_<timestamp>/`, with one subdirectory per method (`simulated_annealing/`, `random_search/`, `k_medoids/`, `filtered_random_search/`) plus `evaluation/` (and `evaluation/by_country/`). All selections, ranked by score, are in `all_year_selections_scored.csv`.

The methods run in this order, with Random Search and K-Medoids matched to SA's compute budget:

1. **Simulated Annealing (SA)**
2. **Random Search (RS)**
3. **K-Medoids**
4. **Filtered Random Search (FRS)**

```bash
cd run
python 02a_run_all_selection_methods.py
```

Both 02a and 02b save a copy of `RUN_SETTINGS.py` alongside the results as `RUN_SETTINGS_of_run.py`, for record keeping.

### 03. Evaluation plots
`run/03_evaluation_plots.py`

Generates detailed evaluation figures and summary tables from a completed `02a` run (it compares the best selection of each method). Set `RUNNAME_` at the top of the script to the output folder of that run.

## Project structure

```
src/climate_year_selection/      # core library
    constants.py                 # PECD dataset definition, seasons, quantiles
    cost_functions.py            # (S)SWD scoring functions used by the search
    sa.py                        # simulated annealing
    k_medoids_stochastic.py      # k-medoids, stochastic variant
    filtered_random_search.py    # FRS: univariate pre-filters + random sampling
    evaluation.py                # scoring and seed-uncertainty analysis
    evaluation_plot_helpers.py   # figure generation
    helper_functions.py          # building dataframes from PECD files and selections

run/                             # pipeline scripts and notebooks
    RUN_SETTINGS.py              # all run configuration (gitignored; copy from example)
    RUN_SETTINGS_example.py      # template — copy to RUN_SETTINGS.py
    00a_download_pecd_energy.py
    00b_aggregate_wind_to_nuts0.ipynb
    01_build_dataset.ipynb
    02a_run_all_selection_methods.py
    02b_run_simulated_annealing.py
    03_evaluation_plots.py
    illustrations/               # standalone scripts for paper figures
    weights/                     # PECD capacity weights for wind aggregation
```

## License

MIT, see `LICENSE`.
