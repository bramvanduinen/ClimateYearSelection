# %%
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from sa_settings import (
    DF_PROD_DEMAND_FNAME,
    MATCH_COUNTRIES,
    MATCH_VARS,
    NUM_CLUSTERS,
    PARAM_GRID,
    SEEDS,
    YEAR_COL,
)

from cluster_weather_years.constants import (
    SA_OUTPUT_DIR,
)
from cluster_weather_years.functions import standardize_df
from cluster_weather_years.sa import (
    ad_total,
    combined_score,
    precompute_tail_quantiles,
    simulated_annealing_selection,
)

# %%
# create folder for output of this run
rundate = datetime.now().strftime("%Y%m%d_%H%M")
country_string = "_".join(MATCH_COUNTRIES)
SA_DIR_RUN = os.path.join(SA_OUTPUT_DIR, f"run_{country_string}_{rundate}/")
os.makedirs(SA_DIR_RUN, exist_ok=True)
# export copy of sa_settings.py to outputs/simulated_annealing/sa_settings.py
shutil.copy("sa_settings.py", os.path.join(SA_DIR_RUN, "sa_settings_of_run.py"))
# %%
# Build dataframes and standardize
print("Taking df_prod_demand_daily from memory...")
vars_to_match = (
    [f"{country}_OFF_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_ON_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_SOLAR_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_Demand" for country in MATCH_COUNTRIES]
)
vars_to_standardize = vars_to_match.copy()

df_daily = pd.read_csv(DF_PROD_DEMAND_FNAME, index_col=0)
df_std = standardize_df(df_daily, vars_to_standardize)

# remove incomplete hydrological years, if this is the selected year column
if YEAR_COL == "Hydrological_year":
    df_daily = df_daily.query("Hydrological_year > 2014 and Hydrological_year < 2044")
    df_std = df_std.query("Hydrological_year > 2014 and Hydrological_year < 2044")

# %% calculate co-exceedance metric, only needed for metric = combined_score

# specify which tail is extreme, for tail co-exceedance
tails = (
    {f"{country}_OFF_prod": "low" for country in MATCH_COUNTRIES}
    | {f"{country}_ON_prod": "low" for country in MATCH_COUNTRIES}
    | {f"{country}_SOLAR_prod": "low" for country in MATCH_COUNTRIES}
    | {f"{country}_Demand": "high" for country in MATCH_COUNTRIES}
)

# Precompute quantiles and compose scoring context
qmap, is_high, probs = precompute_tail_quantiles(df_daily, vars_to_match, tails)
context = {
    "tails": tails,
    "qmap": qmap,
    "is_high": is_high,
    "probs": probs,
    "max_order": 3,
    "w_ad": 1.0,
    "w_joint": 1.0,
}


# %% Functions to run and log the SA algorithm in parallel mode


def plot_sa_log(log_df, metric, alternative_metric, t0, cr, mi, seed):
    """Plot evolution of scores after the run"""
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax2 = ax1.twinx()
    ax4 = ax3.twinx()

    label_opt = metric.__name__
    label_ref = alternative_metric.__name__

    # --- Top panel: scores ---
    ax1.plot(log_df["iter"], log_df["score_opt"], label=label_opt, color="tab:blue")
    ax1.set_ylabel(f"{label_opt} (lower is better)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2.plot(log_df["iter"], log_df["score_ref"], label=label_ref, color="tab:orange")
    ax2.set_ylabel(f"{label_ref} (lower is better)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title(f"SA evolution: t0={t0}, cr={cr}, mi={mi}, seed={seed}")

    # --- Bottom panel: acceptance rate ---
    ax3.plot(log_df["iter"], log_df["acc_rate"], color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")
    ax4.plot(log_df["iter"], log_df["temp"], color="tab:red")
    ax4.tick_params(axis="y", labelcolor="tab:red")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("acceptance rate")
    ax4.set_ylabel("temperature")

    fig.tight_layout()

    # Save fig
    plot_name = f"Selected_{MATCH_VARS}_{MATCH_COUNTRIES}_{YEAR_COL}_{metric.__name__}_{t0}_{cr}_{mi}_{seed}.png"
    os.makedirs(os.path.join(SA_DIR_RUN, "plots"), exist_ok=True)
    fig.savefig(os.path.join(SA_DIR_RUN, f"plots/{plot_name}"), bbox_inches="tight")
    plt.close(fig)


def run_sa(metric, t0=2.0, cr=0.98, mi=5_000, seed=42):
    """Run a single SA instance with given parameters and return results dict."""
    if metric == ad_total:
        alternative_metric = combined_score
    else:
        alternative_metric = ad_total

    selected, best_score, best_ref_score, log_df = simulated_annealing_selection(
        df=df_daily,
        variables=vars_to_match,
        scoring_fn=metric,
        ref_scoring_fn=alternative_metric,
        n_select=NUM_CLUSTERS,
        context=context,
        model_col="model",
        year_col=YEAR_COL,
        temp_start=t0,
        cooling_rate=cr,
        max_iter=mi,
        random_state=seed,
    )

    sel_dict = {
        "selected": selected,
        "optimized_metric": metric.__name__,
        "ref_metric": alternative_metric.__name__,
        "best_score": best_score,
        "best_ref_score": best_ref_score,
        "seed": seed,
        "t0": t0,
        "cr": cr,
        "mi": mi,
    }

    # plot logs of SA run and save the plot
    plot_sa_log(log_df, metric, alternative_metric, t0, cr, mi, seed)

    return sel_dict


def _sa_job(job):
    """Run a single SA job: (seed, metric, t0, cr, mi) -> sel_dict."""
    seed, metric, t0, cr, mi = job
    return run_sa(metric, t0, cr, mi, seed)


# %%
# Actual run
if __name__ == "__main__":
    jobs = [
        (seed, *vals) for seed, vals in product(SEEDS, product(*PARAM_GRID.values()))
    ]
    results = []

    n_workers = min(os.cpu_count() or 1, len(jobs))

    print("Starting Simulated Annealing procedure...")
    print(f"Selecting {NUM_CLUSTERS} years")
    print(f"Countries: {MATCH_COUNTRIES}")
    print(f"Year column: {YEAR_COL}")
    print(f"Parameter grid: {PARAM_GRID}")
    print(f"Number of seeds: {len(SEEDS)}")
    print(f"Number of experiments: {len(jobs)}")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_sa_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            sel_dict = fut.result()
            results.append(sel_dict)
            print(
                f"Experiment {i} done: {sel_dict['optimized_metric']} = {sel_dict['best_score']:.2f}"
            )

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(SA_DIR_RUN, "results_grid.csv"))
    best = min(results, key=lambda d: d["best_score"])
    print(
        f"Run done, best score: {best['best_score']:.2f}, best selection: {best['selected']}"
    )

# %%
