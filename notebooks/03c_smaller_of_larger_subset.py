# %%
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
from sa_settings import MATCH_COUNTRIES, YEAR_COL

from cluster_weather_years.constants import BASE_OUTPUT, SA_OUTPUT_DIR
from cluster_weather_years.sa import (
    ad_total,
    combined_score,
    simulated_annealing_selection,
    wasserstein_score,
    wasserstein_seasonal_score,
)

# %%
RUN_ID = "NL_20251024_1320"  # NL 30y; Run to analyze and compare to random draws
RUN_ID_FULL_5y = "NL_20251023_0942"  # NL 5y optimized run, to compare mini-subset with
NUM_CLUSTERS = 5  # Select mini-subset of 5 out of the 30y subset
CR = 0.975  # cooling rate

# %%
vars_to_match = (
    [f"{country}_OFF_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_ON_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_SOLAR_prod" for country in MATCH_COUNTRIES]
    + [f"{country}_Demand" for country in MATCH_COUNTRIES]
)

# %%
SA_DIR_RUN = os.path.join(
    SA_OUTPUT_DIR, f"subsampling/{RUN_ID}_{NUM_CLUSTERS}y_subset/"
)
os.makedirs(SA_DIR_RUN, exist_ok=True)

DF_PROD_DEMAND_FNAME = os.path.join(
    BASE_OUTPUT, "dataframes/df_prod_demand_daily_multi_country_WRONG_CAPACITIES.csv"
)

df_daily = pd.read_csv(DF_PROD_DEMAND_FNAME, index_col=0)
df_daily["Date"] = pd.to_datetime(df_daily["Date"])
df_daily = df_daily.query("Hydrological_year > 2014 and Hydrological_year < 2044")

df_selections = pd.read_csv(
    os.path.join(SA_OUTPUT_DIR, f"run_{RUN_ID}/results_grid.csv"), index_col=0
)
df_sel_multi_country = df_selections.query(
    "optimized_metric == 'wasserstein_score'"
).sort_values("best_score", ascending=True)

# %% [markdown]
# # Here copy-paste from sa experiment

# %%
# HACK: OBSOLETE
context = {
    "tails": None,
    "qmap": None,
    "is_high": None,
    "probs": None,
    "max_order": 3,
    "w_ad": 1.0,
    "w_joint": 1.0,
}


# %%
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
    plot_name = f"Selected_{MATCH_COUNTRIES}_{YEAR_COL}_{metric.__name__}_{t0}_{cr}_{mi}_{seed}.png"
    os.makedirs(os.path.join(SA_DIR_RUN, "plots"), exist_ok=True)
    fig.savefig(os.path.join(SA_DIR_RUN, f"plots/{plot_name}"), bbox_inches="tight")
    plt.close(fig)


# %%
def run_sa(metric, subset, t0=2.0, cr=0.9, mi=5_000, seed=42):
    """Run a single SA instance with given parameters and return results dict."""
    if metric == ad_total:
        alternative_metric = combined_score
    elif metric == wasserstein_score:
        alternative_metric = wasserstein_seasonal_score
    elif metric == wasserstein_seasonal_score:
        alternative_metric = wasserstein_score
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
        sel_type="subset",
        subset=subset,
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
    subset, metric, cr = job
    return run_sa(metric, subset=subset, cr=cr)


# %%
jobs = []
metric = wasserstein_score

for i in range(len(df_sel_multi_country)):
    modelyears_i = eval(df_sel_multi_country.iloc[i]["selected"])
    job = (modelyears_i, metric, CR)
    jobs.append(job)

# %%
results = []

n_workers = min(os.cpu_count() or 1, len(jobs))

print("Starting Simulated Annealing procedure...")
print(f"Selecting {NUM_CLUSTERS} years")
print(f"Countries: {MATCH_COUNTRIES}")
print(f"Year column: {YEAR_COL}")
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
df_results_pure_5y = pd.read_csv(f"{SA_OUTPUT_DIR}/run_{RUN_ID}/results_grid.csv")
plt.figure()
df_results["best_score"].hist(label="Subset sampling")
df_results_pure_5y["best_score"].hist(alpha=0.5, label="Full sampling")
plt.xlabel("Wasserstein")
plt.legend()
plt.savefig(f"{SA_DIR_RUN}/score_diff_subsampling_{NUM_CLUSTERS}y_from_{RUN_ID}.png")
