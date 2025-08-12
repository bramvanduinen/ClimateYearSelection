# %%
import os
import json
import warnings
import pandas as pd
import random

import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp
from typing import List, Tuple, Sequence, Callable, Optional
from itertools import product, combinations
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from cluster_weather_years.functions import build_df
from cluster_weather_years.constants import BASE_OUTPUT, MODELS, YEARS, NUM_CLUSTERS

# %%
data = "energy" # energy or meteo
match_vars = "full" # optimize on full (excl residual) or on just residual load
DF_DAILY_FNAME = os.path.join(BASE_OUTPUT, "dataframes/df_daily.csv")
DF_PROD_DEMAND_FNAME = os.path.join(BASE_OUTPUT, "dataframes/df_prod_demand_daily.csv")
OUTPUT_DIR = os.path.join(BASE_OUTPUT, "simulated_annealing")
# %%
def standardize_df(
    df: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    """
    Standardise `variables` to zero mean and unit variance (z-score).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    variables : Sequence[str]
        Columns to standardise.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with standardised variables.
    """
    df_std = df.copy()
    df_std[variables] = (df_std[variables] - df_std[variables].mean()) / df_std[
        variables
    ].std()
    return df_std

def ks_total(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Unweighted Kolmogorov–Smirnov score (lower ⇒ closer)."""
    mask = df[[model_col, year_col]].apply(tuple, axis=1).isin(combos)
    sub = df[mask]
    return sum(
        ks_2samp(df[var].values, sub[var].values, method="auto").statistic
        for var in variables
    )

def _ad_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Anderson–Darling statistic with p-value warnings silenced."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"p-value (?:capped|floored).*",
            category=UserWarning,
        )
        return anderson_ksamp([a, b]).statistic

def ad_total(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Anderson–Darling k-sample score (lower ⇒ closer; tail-weighted)."""
    mask = df[[model_col, year_col]].apply(tuple, axis=1).isin(combos)
    sub = df[mask]
    return sum(_ad_stat(df[v].values, sub[v].values) for v in variables)


def joint_tail_penalty(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    tails: dict,  # e.g. {"wind_off": "low", "wind_on": "low", "solar": "low", "temp": "low"}
    probs: Sequence[float] = (0.10, 0.05, 0.02),
    max_order: int = 3,
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Match joint tail co-exceedance rates between full sample and selected years.
       Lower = better. Supports mixed tail directions via `tails`."""
    idx_sel = df[[model_col, year_col]].apply(tuple, axis=1).isin(combos).to_numpy()
    X = df[variables].to_numpy()
    # Drop rows with any NaNs across variables
    ok = ~np.isnan(X).any(axis=1)
    X, idx_sel = X[ok], idx_sel[ok]
    X_sub = X[idx_sel]

    d = len(variables)
    eps = 1e-12
    score = 0.0

    # Precompute full-sample quantiles for each tail severity
    qmap = {p: np.array([
        np.quantile(X[:, j], p if tails.get(variables[j], "low") == "low" else 1 - p)
        for j in range(d)
    ]) for p in probs}

    for p in probs:
        q = qmap[p]
        # Indicators for each variable being in the requested tail (full & subset)
        ind_full = []
        ind_sub = []
        for j, v in enumerate(variables):
            if tails.get(v, "low") == "low":
                ind_full.append(X[:, j] <= q[j])
                ind_sub.append(X_sub[:, j] <= q[j])
            else:
                ind_full.append(X[:, j] >= q[j])
                ind_sub.append(X_sub[:, j] >= q[j])
        ind_full = np.array(ind_full)   # (d, n_full)
        ind_sub  = np.array(ind_sub)    # (d, n_sub)

        for order in range(1, min(max_order, d) + 1):
            for idxs in combinations(range(d), order):
                # calculate mean of joint exceedance rate for both full and subset
                jf = np.logical_and.reduce([ind_full[k] for k in idxs]).mean()
                js = np.logical_and.reduce([ind_sub[k]  for k in idxs]).mean()
                if jf < eps:
                    continue  # nothing to match at this severity
                # relative squared error of joint rate
                score += ((js - jf) / (jf + eps)) ** 2

    return float(score)

def sa_cost(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    tails: dict,
    w_ad: float = 1.0,
    w_joint: float = 1.0,
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    return (
        w_ad * ad_total(df, combos, variables, model_col=model_col, year_col=year_col)
        + w_joint * joint_tail_penalty(df, combos, variables, tails, model_col=model_col, year_col=year_col)
    )


def simulated_annealing_selection(
    df: pd.DataFrame,
    variables: Sequence[str],
    n_select: int,
    tails: dict,
    model_col: str = "model",
    year_col: str = "year",
    scoring_fn: Callable[..., float] = ad_total,
    ref_scoring_fn: Callable[..., float] = ks_total,
    temp_start: float = 1.0,
    temp_end: float = 1e-3,
    cooling_rate: float = 0.98,
    max_iter: int = 10_000,
    random_state: Optional[int] = None,
) -> Tuple[List[Tuple[str, int]], float, pd.DataFrame]:
    """
    Simulated annealing that optimises `scoring_fn` (default AD) and logs the
    reference score from `ref_scoring_fn` (default unweighted KS).

    Returns
    -------
    (best_combos, best_score, log_df)
        best_combos : list[(model, year)]
        best_score  : float (optimised score)
        log_df      : pandas.DataFrame with columns
                      ["iter", "temp", "score_opt", "score_ref"]
    """
    rng = np.random.default_rng(random_state)
    all_combos = list(
        df[[model_col, year_col]].drop_duplicates().itertuples(index=False, name=None)
    )
    if n_select > len(all_combos):
        raise ValueError("n_select exceeds available (model, year) pairs")

    idx = rng.choice(len(all_combos), n_select, replace=False)
    current = [all_combos[i] for i in idx]
    best = current.copy()
    if scoring_fn == sa_cost:
        current_score = best_score = scoring_fn(
            df = df, combos = current, variables = variables, tails=tails, model_col=model_col, year_col=year_col
        )
    else:
        current_score = best_score = scoring_fn(
            df = df, combos = current, variables = variables, model_col=model_col, year_col=year_col
        )

    if ref_scoring_fn == sa_cost:
        ref_score = ref_scoring_fn(
            df, current, variables, tails=tails, model_col=model_col, year_col=year_col
        )
    else:
        ref_score = ref_scoring_fn(
            df, current, variables, model_col=model_col, year_col=year_col
        )

    temp = temp_start
    log: list[dict] = [
        {
            "iter": 0,
            "temp": temp,
            "score_opt": current_score,
            "score_ref": ref_score,
        }
    ]

    for it in tqdm(range(1, max_iter + 1)):
        if temp < temp_end:
            break

        remaining = [c for c in all_combos if c not in current]
        out_idx = rng.integers(n_select)
        in_combo = remaining[rng.integers(len(remaining))]

        neighbour = current.copy()
        neighbour[out_idx] = in_combo
        if scoring_fn == sa_cost:
            neighbour_score = scoring_fn(
                df = df, combos = neighbour, variables = variables, tails=tails, model_col=model_col, year_col=year_col
            )
        else:
            neighbour_score = scoring_fn(
                df = df, combos = neighbour, variables = variables, model_col=model_col, year_col=year_col
            )

        delta = neighbour_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current, current_score = neighbour, neighbour_score
            if current_score < best_score:
                best, best_score = current.copy(), current_score

        if ref_scoring_fn == sa_cost:
            ref_score = ref_scoring_fn(
                df, current, variables, tails=tails, model_col=model_col, year_col=year_col
            )
        else:
            ref_score = ref_scoring_fn(
                df, current, variables, model_col=model_col, year_col=year_col
            )
        log.append(
            {
                "iter": it,
                "temp": temp,
                "score_opt": current_score,
                "score_ref": ref_score,
            }
        )
        temp *= cooling_rate

    return sorted(best), best_score, ref_score, pd.DataFrame(log)

# %%
# Build NL dataframe
if data == "meteo":
    if match_vars == "full":
        vars_to_match = ["NL_OFF", "NL_ON", "NL_SOLAR", "NL_TEMP"]
        tails = {"NL_OFF": "low", "NL_ON": "low", "NL_SOLAR": "low", "NL_TEMP": "low"}
    elif match_vars == "residual":
        vars_to_match = ["Residual_ish"]
        tails = {"Residual_ish": "high"}
    print(vars_to_match)
    if not os.path.exists(DF_DAILY_FNAME):
        print("Calculating df_daily...")
        df_daily, _ = build_df()
        df_daily = df_daily.reset_index()
        #df_daily["year"] = df_daily["Date"].dt.year
        df_daily.to_csv(DF_DAILY_FNAME)
    else:
        print("Taking df_daily from memory...")
        df_daily = pd.read_csv(DF_DAILY_FNAME, index_col=0)

    vars_to_standardize = ["NL_OFF", "NL_ON", "NL_SOLAR", "NL_TEMP"]
    df_std = standardize_df(df_daily, vars_to_standardize)
    # compute some residual load, but not any weighting yet (e.g. based on capacity). Just to see joint-tail behavior
    df_std["Residual_ish"] = df_std["NL_TEMP"] + (df_std["NL_SOLAR"] + df_std["NL_OFF"] + df_std["NL_ON"])
elif data == "energy":
    print("Taking df_prod_demand_daily from memory...")
    if match_vars == "full":
        vars_to_match = ["NL_OFF_prod", "NL_ON_prod", "NL_SOLAR_prod", "Demand"]
        vars_to_standardize = ["NL_OFF_prod", "NL_ON_prod", "NL_SOLAR_prod", "Demand"]
        tails = {"NL_OFF_prod": "low", "NL_ON_prod": "low", "NL_SOLAR_prod": "low", "Demand": "high"}
    elif match_vars == "residual":
        vars_to_match = ["Residual_load"]
        vars_to_standardize = ["Residual_load"]
        tails = {"Residual_load": "high"}
    print(vars_to_match)
    df_daily = pd.read_csv(DF_PROD_DEMAND_FNAME, index_col=0)
    df_std = standardize_df(df_daily, vars_to_standardize)

#%%
def run_sa(metric, t0=2.0, cr=0.98, mi=5_000, seed=42):
    print(f"Starting Simulated Annealing procedure, metric = {metric.__name__}...")
    print(f"Year column: {year_col}")
    if metric == sa_cost:
        alternative_metric = ad_total
    else:
        alternative_metric = sa_cost
    selected, best_score, best_ref_score, log_df = simulated_annealing_selection(
        df=df_std,
        variables=vars_to_match,
        scoring_fn=metric, # now implement the cost function with joint exceedance in it
        ref_scoring_fn = alternative_metric, # this is the reference score, the one that was originally used
        n_select=NUM_CLUSTERS,
        year_col=year_col,
        tails=tails,
        temp_start=t0,
        cooling_rate=cr,
        max_iter=mi,
        random_state=seed,
    )
    sel_dict = {"selected": selected, "best_score": best_score, "best_alternative_score": best_ref_score}
    json.dump(sel_dict, open(os.path.join(OUTPUT_DIR, f"{data}_selected_{match_vars}_{year_col}_{metric.__name__}_{t0}_{cr}_{mi}_{seed}.json"), "w"))

    print("Selected (model, year) pairs:", selected)
    print(f"Best score: {best_score:.4f}")
    print(log_df.tail())          # shows the last few iterations
    return selected, best_score, best_ref_score


# %%
metric = sa_cost # sa_cost, or ad_total
year_col = "Hydrological_year" # Hydrological_year or Calendar_year
# just one run
#selected, best_score, best_ref_score = run_sa(metric)

#%%
i = 0
param_grid = {
    "metric": [sa_cost, ad_total],
    "temp_start": [2.0], #[0.5, 1.0, 2.0], #higher start doesn't really help
    "cooling_rate": [0.98, 0.99], #higher cooling rate (i.e. slower cooling) does help
    "max_iter": [5_000], #doesn't even reach this, I think
}
seeds = range(3)  # maybe more later

results = []
for seed, (metric, t0, cr, mi) in product(seeds, product(*param_grid.values())):
    selected, best_score, best_ref_score = run_sa(metric, t0, cr, mi, seed)

    results.append({"seed": seed, "metric": metric.__name__, "t0": t0, "cr": cr, "score": best_score, "sel": selected})
    print(i, {"seed": seed, "metric": metric.__name__, "t0": t0, "cr": cr, "score": best_score, "sel": selected})
    i += 1

best = min(results, key=lambda d: d["score"])
print(best["score"], best["sel"])
#%%