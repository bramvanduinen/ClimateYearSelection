import warnings
from itertools import combinations
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ot import sliced_wasserstein_distance
from scipy.stats import anderson_ksamp, ks_2samp
from tqdm import tqdm

from cluster_weather_years.constants import SEASONS

# %% Main SA algorithm


def simulated_annealing_selection(
    df: pd.DataFrame,
    variables: Sequence[str],
    n_select: int,
    scoring_fn: Callable[..., float],
    ref_scoring_fn: Callable[..., float],
    context: Optional[dict] = None,
    model_col: str = "model",
    year_col: str = "year",
    temp_start: float = 1.0,
    temp_end: float = 1e-3,
    cooling_rate: float = 0.98,
    max_iter: int = 10_000,
    random_state: Optional[int] = None,
    sel_type="full",  # full to sample from full df, or 'subset' to choose mini-subset from larger subset.
    subset=None,  # only specified if sel_type = "subset"
) -> Tuple[list[Tuple[str, int]], float, float, pd.DataFrame]:
    """
    Simulated annealing that optimises `scoring_fn` and logs the reference score
    from `ref_scoring_fn` at each step.

    Returns
    -------
    (best_combos, best_score, best_ref_score, log_df)
    - best_combos : list[(model, year)]
    - best_score  : float (optimised score)
    - best_ref_score : float (reference score for best selection)
    - log_df      : pandas.DataFrame with columns ["iter", "temp", "score_opt", "score_ref"]
    """
    rng = np.random.default_rng(random_state)

    # --- SA control tweaks ---
    if cooling_rate < 0.99:
        REHEAT_G = 100  # iterations without new best before reheating
    else:
        REHEAT_G = 300

    REHEAT_GAMMA = 1.8  # reheating factor
    accepted = 0
    steps_since_best = 0

    if sel_type == "full":
        all_combos = list(
            df[[model_col, year_col]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    elif sel_type == "subset":
        all_combos = subset

    # initialize first random draw and calculate scores
    idx = rng.choice(len(all_combos), n_select, replace=False)
    current = [all_combos[i] for i in idx]
    best = current.copy()

    current_score = best_score = scoring_fn(
        df=df,
        combos=current,
        variables=variables,
        context=context,
        model_col=model_col,
        year_col=year_col,
    )

    # reference score, not optimized on!
    ref_score = best_ref_score = ref_scoring_fn(
        df=df,
        combos=current,
        variables=variables,
        context=context,
        model_col=model_col,
        year_col=year_col,
    )

    temp = temp_start
    log: list[dict] = [
        {"iter": 0, "temp": temp, "score_opt": current_score, "score_ref": ref_score}
    ]

    for it in tqdm(range(1, max_iter + 1)):
        if temp < temp_end:
            break

        # --- mixed-move: 1-swap / multi-swap ---
        remaining = [c for c in all_combos if c not in current]

        u = rng.random()
        if n_select < 10:  # built on n_select = 5
            # in 75% of cases just swap out one
            if u < 0.75:
                k = 1
            else:
                k = int(rng.integers(2, 4))  # 2 or 3

        # TODO: Tweak with this. Does this swap enough/too much?
        elif n_select > 10:  # built on n_select = 30
            if u < 0.5:
                k = int(rng.integers(1, 4))  # 1, 2, or 3
            elif u < 0.85:
                k = int(rng.integers(4, 7))  # 4, 5, or 6
            else:
                k = int(rng.integers(7, 11))  # 7, 8, 9, or 10

        k = int(min(k, n_select, len(remaining)))
        out_idx = rng.choice(n_select, size=k, replace=False)
        in_idx = rng.choice(len(remaining), size=k, replace=False)

        neighbour = current.copy()
        for oi, ii in zip(np.atleast_1d(out_idx), np.atleast_1d(in_idx)):
            neighbour[oi] = remaining[int(ii)]

        neighbour_score = scoring_fn(
            df=df,
            combos=neighbour,
            variables=variables,
            context=context,
            model_col=model_col,
            year_col=year_col,
        )

        ref_score = ref_scoring_fn(
            df=df,
            combos=current,
            variables=variables,
            context=context,
            model_col=model_col,
            year_col=year_col,
        )

        # --- Metropolis accept/reject with tracking ---
        delta = neighbour_score - current_score
        accept = (delta < 0) or (rng.random() < np.exp(-delta / temp))
        if accept:
            # update scores
            current, current_score = neighbour, neighbour_score
            accepted += 1
            if current_score < best_score:
                best, best_score, best_ref_score = (
                    current.copy(),
                    current_score,
                    ref_score,
                )
                steps_since_best = 0
        else:
            steps_since_best += 1

        acc_rate = accepted / it

        log.append(
            {
                "iter": it,
                "temp": temp,
                # "score_opt": current_score,
                "score_accepted": current_score,
                "score_best": best_score,
                "score_tried": neighbour_score,
                "score_ref": ref_score,  # CAN PROBABLY BE REMOVED, NOT THAT USEFUL ANYMORE
                "acc_rate": acc_rate,
            }
        )

        # --- reheating on stalls (kick out of plateaus) ---
        if steps_since_best > REHEAT_G:
            print(f"Reheating on stalls: {steps_since_best}")
            temp *= REHEAT_GAMMA
            steps_since_best = 0

        # cooling step
        temp *= cooling_rate

    return sorted(best), float(best_score), float(best_ref_score), pd.DataFrame(log)


# %% SCORES
def build_selection_mask(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    model_col: str = "model",
    year_col: str = "year",
) -> np.ndarray:
    """Return a boolean mask selecting rows whose (model, year) is in combos."""
    mi = pd.MultiIndex.from_arrays([df[model_col].to_numpy(), df[year_col].to_numpy()])
    return mi.isin(set(combos))  # .to_numpy()


def wasserstein_score(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    context: dict,
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Wasserstein score: Wasserstein distance between full sample and selected years."""
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)

    sub_data = df.loc[idx_sel, variables].to_numpy()
    full_data = df[variables].to_numpy()

    # Replace NaNs with zeros to avoid issues in distance calculation
    sub_data[np.isnan(sub_data)] = 0
    full_data[np.isnan(full_data)] = 0

    # 50 projections is compromise between accuracy and speed. Better would be: 200
    return sliced_wasserstein_distance(sub_data, full_data, n_projections=50)


def wasserstein_seasonal_score(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    context: dict,
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Wasserstein score: Wasserstein distance between full sample and selected years."""
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)

    wasserstein_seasonal = []
    for season, months in SEASONS.items():
        in_season = df["Date"].dt.month.isin(months).values
        # combine random draw boolean & seasonal boolean for subset
        sub_mask = idx_sel & in_season
        sub_data = df.loc[sub_mask, variables].to_numpy()
        # only use seasonal boolean for full set
        full_data = df.loc[in_season, variables].to_numpy()

        # Replace NaNs with zeros to avoid issues in distance calculation
        sub_data[np.isnan(sub_data)] = 0
        full_data[np.isnan(full_data)] = 0

        # 50 projections is compromise between accuracy and speed. Better would be: 200
        wasserstein_seasonal.append(
            sliced_wasserstein_distance(sub_data, full_data, n_projections=50)
        )
    # base score on worst-performing season
    return max(wasserstein_seasonal)


def ks_total(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    context: Optional[dict] = None,
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Unweighted Kolmogorov–Smirnov score (lower ⇒ closer)."""
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    sub = df.loc[idx_sel, variables]
    return float(
        sum(
            ks_2samp(df[v].to_numpy(), sub[v].to_numpy(), method="auto").statistic
            for v in variables
        )
    )


def _ad_stat(a: np.ndarray, b: np.ndarray) -> float:
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
    context: Optional[dict] = None,
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Anderson–Darling k-sample score (lower ⇒ closer; tail-weighted)."""
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    sub = df.loc[idx_sel, variables]
    full_disjoint = df.loc[~df.index.isin(sub.index), variables]
    return float(
        sum(_ad_stat(full_disjoint[v].to_numpy(), sub[v].to_numpy()) for v in variables)
    )


def ad_seasonal(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    context: Optional[dict] = None,
    model_col: str = "model",
    year_col: str = "year",
    date_col: str = "Date",
) -> float:
    """
    Compute Anderson–Darling k-sample statistics for each season separately.

    Args:
        df: Input DataFrame with datetime column.
        combos: Tuples of (model, year) for selection.
        variables: Variables to evaluate.
        context: Optional context (unused, for compatibility).
        model_col: Column with model identifiers.
        year_col: Column with year identifiers.
        date_col: Column with datetime values.

    Returns:
        AD Scores summed over variable and seasons.
    """
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    sub = df.loc[idx_sel, [date_col] + variables]
    full_disjoint = df.loc[~df.index.isin(sub.index), [date_col] + variables]

    # Ensure datetime
    sub[date_col] = pd.to_datetime(sub[date_col])
    full_disjoint[date_col] = pd.to_datetime(full_disjoint[date_col])

    # Add seasons
    sub["season"] = sub[date_col].dt.month % 12 // 3 + 1
    full_disjoint["season"] = full_disjoint[date_col].dt.month % 12 // 3 + 1
    season_map = {1: "DJF", 2: "MAM", 3: "JJA", 4: "SON"}

    result = 0

    # Seasonal distributions
    for v in variables:
        for s_code, s_name in season_map.items():
            sub_s = sub.loc[sub["season"] == s_code, v].to_numpy()
            full_s = full_disjoint.loc[full_disjoint["season"] == s_code, v].to_numpy()
            if len(sub_s) > 0 and len(full_s) > 0:
                result += float(_ad_stat(full_s, sub_s))

    return result


def precompute_tail_quantiles(
    df: pd.DataFrame,
    variables: Sequence[str],
    tails: dict,
    probs: Sequence[float] = (0.10, 0.05, 0.02),
) -> tuple[dict[float, np.ndarray], np.ndarray, tuple[float, ...]]:
    """Precompute per-variable thresholds for each probability and tail direction.

    Returns (qmap, is_high, probs_tuple)
    - qmap: dict[p] -> np.ndarray of thresholds aligned with `variables`
    - is_high: boolean array indicating high-tail for each variable
    - probs_tuple: tuple of probabilities in increasing severity order
    """
    X = df[variables].to_numpy()
    is_high = np.array([tails.get(v, "low") == "high" for v in variables], dtype=bool)
    probs_tuple = tuple(probs)
    qmap: dict[float, np.ndarray] = {}
    for p in probs_tuple:
        low = np.quantile(X, p, axis=0)
        high = np.quantile(X, 1 - p, axis=0)
        qmap[p] = np.where(is_high, high, low)
    return qmap, is_high, probs_tuple


def joint_tail_penalty(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    context: dict,
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Match joint tail co-exceedance rates between full sample and selected years.

    Lower = better. Uses precomputed thresholds and tail directions from `context`.
    """
    if context is None:
        raise ValueError(
            "joint_tail_penalty requires a context with 'qmap' and 'is_high'"
        )

    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    X = df[variables].to_numpy()
    # Drop rows with any NaNs across variables for full sample and subset
    ok = ~np.isnan(X).any(axis=1)
    X = X[ok]
    X_sub = X[idx_sel[ok]]

    is_high: np.ndarray = context["is_high"]
    qmap: dict[float, np.ndarray] = context["qmap"]
    probs: tuple[float, ...] = context.get("probs", tuple(qmap.keys()))
    max_order: int = int(context.get("max_order", 3))

    d = len(variables)
    eps = 1e-12
    score = 0.0

    for p in probs:
        q = qmap[p]
        # Indicators for each variable being in the requested tail
        # low-tail => <= q; high-tail => >= q
        ind_full = np.where(is_high, X >= q, X <= q)
        ind_sub = np.where(is_high, X_sub >= q, X_sub <= q)

        for order in range(1, min(max_order, d) + 1):
            for idxs in combinations(range(d), order):
                jf = np.logical_and.reduce([ind_full[:, k] for k in idxs]).mean()
                js = np.logical_and.reduce([ind_sub[:, k] for k in idxs]).mean()
                if jf < eps:
                    continue
                score += ((js - jf) / (jf + eps)) ** 2

    return float(score)


def combined_score(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    context: dict,
    model_col: str = "model",
    year_col: str = "year",
) -> float:
    """Combined score: AD distance + joint-tail penalty with weights from context."""
    w_ad: float = float(context.get("w_ad", 1.0))
    w_joint: float = float(context.get("w_joint", 1.0))
    return w_ad * ad_total(
        df, combos, variables, context, model_col=model_col, year_col=year_col
    ) + w_joint * joint_tail_penalty(
        df, combos, variables, context, model_col=model_col, year_col=year_col
    )
