import os
from typing import Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from cluster_weather_years.scores import (
    wasserstein_score,
    wasserstein_seasonal_score,
)

# %% Main SA algorithm


def simulated_annealing_selection(
    df: pd.DataFrame,
    variables: Sequence[str],
    n_select: int,
    scoring_fn: Callable[..., float],
    model_col: str = "model",
    year_col: str = "year",
    temp_start: float = 1.0,
    temp_end: float = 1e-3,
    cooling_rate: float = 0.98,
    max_iter: int = 10_000,
    random_state: Optional[int] = None,
    sel_type="full",  # full to sample from full df, or 'subset' to choose mini-subset from larger subset.
    subset=None,  # only specified if sel_type = "subset"
) -> Tuple[list[Tuple[str, int]], float, pd.DataFrame]:
    """
    Simulated annealing that optimises `scoring_fn`.

    Returns
    -------
    (best_combos, best_score, log_df)
    - best_combos : list[(model, year)]
    - best_score  : float (optimised score)
    - log_df      : pandas.DataFrame with columns ["iter", "temp", "score_opt"]
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
    else:
        raise ValueError(f"Unknown sel_type: {sel_type}")

    # initialize first random draw and calculate scores
    idx = rng.choice(len(all_combos), n_select, replace=False)
    current = [all_combos[i] for i in idx]
    best = current.copy()

    current_score = best_score = scoring_fn(
        df=df,
        combos=current,
        variables=variables,
        model_col=model_col,
        year_col=year_col,
    )

    temp = temp_start
    log: list[dict] = [{"iter": 0, "temp": temp, "score_opt": current_score}]

    n_it = np.log(temp_end / temp_start) / np.log(cooling_rate)

    for it in tqdm(
        range(1, max_iter + 1), total=int(min(max_iter, n_it)), desc="SA iterations"
    ):
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
        else:  # built on n_select = 30
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
            model_col=model_col,
            year_col=year_col,
        )

        # --- Metropolis accept/reject with tracking ---
        random_check = rng.random()
        # first based on low-precision scores
        delta_init = neighbour_score - current_score
        close = (neighbour_score / current_score < 1.1) or (
            random_check < np.exp(-delta_init / temp)
        )
        if close and (
            scoring_fn == wasserstein_seasonal_score or scoring_fn == wasserstein_score
        ):
            # re-calculate score with higher precision
            neighbour_score = scoring_fn(
                df=df,
                combos=neighbour,
                variables=variables,
                model_col=model_col,
                year_col=year_col,
                n_projections=200,
            )

        delta = neighbour_score - current_score
        accept = (delta < 0) or (random_check < np.exp(-delta / temp))
        if accept:
            # update scores
            current, current_score = neighbour, neighbour_score
            accepted += 1
            if current_score < best_score:
                best, best_score = (
                    current.copy(),
                    current_score,
                )
                steps_since_best = 0
        else:
            steps_since_best += 1

        acc_rate = accepted / it

        log.append(
            {
                "iter": it,
                "temp": temp,
                "score_accepted": current_score,
                "score_best": best_score,
                "score_tried": neighbour_score,
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

    return sorted(best), float(best_score), pd.DataFrame(log)


# %% Functions to run and log the SA algorithm in parallel mode


def plot_sa_log(log_df, metric, t0, cr, mi, seed, output_dir_sa):
    """Plot evolution of scores after the run"""
    fig, (ax1, ax3) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # ax2 = ax1.twinx()
    ax4 = ax3.twinx()

    metric_label = metric.__name__

    # --- Top panel: scores ---
    ax1.plot(log_df["iter"], log_df["score_tried"], label="Tried", color="C0")
    ax1.plot(log_df["iter"], log_df["score_accepted"], label="Accepted", color="C1")
    ax1.plot(log_df["iter"], log_df["score_best"], label="Best", color="tab:green")
    ax1.set_ylabel(f"{metric_label} (lower is better)")
    ax1.legend(loc="upper right")
    # ax1.tick_params(axis="y", labelcolor="tab:blue")

    # ax2.plot(log_df["iter"], log_df["score_ref"], label=label_ref, color="tab:orange")
    # ax2.set_ylabel(f"{label_ref} (lower is better)", color="tab:orange")
    # ax2.tick_params(axis="y", labelcolor="tab:orange")

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
    plot_name = f"sa_{metric.__name__}_t0={t0}_cr={cr}_seed={seed}.png"
    os.makedirs(os.path.join(output_dir_sa, "plots"), exist_ok=True)
    fig.savefig(os.path.join(output_dir_sa, f"plots/{plot_name}"), bbox_inches="tight")
    plt.close(fig)


def run_sa(
    inputs, metric, t0=2.0, cr=0.98, mi=5_000, seed=42, sel_type="full", subset=None
):
    """Run a single SA instance with given parameters and return results dict."""
    df_daily = inputs["df_daily"]
    vars_to_match = inputs["vars_to_match"]
    NUM_CLUSTERS = inputs["NUM_CLUSTERS"]
    output_dir_sa = inputs["output_dir_sa"]
    YEAR_COL = inputs["YEAR_COL"]

    selected, best_score, log_df = simulated_annealing_selection(
        df=df_daily,
        variables=vars_to_match,
        scoring_fn=metric,
        n_select=NUM_CLUSTERS,
        model_col="model",
        year_col=YEAR_COL,
        temp_start=t0,
        cooling_rate=cr,
        max_iter=mi,
        random_state=seed,
        sel_type=sel_type,
        subset=subset,
    )

    sel_dict = {
        "model_years": tuple(tuple(my) for my in selected),
        "optimized_metric": metric.__name__,
        "best_score": best_score,
        "seed": seed,
        # "t0": t0,
        "cr": cr,
        # "mi": mi,
        "n_iter": int(log_df["iter"].iloc[-1]),
    }

    # plot logs of SA run and save the plot
    plot_sa_log(log_df, metric, t0, cr, mi, seed, output_dir_sa)

    return sel_dict
