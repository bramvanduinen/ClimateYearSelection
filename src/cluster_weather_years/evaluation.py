import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cluster_weather_years.constants import PROBS


def ecdf_comparison(df_daily, datasets, columns):
    """Simple ECDF plots for visual inspection"""

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    for i, col in enumerate(columns):
        ax = axs.flat[i]
        for j, (name, df) in enumerate(datasets.items()):
            sns.ecdfplot(df, x=col, label=name, ax=ax)
        sns.ecdfplot(df_daily, x=col, label="Full climatology", ax=ax)
        ax.set_title(col)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")

    # Hide any unused axes (in case columns < 4)
    for j in range(len(columns), 4):
        axs.flat[j].axis("off")


def qq_plot(df_daily, datasets, columns):
    """
    Plot QQ curves and quantile differences for provided datasets vs full df.

    Parameters
    - df_daily: reference dataframe (full climatology)
    - datasets: dict[str, pd.DataFrame], e.g. {"Centroid": df_centroids, "Random": df_random}
    - columns: list[str]
    """
    fig_qq, axs_qq = plt.subplots(2, 2, figsize=(12, 10))
    fig_diff, axs_diff = plt.subplots(2, 2, figsize=(12, 10), sharey=False)

    for i, col in enumerate(columns):
        q_full = df_daily[col].quantile(PROBS)
        x = q_full.values

        # QQ plot
        axq = axs_qq.flat[i]
        for j, (name, df) in enumerate(datasets.items()):
            q_sub = df[col].quantile(PROBS)
            axq.plot(x, q_sub.values, label=name)
        lo, hi = np.nanmin(x), np.nanmax(x)
        axq.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        axq.set_title(f"QQ: {col}")
        axq.set_xlabel("Full climatology value")
        axq.set_ylabel("Subset value")
        axq.legend(loc="best")

        # Differences
        axd = axs_diff.flat[i]
        for j, (name, df) in enumerate(datasets.items()):
            q_sub = df[col].quantile(PROBS)
            diff = (q_sub - q_full).values
            axd.plot(PROBS, diff, color=f"C{j}", label=f"{name} - Full")
            axd.plot(PROBS, np.abs(diff), color=f"C{j}", ls="--")
            axd.axhline(np.mean(np.abs(diff)), color=f"C{j}")
        axd.axhline(0.0, color="k", lw=1, ls="--")
        axd.set_title(f"QQ-differences: {col}")
        axd.set_xlabel("Quantile")
        axd.set_ylabel("Δ value")
        axd.legend(loc="best")

    for j in range(len(columns), 4):
        axs_qq.flat[j].axis("off")
        axs_diff.flat[j].axis("off")

    fig_qq.tight_layout()
    fig_diff.tight_layout()

    return fig_qq, fig_diff


def qq_univariate_rmse(df_daily, datasets, columns):
    """
    Compute QQ-based metrics for one or many datasets vs the full df_daily.

    Parameters
    - df_daily: reference dataframe (full climatology)
    - datasets: dict[str, pd.DataFrame], e.g. {"Centroid": df_centroids, "Random": df_random}
    - columns: list[str] variables to analyze

    Returns
    - pd.DataFrame with rows per (dataset, variable) and columns [MAE, RMSE, MaxAbs]
    """

    rows = []
    for col in columns:
        q_full = df_daily[col].quantile(PROBS)
        for name, df in datasets.items():
            q_sub = df[col].quantile(PROBS)
            diff = (q_sub - q_full).values
            rows.append(
                {
                    "dataset": name,
                    "variable": col,
                    "RMSE": float(np.sqrt(np.average(diff**2))),
                }
            )
    return pd.DataFrame(rows)
