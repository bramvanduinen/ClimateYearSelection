"""K-Medoids clustering for weather year selection."""

# %%
import matplotlib.pyplot as plt
import pandas as pd

# probably temporary
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

# from cluster_weather_years.functions import build_df, plot_clim_per_cluster
# from cluster_weather_years.constants import YEAR_COL


# %%
def k_medoids_clustering(
    df: pd.DataFrame,
    n_clusters: int,
    vars_to_match: list[str],
    output_dir: str,
    year_col="Hydrological_year",
    random_state: int = 42,
):
    """Perform K-Medoids clustering on the provided DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_clusters (int): Number of weather years to select (clusters for K-Medoids).
        vars_to_match (list[str]): List of variable names to consider for clustering.
        output_dir (str): Directory to save output files.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
    """

    # Standardize the data
    df_raw = df[[year_col] + vars_to_match].copy()
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(
        scaler.fit_transform(df_raw[vars_to_match]),
        columns=vars_to_match,
        index=df_raw.index,
    )
    df_standardized[year_col] = df_raw[year_col]

    # Compute yearly statistics of vars_to_match (mean and variance)
    yearly_stats = df_standardized.groupby([pd.Grouper(level="model"), year_col]).agg(
        ["mean", "var"]
    )
    yearly_stats.columns = [f"{var}_{stat}" for var, stat in yearly_stats.columns]
    yearly_stats = yearly_stats.reset_index()

    stats_scaled = scaler.fit_transform(yearly_stats.drop(columns=[year_col, "model"]))

    # Apply K-Medoids clustering
    kmed = KMedoids(n_clusters=n_clusters, random_state=random_state).fit(stats_scaled)
    yearly_stats = yearly_stats.assign(cluster=kmed.labels_)

    # Extract medoid years
    medoid_idx = kmed.medoid_indices_
    medoid_years = yearly_stats.iloc[medoid_idx].copy()

    medoid_years_to_save = medoid_years.copy()[
        ["model", year_col, "cluster"]
    ]  # only save relevant columns
    medoid_years_to_save.to_csv(
        f"{output_dir}/kmedoids_year_selection.csv", index="cluster"
    )

    analyze_kmedoids(kmed, stats_scaled, medoid_idx, medoid_years, output_dir)
    print(f"K-Medoids clustering completed. Results saved to {output_dir}.")
    return None


def analyze_kmedoids(kmed, stats_scaled, medoid_idx, medoid_years, output_dir):
    """Analyze K-Medoids clustering results using PCA for dimensionality reduction."""
    pca = PCA(n_components=2)
    stats_pca = pca.fit_transform(stats_scaled)
    medoid_pca = stats_pca[medoid_idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        stats_pca[:, 0], stats_pca[:, 1], c=kmed.labels_, cmap="viridis", alpha=0.5
    )
    plt.scatter(
        medoid_pca[:, 0],
        medoid_pca[:, 1],
        c=kmed.labels_[medoid_idx],
        cmap="viridis",
        marker="X",
        s=200,
        label="Medoids",
    )
    plt.title("K-Medoids Clustering of Weather Years (PCA-reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(f"{output_dir}/kmedoids_pca_plot.png")

    # Print PCA loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # shape (n_features, n_components)
        index=medoid_years.columns[
            2:-1
        ],  # exclude 'model', 'Hydrological_year', 'cluster'
        columns=["PC1", "PC2"],
    ).round(3)
    # print("PCA Loadings:\n", loadings)
    # print("\nExplained Variance Ratio:", pca.explained_variance_ratio_.round(3))
    loadings.to_csv(f"{output_dir}/kmedoids_pca_loadings.csv")


# %%
