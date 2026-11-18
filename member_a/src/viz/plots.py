"""Statistical EDA plotting utilities for PathSnap.

Generates publication-quality charts covering trajectory sizes, network dimensions,
sampling intervals, speeds, candidate distributions, recall, and feature correlations.
"""

from pathlib import Path
from typing import Optional
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from member_a.config.settings import PLOTS_DIR, RESULTS_DIR


def plot_trip_size_distributions(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of GPS observations, road nodes, and arcs per trip."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.set_theme(style="whitegrid")

    # 1. GPS observations
    sns.histplot(
        data=summary_df,
        x="num_gps",
        hue="partition",
        bins=25,
        kde=True,
        ax=axes[0],
        palette="tab10",
    )
    axes[0].set_title("GPS Observations per Trip", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Observation Count")
    axes[0].set_ylabel("Number of Trips")

    # 2. Road Network Nodes
    sns.histplot(
        data=summary_df,
        x="num_nodes",
        hue="partition",
        bins=25,
        kde=True,
        ax=axes[1],
        palette="tab10",
    )
    axes[1].set_title("Road Network Nodes per Trip", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Node Count")
    axes[1].set_ylabel("")

    # 3. Directed Road Arcs
    sns.histplot(
        data=summary_df,
        x="num_arcs",
        hue="partition",
        bins=25,
        kde=True,
        ax=axes[2],
        palette="tab10",
    )
    axes[2].set_title("Directed Road Arcs per Trip", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Arc Count")
    axes[2].set_ylabel("")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_temporal_and_speed_distributions(cand_features_df: pd.DataFrame, out_path: Path) -> None:
    """Plot distributions of GPS sampling intervals, step distances, and speeds."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.set_theme(style="whitegrid")

    # 1. Delta t
    dt_vals = cand_features_df["gps_delta_t_s"].clip(upper=10.0)
    sns.histplot(dt_vals, bins=20, ax=axes[0], color="steelblue", stat="density")
    axes[0].set_title("GPS Sampling Interval (clipped at 10s)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(r"$\Delta t$ (seconds)")
    axes[0].set_ylabel("Density")

    # 2. Step distance
    step_vals = cand_features_df["gps_step_distance_m"].clip(upper=60.0)
    sns.histplot(step_vals, bins=30, ax=axes[1], color="darkorange", kde=True)
    axes[1].set_title("GPS Step Displacement (m)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Displacement (metres)")
    axes[1].set_ylabel("Count")

    # 3. Speed
    speed_vals = cand_features_df["gps_speed_mps"].clip(upper=40.0)
    sns.histplot(speed_vals, bins=30, ax=axes[2], color="forestgreen", kde=True)
    axes[2].set_title("Instantaneous Speed (m/s)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Speed (m/s)")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_candidate_recall_and_density(cand_stats_df: pd.DataFrame, out_path: Path) -> None:
    """Plot candidate recall and candidate density across analyzed training trips."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    # 1. Candidate Recall Barplot
    trips = cand_stats_df["trip_id"].astype(str)
    recalls = cand_stats_df["recall"] * 100.0
    colors = ["crimson" if r < 50.0 else "mediumseagreen" for r in recalls]

    axes[0].bar(range(len(trips)), recalls, color=colors, edgecolor="black", alpha=0.85)
    axes[0].axhline(100.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_title("Candidate Segment Recall per Trip (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sampled Training Trip")
    axes[0].set_ylabel("Recall (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(range(len(trips)))
    axes[0].set_xticklabels([t[-4:] for t in trips], rotation=45)

    # 2. Candidate density distribution
    sns.boxplot(y=cand_stats_df["cands_per_pt"], ax=axes[1], color="mediumpurple", width=0.3)
    sns.stripplot(y=cand_stats_df["cands_per_pt"], ax=axes[1], color="black", size=6, jitter=0.2)
    axes[1].set_title("Candidate Count per GPS Point", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Candidates per Observation")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_distributions_and_labels(features_df: pd.DataFrame, out_path: Path) -> None:
    """Plot feature distributions conditioned on ground-truth label (positive vs negative)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.set_theme(style="whitegrid")

    has_label = "label" in features_df.columns
    hue_col = "label" if has_label else None

    # 1. Perpendicular Distance
    sns.histplot(
        data=features_df,
        x="distance_m",
        hue=hue_col,
        bins=30,
        kde=True,
        ax=axes[0, 0],
        palette="Set1",
        common_norm=False,
    )
    axes[0, 0].set_title("Perpendicular Distance to Road (m)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Distance (metres)")

    # 2. Angular difference
    sns.histplot(
        data=features_df,
        x="heading_bearing_diff_deg",
        hue=hue_col,
        bins=30,
        kde=True,
        ax=axes[0, 1],
        palette="Set1",
        common_norm=False,
    )
    axes[0, 1].set_title("Heading vs Bearing Angular Delta", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel(r"Angular Delta ($^\circ$)")

    # 3. Cosine similarity
    sns.histplot(
        data=features_df,
        x="heading_bearing_cos_sim",
        hue=hue_col,
        bins=30,
        kde=True,
        ax=axes[1, 0],
        palette="Set1",
        common_norm=False,
    )
    axes[1, 0].set_title("Heading vs Bearing Cosine Similarity", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Cosine Similarity")

    # 4. Relative distance to closest candidate
    sns.histplot(
        data=features_df,
        x="distance_to_closest_cand",
        hue=hue_col,
        bins=30,
        kde=True,
        ax=axes[1, 1],
        palette="Set1",
        common_norm=False,
    )
    axes[1, 1].set_title("Distance to Closest Candidate (m)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel(r"$\Delta$ Distance to Best (m)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_correlation_matrix(features_df: pd.DataFrame, out_path: Path) -> None:
    """Plot correlation heatmap across numerical features and target label."""
    num_cols = [
        "distance_m",
        "distance_to_closest_cand",
        "dist_rank",
        "cand_count_for_point",
        "projection_fraction",
        "arc_length_m",
        "heading_bearing_diff_deg",
        "heading_bearing_cos_sim",
        "gps_speed_mps",
        "u_in_degree",
        "v_out_degree",
        "has_reverse_twin",
    ]
    if "label" in features_df.columns:
        num_cols.append("label")

    corr = features_df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        cbar=True,
        square=True,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Feature & Target Pearson Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def generate_all_eda_plots(
    summary_csv_path: Optional[Path] = None,
    cand_stats_csv_path: Optional[Path] = None,
    features_df: Optional[pd.DataFrame] = None,
    out_dir: Optional[Path] = None,
) -> None:
    """Generate all five publication-quality EDA figures."""
    target_dir = out_dir if out_dir is not None else PLOTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    sum_path = summary_csv_path if summary_csv_path else RESULTS_DIR / "dataset_summary.csv"
    if sum_path.exists():
        df_sum = pd.read_csv(sum_path)
        plot_trip_size_distributions(df_sum, target_dir / "eda_01_trip_sizes.png")

    cand_path = cand_stats_csv_path if cand_stats_csv_path else RESULTS_DIR / "candidate_statistics.csv"
    if cand_path.exists():
        df_cands = pd.read_csv(cand_path)
        plot_candidate_recall_and_density(df_cands, target_dir / "eda_03_candidate_recall_and_density.png")

    if features_df is not None:
        plot_temporal_and_speed_distributions(features_df, target_dir / "eda_02_temporal_and_speed.png")
        plot_feature_distributions_and_labels(features_df, target_dir / "eda_04_feature_distributions.png")
        plot_feature_correlation_matrix(features_df, target_dir / "eda_05_feature_correlation_matrix.png")
