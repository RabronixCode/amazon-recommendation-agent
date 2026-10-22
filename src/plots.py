from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_rating_distribution(rating_counts: pd.Series, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(rating_counts.index.astype(str), rating_counts.values)
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of interactions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_rating_distribution.png"), dpi=200)
    plt.close()

def plot_cdf(ux: np.ndarray, uy: np.ndarray, ix: np.ndarray, iy: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(ux, uy, label="User activity CDF")
    plt.plot(ix, iy, label="Item activity CDF")
    plt.xscale("log")  # activity is usually heavy-tailed
    plt.title("User/Item activity distributions (CDF)")
    plt.xlabel("Number of interactions (log scale)")
    plt.ylabel("CDF (share of entities with <= x interactions)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_user_item_activity_cdf.png"), dpi=200)
    plt.close()

def plot_interaction_timeline(timeline: pd.Series, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(timeline.index, timeline.values)
    plt.title("Interactions over time (monthly)")
    plt.xlabel("Time")
    plt.ylabel("Number of interactions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_interactions_timeline_monthly.png"), dpi=200)
    plt.close()

def plot_histogram_user_activity(hist_u: pd.Series, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.hist(hist_u.to_numpy(), bins=50)
    plt.xlabel("|history(u)| (number of interactions per user in TRAIN)")
    plt.ylabel("Number of users")
    plt.title("User activity histogram (TRAIN)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "histogram_user_activity_train.png"), dpi=200)
    plt.close()

def plot_histogram_user_logx(hist_u: pd.Series, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.hist(hist_u.to_numpy(), bins=50)
    plt.xscale("log")
    plt.xlabel("|history(u)| (log scale, TRAIN)")
    plt.ylabel("Number of users")
    plt.title("User activity histogram (TRAIN), log x axis")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "histogram_user_activity_train_logx.png"), dpi=200)
    plt.close()

def plot_itemknn_similarity_hist(sims: np.ndarray, out_dir: str, top_k: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(sims, bins=60, edgecolor="black", alpha=0.85)
    plt.xlabel("item-item cosine similarity (from the top-K lists)")
    plt.ylabel("Number of values")
    plt.title(f"Item-kNN: distribution of item-item similarities (topK={top_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "item_knn_similarity_hist.png"), dpi=200)
    plt.close()

def plot_filtering_effects(df_before: pd.DataFrame, df_after: pd.DataFrame, out_dir: str) -> None:
    """Before/after histograms of ratings, ratings per user and ratings per item."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(df_before["rating"], bins=10, edgecolor="black", alpha=0.6, label="Before")
    plt.hist(df_after["rating"], bins=10, edgecolor="black", alpha=0.6, label="After")
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("Rating distribution: before vs after filtering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratings_distribution_before_after.png"), dpi=200)
    plt.close()

    u_before = df_before.groupby("user_id")["rating"].count()
    u_after = df_after.groupby("user_id")["rating"].count()
    plt.figure(figsize=(10, 6))
    plt.hist(u_before, bins=60, edgecolor="black", alpha=0.6, label="Before")
    plt.hist(u_after, bins=60, edgecolor="black", alpha=0.6, label="After")
    plt.yscale("log")
    plt.xlabel("Number of ratings per user")
    plt.ylabel("Number of users (log)")
    plt.title("Ratings per user: before vs after filtering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratings_per_user_before_after.png"), dpi=200)
    plt.close()

    i_before = df_before.groupby("item_id")["rating"].count()
    i_after = df_after.groupby("item_id")["rating"].count()
    plt.figure(figsize=(10, 6))
    plt.hist(i_before, bins=60, edgecolor="black", alpha=0.6, label="Before")
    plt.hist(i_after, bins=60, edgecolor="black", alpha=0.6, label="After")
    plt.yscale("log")
    plt.xlabel("Number of ratings per item")
    plt.ylabel("Number of items (log)")
    plt.title("Ratings per item: before vs after filtering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratings_per_item_before_after.png"), dpi=200)
    plt.close()


def plot_user_knn_similarity_diagnostics(
    neigh_u_sim: np.ndarray,
    out_dir: str,
    top_k: int = 50,
) -> None:
    """
    Plots describing user-user similarities:
      1) Histogram of all top-K similarities (flattened)
      2) Distribution of mean(top-K sim) per user (signal stability)
      3) Distribution of max(sim) per user (how close the nearest neighbor is)
    """
    os.makedirs(out_dir, exist_ok=True)

    sims = neigh_u_sim.copy().astype(float)
    sims = sims[np.isfinite(sims)]
    sims = sims.reshape(-1)

    # Histogram of the top-K similarity values.
    plt.figure(figsize=(10, 6))
    plt.hist(neigh_u_sim.astype(float).reshape(-1), bins=60, edgecolor="black", alpha=0.85)
    plt.xlabel("Cosine similarity (user-user) over top-K edges")
    plt.ylabel("Number of edges")
    plt.title(f"User-kNN: distribution of user-user similarities (topK={top_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "user_knn_topk_similarity_hist.png"), dpi=200)
    plt.close()

    # Mean similarity per user.
    mean_s = np.mean(neigh_u_sim.astype(float), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(mean_s, bins=60, edgecolor="black", alpha=0.85)
    plt.xlabel("Mean(top-K cosine similarity) per user")
    plt.ylabel("Number of users")
    plt.title(f"User-kNN: distribution of mean similarity per user (topK={top_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "user_knn_mean_similarity_per_user.png"), dpi=200)
    plt.close()

    # Max similarity per user.
    max_s = np.max(neigh_u_sim.astype(float), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(max_s, bins=60, edgecolor="black", alpha=0.85)
    plt.xlabel("Max cosine similarity per user (among top-K)")
    plt.ylabel("Number of users")
    plt.title(f"User-kNN: distribution of maximum similarity per user (topK={top_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "user_knn_max_similarity_per_user.png"), dpi=200)
    plt.close()


def plot_baseline_comparison_bars(
    rating_table: pd.DataFrame,
    out_dir: str,
    sort_by: str = "val_RMSE",
    only_models: list[str] | None = None,
) -> None:
    """
    Saves two bar charts as PNG in out_dir:
      1) RMSE per model (val and test)
      2) MAE per model (val and test)
    """
    os.makedirs(out_dir, exist_ok=True)

    dfp = rating_table.copy()

    # Pass only_models to restrict the chart to a subset of models.
    if only_models is not None:
        dfp = dfp[dfp["model"].isin(only_models)].copy()

    # Sort for readability (usually by val_RMSE).
    if sort_by in dfp.columns:
        dfp = dfp.sort_values(sort_by, ascending=True).reset_index(drop=True)

    models = dfp["model"].tolist()
    x = np.arange(len(models))
    width = 0.38

    # RMSE bar chart.
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, dfp["val_RMSE"].to_numpy(float), width, label="VAL RMSE")
    plt.bar(x + width/2, dfp["test_RMSE"].to_numpy(float), width, label="TEST RMSE")
    plt.xticks(x, models, rotation=25, ha="right")
    plt.ylabel("RMSE")
    plt.title("Model comparison: RMSE (VAL vs TEST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_comparison_rmse_bar.png"), dpi=200)
    plt.close()

    # MAE bar chart.
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, dfp["val_MAE"].to_numpy(float), width, label="VAL MAE")
    plt.bar(x + width/2, dfp["test_MAE"].to_numpy(float), width, label="TEST MAE")
    plt.xticks(x, models, rotation=25, ha="right")
    plt.ylabel("MAE")
    plt.title("Model comparison: MAE (VAL vs TEST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_comparison_mae_bar.png"), dpi=200)
    plt.close()


def plot_train_rating_histogram(
    df_train: pd.DataFrame,
    out_dir: str,
    bins: int = 10,
) -> None:
    """Histogram of TRAIN ratings (motivates the global mean mu as a default prediction)."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(df_train["rating"].to_numpy(float), bins=bins, edgecolor="black", alpha=0.8)
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("TRAIN: rating distribution (motivation for the global mean mu)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_rating_histogram.png"), dpi=200)
    plt.close()


def plot_train_interaction_distributions(
    df_train: pd.DataFrame,
    out_dir: str,
    min_user_interactions: int | None = None,
    min_item_interactions: int | None = None,
    bins: int = 60,
    log_y: bool = True,
) -> None:
    """
    Distribution of the number of TRAIN interactions per user and per item.
    Shows why user/item means only become stable after filtering.
    """
    os.makedirs(out_dir, exist_ok=True)

    user_counts = df_train.groupby("user_id")["rating"].size()
    item_counts = df_train.groupby("item_id")["rating"].size()

    # Users.
    plt.figure(figsize=(10, 6))
    plt.hist(user_counts.to_numpy(int), bins=bins, edgecolor="black", alpha=0.8)
    if log_y:
        plt.yscale("log")
    if min_user_interactions is not None:
        plt.axvline(min_user_interactions, linestyle="--", linewidth=2, label=f"min_user={min_user_interactions}")
        plt.legend()
    plt.xlabel("Number of interactions per user (in TRAIN)")
    plt.ylabel("Number of users" + (" (log)" if log_y else ""))
    plt.title("TRAIN: distribution of interactions per user")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_interactions_per_user.png"), dpi=200)
    plt.close()

    # Items.
    plt.figure(figsize=(10, 6))
    plt.hist(item_counts.to_numpy(int), bins=bins, edgecolor="black", alpha=0.8)
    if log_y:
        plt.yscale("log")
    if min_item_interactions is not None:
        plt.axvline(min_item_interactions, linestyle="--", linewidth=2, label=f"min_item={min_item_interactions}")
        plt.legend()
    plt.xlabel("Number of interactions per item (in TRAIN)")
    plt.ylabel("Number of items" + (" (log)" if log_y else ""))
    plt.title("TRAIN: distribution of interactions per item")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_interactions_per_item.png"), dpi=200)
    plt.close()


def plot_topn_bar_test_metric(
    ranking_table: pd.DataFrame,
    out_dir: str,
    metric_col: str,
    title: str,
    filename: str,
) -> None:
    """Bar chart of one Top-N metric per model (TEST_*)."""
    os.makedirs(out_dir, exist_ok=True)

    dfp = ranking_table[["model", metric_col]].dropna().copy()
    dfp = dfp.sort_values(metric_col, ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(dfp)), dfp[metric_col].to_numpy(float))
    plt.xticks(np.arange(len(dfp)), dfp["model"].tolist(), rotation=25, ha="right")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


def plot_positive_rank_histogram(
    ranks_a: np.ndarray,
    label_a: str,
    out_dir: str,
    filename: str,
    ranks_b: np.ndarray | None = None,
    label_b: str | None = None,
    max_rank: int = 100,
) -> None:
    """Histogram of the positive item's rank (1..100) for one or two models."""
    os.makedirs(out_dir, exist_ok=True)

    bins = np.arange(1, max_rank + 2)  # 1..101 gives 100 bins
    plt.figure(figsize=(12, 6))

    plt.hist(ranks_a, bins=bins, edgecolor="black", alpha=0.6, label=label_a)

    if ranks_b is not None and label_b is not None:
        plt.hist(ranks_b, bins=bins, edgecolor="black", alpha=0.6, label=label_b)

    plt.xlabel(f"Rank of the positive item (1 to {max_rank})")
    plt.ylabel("Number of cases")
    plt.title(f"Distribution of the positive item's rank (test: 1 positive + {max_rank - 1} negatives)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


def plot_negative_pool_histogram(
    pool_sizes: np.ndarray,
    out_dir: str,
    filename: str,
) -> None:
    """Histogram of the negative pool size per user (per positive example)."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.hist(pool_sizes, bins=60, edgecolor="black", alpha=0.85)
    plt.xlabel("|pool| = number of available negatives (after removing already-interacted items)")
    plt.ylabel("Number of cases")
    plt.title("Negative pool size per user (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


def plot_tie_rate_bar(
    tie_table: pd.DataFrame,
    out_dir: str,
    filename: str,
    title: str,
) -> None:
    """Bar chart of the tie rate per model."""
    os.makedirs(out_dir, exist_ok=True)

    dfp = tie_table.sort_values("tie_rate", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(dfp)), dfp["tie_rate"].to_numpy(float))
    plt.xticks(np.arange(len(dfp)), dfp["model"].tolist(), rotation=25, ha="right")
    plt.ylabel("tie_rate (ties > 1)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


def plot_hybrid_alpha_curve(
    sweep_df: pd.DataFrame,
    out_dir: str,
    k: int = 10,
    filename: str = "hybrid_alpha_curve.png",
    title: str | None = None,
) -> None:
    """Line plot of HR/NDCG/MRR vs alpha. Input: DataFrame with columns (alpha, VAL_HR@k, VAL_NDCG@k, VAL_MRR@k)."""
    if sweep_df is None or len(sweep_df) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    sub = sweep_df.sort_values("alpha")
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["alpha"], sub[f"VAL_NDCG@{k}"], "o-", label=f"NDCG@{k}", color="#1f77b4")
    ax.plot(sub["alpha"], sub[f"VAL_HR@{k}"], "s-", label=f"HR@{k}", color="#2ca02c")
    ax.plot(sub["alpha"], sub[f"VAL_MRR@{k}"], "^-", label=f"MRR@{k}", color="#d62728")
    ax.set_xlabel("α (Hybrid blend; α=1 → pure MF, α=0 → pure Item-kNN)")
    ax.set_ylabel("Validation metric")
    ax.set_title(title or f"Hybrid α sweep (val, top-{k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


def plot_hybrid_grid_heatmap(
    df_grid: pd.DataFrame,
    out_dir: str,
    k: int = 10,
    filename: str = "hybrid_grid_heatmap_ndcg.png",
) -> None:
    """Heatmap of max VAL_NDCG@k over (knn_topk x mf_factors), aggregated over alpha and reg."""
    if df_grid is None or len(df_grid) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    pivot = df_grid.pivot_table(
        index="knn_topk", columns="mf_factors",
        values=f"VAL_NDCG@{k}", aggfunc="max",
    )
    _, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("MF factors")
    ax.set_ylabel("Item-kNN top_k")
    ax.set_title(f"Hybrid grid: max VAL_NDCG@{k} per cell (best α/reg)")
    mid = float(np.nanmean(pivot.values))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    color="white" if v < mid else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label=f"VAL_NDCG@{k}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


def plot_hybrid_pred_distribution(
    mf_pred: np.ndarray,
    knn_pred: np.ndarray,
    hybrid_pred: np.ndarray,
    out_dir: str,
    filename: str = "hybrid_prediction_distribution_test.png",
    title: str = "Prediction distribution per model (TEST)",
) -> None:
    """Overlaid histograms of MF / Item-kNN / Hybrid predictions on the test set."""
    os.makedirs(out_dir, exist_ok=True)
    _, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(1.0, 5.0, 41)
    ax.hist(np.asarray(mf_pred, dtype=float), bins=bins, alpha=0.5, label="MF", density=True)
    ax.hist(np.asarray(knn_pred, dtype=float), bins=bins, alpha=0.5, label="Item-kNN", density=True)
    ax.hist(np.asarray(hybrid_pred, dtype=float), bins=bins, alpha=0.5, label="Hybrid", density=True)
    ax.set_xlabel("Predicted rating")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()
