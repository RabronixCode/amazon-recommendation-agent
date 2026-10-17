# data_loading.py (ENHANCED for thesis-ready EDA + plots + small sweeps)
from __future__ import annotations

import os
import math
import argparse
import itertools
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import pandas as pd

# Use non-interactive backend for servers / remote machines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from config import RAW_REVIEWS_PATH  # Putanja do Amazon JSONL.GZ.

try:
    from scipy.sparse import csr_matrix
except ImportError:
    csr_matrix = None

pd.set_option("display.max_rows", None)


# =========================
# 0) Učitavanje + mapiranje
# =========================

def load_raw_reviews() -> pd.DataFrame:
    """Učita Amazon reviews JSONL.GZ (lines=True)."""
    return pd.read_json(RAW_REVIEWS_PATH, lines=True)


def map_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizuje kolone na: user_id, item_id, rating, timestamp.
    Čisti: dropna + drop_duplicates.
    """
    column_mapping = {
        "reviewerID": "user_id",
        "asin": "item_id",
        "overall": "rating",
        "unixReviewTime": "timestamp",
    }
    missing = [c for c in column_mapping if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Nedostaju kolone: {missing}. Dostupne: {list(df_raw.columns)}")

    df = df_raw[list(column_mapping.keys())].copy()
    df.rename(columns=column_mapping, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["rating"] = df["rating"].astype(float)

    df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)
    df.dropna(subset=["user_id", "item_id", "rating", "timestamp"], inplace=True)

    # stabilno sortiranje po vremenu (za time-aware protokol)
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df


# ==========================================
# 1) Stabilno filtriranje sparsity (iterativno)
# ==========================================

@dataclass
class FilterTrajectoryRow:
    iter_no: int
    interactions: int
    users: int
    items: int
    density: float


def compute_dataset_profile(df: pd.DataFrame, label: str = "") -> dict[str, object]:
    n_interactions = int(len(df))
    n_users = int(df["user_id"].nunique())
    n_items = int(df["item_id"].nunique())
    density = float(n_interactions) / (float(n_users) * float(n_items)) if n_users and n_items else float("nan")
    return {
        "label": label,
        "interactions": n_interactions,
        "users": n_users,
        "items": n_items,
        "density": density,
    }


def iterative_filter_min_interactions_with_trajectory(
    df: pd.DataFrame,
    min_user_interactions: int = 3,
    min_item_interactions: int = 4,
    max_iters: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterativno filtrira dok se dataset ne stabilizuje.
    Vraća (df_filtered, traj_df) gde traj_df sadrži profil po iteracijama.
    """
    out = df.copy()
    traj = []

    def snapshot(iter_no: int, cur: pd.DataFrame):
        prof = compute_dataset_profile(cur)
        traj.append(FilterTrajectoryRow(
            iter_no=iter_no,
            interactions=int(prof["interactions"]),
            users=int(prof["users"]),
            items=int(prof["items"]),
            density=float(prof["density"]),
        ))

    snapshot(0, out)
    for it in range(1, max_iters + 1):
        before = len(out)
        out = out.groupby("user_id").filter(lambda x: len(x) >= min_user_interactions)
        out = out.groupby("item_id").filter(lambda x: len(x) >= min_item_interactions)
        after = len(out)
        snapshot(it, out)
        if after == before:
            break

    traj_df = pd.DataFrame([r.__dict__ for r in traj])
    return out, traj_df


# =========================
# 2) Time-aware LOO split
# =========================

def build_splits_leave_one_out(df_cut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-out po korisniku:
    - poslednja interakcija -> test
    - pretposlednja -> val
    - ostalo -> train
    """
    df_cut = df_cut.sort_values(["user_id", "timestamp"], ascending=[True, True]).copy()
    n = df_cut.groupby("user_id")["rating"].transform("size")
    r = df_cut.groupby("user_id").cumcount()
    df_cut["split"] = "train"
    df_cut.loc[r == (n - 1), "split"] = "test"
    df_cut.loc[r == (n - 2), "split"] = "val"
    df_train = df_cut[df_cut["split"] == "train"].reset_index(drop=True)
    df_val = df_cut[df_cut["split"] == "val"].reset_index(drop=True)
    df_test = df_cut[df_cut["split"] == "test"].reset_index(drop=True)
    return df_train, df_val, df_test


def add_reindex_columns(df_cut: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Kreira user_index i item_index (0..N-1) radi brzine i matrica.
    Vraća df + mape index->id (korisno u agentu / izveštajima).
    """
    df_cut = df_cut.copy()
    user_cats = df_cut["user_id"].astype("category")
    item_cats = df_cut["item_id"].astype("category")
    df_cut["user_index"] = user_cats.cat.codes.astype(int)
    df_cut["item_index"] = item_cats.cat.codes.astype(int)
    maps = {
        "user_index_to_id": list(user_cats.cat.categories),
        "item_index_to_id": list(item_cats.cat.categories),
    }
    return df_cut, maps


# =========================
# 3) Baselines + Bias model
# =========================

def fit_bias_model_by_index(
    df_train: pd.DataFrame,
    mu: float,
    n_users: int,
    n_items: int,
    reg: float = 10.0,
    n_iters: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)
    ug = df_train.groupby("user_index")
    ig = df_train.groupby("item_index")
    for _ in range(n_iters):
        for u, g in ug:
            items = g["item_index"].to_numpy(dtype=int)
            resid = g["rating"].to_numpy(dtype=float) - mu - bi[items]
            bu[int(u)] = float(resid.sum() / (reg + len(g)))
        for i, g in ig:
            users = g["user_index"].to_numpy(dtype=int)
            resid = g["rating"].to_numpy(dtype=float) - mu - bu[users]
            bi[int(i)] = float(resid.sum() / (reg + len(g)))
    return bu, bi


def pred_global_mean(df: pd.DataFrame, mu: float) -> np.ndarray:
    return np.full(len(df), mu, dtype=float)


def pred_user_mean(df: pd.DataFrame, user_mean_arr: np.ndarray) -> np.ndarray:
    return user_mean_arr[df["user_index"].to_numpy(dtype=int)].astype(float)


def pred_item_mean(df: pd.DataFrame, item_mean_arr: np.ndarray) -> np.ndarray:
    return item_mean_arr[df["item_index"].to_numpy(dtype=int)].astype(float)


def pred_bias(df: pd.DataFrame, mu: float, bu: np.ndarray, bi: np.ndarray) -> np.ndarray:
    u = df["user_index"].to_numpy(dtype=int)
    i = df["item_index"].to_numpy(dtype=int)
    return np.clip(mu + bu[u] + bi[i], 1.0, 5.0).astype(float)


def eval_rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# =========================
# 4) MF (Biased MF, SGD) + trening kriva
# =========================

def eval_mf_rmse(df: pd.DataFrame, mu: float, bu: np.ndarray, bi: np.ndarray, P: np.ndarray, Q: np.ndarray) -> float:
    u = df["user_index"].to_numpy(dtype=int)
    i = df["item_index"].to_numpy(dtype=int)
    r = df["rating"].to_numpy(dtype=float)
    preds = np.clip(mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1), 1.0, 5.0)
    return float(root_mean_squared_error(r, preds))


def train_biased_mf_sgd(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    n_users: int,
    n_items: int,
    factors: int = 32,
    lr: float = 0.02,
    reg: float = 0.05,
    n_epochs: int = 10,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mu = float(df_train["rating"].mean())
    P = (0.1 * rng.standard_normal((n_users, factors))).astype(np.float32)
    Q = (0.1 * rng.standard_normal((n_items, factors))).astype(np.float32)
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)

    tr_u = df_train["user_index"].to_numpy(dtype=int)
    tr_i = df_train["item_index"].to_numpy(dtype=int)
    tr_r = df_train["rating"].to_numpy(dtype=np.float32)
    order = np.arange(len(df_train), dtype=int)

    best_val = float("inf")
    best_state = None
    val_curve = []

    for epoch in range(1, n_epochs + 1):
        rng.shuffle(order)
        for idx in order:
            u = tr_u[idx]
            i = tr_i[idx]
            r_ui = tr_r[idx]
            pred = mu + bu[u] + bi[i] + float(np.dot(P[u], Q[i]))
            err = float(r_ui - pred)

            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

            Pu = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * Pu - reg * Q[i])

        val_rmse = eval_mf_rmse(df_val, mu, bu, bi, P, Q)
        val_curve.append({"epoch": epoch, "val_rmse": float(val_rmse)})
        print(f"[MF] Epoch {epoch}/{n_epochs}  Val RMSE={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = float(val_rmse)
            best_state = (mu, bu.copy(), bi.copy(), P.copy(), Q.copy())

    mu_b, bu_b, bi_b, P_b, Q_b = best_state
    return {
        "mu": mu_b,
        "bu": bu_b,
        "bi": bi_b,
        "P": P_b,
        "Q": Q_b,
        "best_val_rmse": float(best_val),
        "val_curve": pd.DataFrame(val_curve),
        "factors": int(factors),
        "lr": float(lr),
        "reg": float(reg),
        "n_epochs": int(n_epochs),
        "seed": int(seed),
    }


def mf_score_candidates(mu: float, bu: np.ndarray, bi: np.ndarray, P: np.ndarray, Q: np.ndarray, u_idx: int, cand_items: np.ndarray) -> np.ndarray:
    return (mu + bu[u_idx] + bi[cand_items] + (Q[cand_items] @ P[u_idx])).astype(float)


# =========================
# 5) Item-kNN (Adjusted Cosine) (sparse varijanta)
# =========================

def build_sparse_deviation_matrix(df_train: pd.DataFrame, n_users: int, n_items: int, user_mean_arr: np.ndarray) -> "csr_matrix":
    if csr_matrix is None:
        raise ImportError("scipy nije dostupan (potreban za sparse item-kNN).")
    rows = df_train["user_index"].to_numpy(dtype=int)
    cols = df_train["item_index"].to_numpy(dtype=int)
    vals = df_train["rating"].to_numpy(dtype=float) - user_mean_arr[rows]
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


def compute_item_topk_similarities(dev_mat: "csr_matrix", top_k: int = 50, block_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    item_user = dev_mat.T.tocsr()
    n_items = item_user.shape[0]
    neigh_idx = np.zeros((n_items, top_k), dtype=np.int32)
    neigh_sim = np.zeros((n_items, top_k), dtype=np.float32)

    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        sims_block = cosine_similarity(item_user[start:end], item_user)
        for r in range(sims_block.shape[0]):
            item_i = start + r
            sims_row = sims_block[r]
            sims_row[item_i] = -np.inf
            top = np.argpartition(-sims_row, top_k)[:top_k]
            top = top[np.argsort(-sims_row[top])]
            neigh_idx[item_i, :] = top.astype(np.int32)
            neigh_sim[item_i, :] = sims_row[top].astype(np.float32)

    return neigh_idx, neigh_sim


def build_user_history_ratings(df_train: pd.DataFrame) -> dict[int, dict[int, float]]:
    hist: dict[int, dict[int, float]] = {}
    for u, i, r in zip(
        df_train["user_index"].to_numpy(int),
        df_train["item_index"].to_numpy(int),
        df_train["rating"].to_numpy(float),
    ):
        hist.setdefault(int(u), {})[int(i)] = float(r)
    return hist


def predict_item_knn_one(
    u_idx: int,
    i_idx: int,
    user_hist: dict[int, dict[int, float]],
    user_mean_arr: np.ndarray,
    neigh_idx: np.ndarray,
    neigh_sim: np.ndarray,
    fallback: float,
) -> float:
    if u_idx not in user_hist:
        return float(fallback)
    rated = user_hist[u_idx]
    u_mean = float(user_mean_arr[u_idx])

    num = 0.0
    den = 0.0
    for j, s in zip(neigh_idx[i_idx], neigh_sim[i_idx]):
        if not np.isfinite(s):
            continue
        jj = int(j)
        if jj in rated:
            dev = float(rated[jj] - u_mean)
            num += float(s) * dev
            den += abs(float(s))

    if den == 0.0:
        return float(fallback)
    pred = u_mean + (num / den)
    return float(np.clip(pred, 1.0, 5.0))


# =========================
# 6) Ranking evaluacija: HR/NDCG/MRR uz negative sampling
# =========================

def sample_negatives_for_user(all_items: np.ndarray, interacted_set: set[int], positive_item: int, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.array([(i not in interacted_set) and (i != positive_item) for i in all_items], dtype=bool)
    pool = all_items[mask]
    if len(pool) <= n_neg:
        return pool.astype(int)
    return rng.choice(pool, size=n_neg, replace=False).astype(int)


def rank_of_positive_tie_aware(scores: np.ndarray, pos_index: int, rng: np.random.Generator) -> int:
    pos_score = float(scores[pos_index])
    higher = int(np.sum(scores > pos_score))
    ties = int(np.sum(scores == pos_score))
    tie_offset = int(rng.integers(low=1, high=ties + 1))
    return higher + tie_offset  # 1-based


def hr_at_k(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    return 0.0 if rank > k else 1.0 / math.log2(rank + 1)


def mrr_at_k(rank: int, k: int) -> float:
    return 0.0 if rank > k else 1.0 / float(rank)


def evaluate_ranking(
    df_pos: pd.DataFrame,
    user_interactions: dict[int, set[int]],
    all_items: np.ndarray,
    score_fn: Callable[[int, np.ndarray], np.ndarray],
    k: int = 10,
    n_neg: int = 99,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    hrs, ndcgs, mrrs = [], [], []
    for _, row in df_pos.iterrows():
        u = int(row["user_index"])
        pos_item = int(row["item_index"])
        interacted = user_interactions.get(u, set())
        neg_items = sample_negatives_for_user(all_items, interacted, pos_item, n_neg, rng)
        candidates = np.concatenate([[pos_item], neg_items]).astype(int)
        scores = score_fn(u, candidates)
        rnk = rank_of_positive_tie_aware(scores, pos_index=0, rng=rng)
        hrs.append(hr_at_k(rnk, k))
        ndcgs.append(ndcg_at_k(rnk, k))
        mrrs.append(mrr_at_k(rnk, k))
    return {f"HR@{k}": float(np.mean(hrs)), f"NDCG@{k}": float(np.mean(ndcgs)), f"MRR@{k}": float(np.mean(mrrs))}


# =========================
# 7) Plot helperi (THESIS FRIENDLY)
# =========================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_df(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def plot_hist_two(
    s1: pd.Series, s2: pd.Series,
    title: str, xlabel: str, ylabel: str,
    out_path: str,
    bins: int = 50,
    log_y: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(s1.dropna().to_numpy(), bins=bins, alpha=0.6, label="Before", edgecolor="black")
    plt.hist(s2.dropna().to_numpy(), bins=bins, alpha=0.6, label="After", edgecolor="black")
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cdf_two(
    x1: np.ndarray, x2: np.ndarray,
    title: str, xlabel: str,
    out_path: str,
    log_x: bool = False,
) -> None:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    def cdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / float(len(x))
        return x, y

    x1s, y1 = cdf(x1)
    x2s, y2 = cdf(x2)

    plt.figure(figsize=(10, 6))
    plt.plot(x1s, y1, label="Before")
    plt.plot(x2s, y2, label="After")
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini koeficijent za ne-negativne vrednosti (npr. broj interakcija po useru).
    """
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if len(x) == 0:
        return float("nan")
    if np.all(x == 0):
        return 0.0
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumx = np.cumsum(x_sorted)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


def plot_lorenz(x: np.ndarray, title: str, out_path: str) -> None:
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if len(x) == 0:
        return
    x_sorted = np.sort(x)
    cumx = np.cumsum(x_sorted)
    cumx = np.insert(cumx, 0, 0.0)
    cumx = cumx / cumx[-1] if cumx[-1] > 0 else cumx
    p = np.linspace(0, 1, len(cumx))

    plt.figure(figsize=(7, 7))
    plt.plot(p, cumx, label="Lorenz curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Equality line")
    plt.title(title)
    plt.xlabel("Kumulativni udeo entiteta")
    plt.ylabel("Kumulativni udeo interakcija")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_popularity_rank(counts: np.ndarray, title: str, out_path: str) -> None:
    counts = np.asarray(counts, dtype=float)
    counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(counts) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, counts)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Rank (log)")
    plt.ylabel("Broj interakcija (log)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_time_series(df: pd.DataFrame, out_path_prefix: str) -> None:
    """
    Plots:
    - interactions per month
    - average rating per month
    """
    d = df.copy()
    d["month"] = d["timestamp"].dt.to_period("M").dt.to_timestamp()

    agg = d.groupby("month").agg(
        interactions=("rating", "size"),
        avg_rating=("rating", "mean"),
    ).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(agg["month"], agg["interactions"])
    plt.title("Broj interakcija kroz vreme (mesečno)")
    plt.xlabel("Mesec")
    plt.ylabel("# interakcija")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_interactions_per_month.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(agg["month"], agg["avg_rating"])
    plt.title("Prosečna ocena kroz vreme (mesečno)")
    plt.xlabel("Mesec")
    plt.ylabel("Avg rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_avg_rating_per_month.png")
    plt.close()

    agg.to_csv(out_path_prefix + "_time_series_monthly.csv", index=False)


def plot_split_timestamp_distributions(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, out_path: str) -> None:
    """
    Histogram timestamp-a po splitu (da se vidi time-aware logika).
    """
    plt.figure(figsize=(12, 6))
    plt.hist(df_train["timestamp"].astype("int64") // 10**9, bins=60, alpha=0.6, label="train", edgecolor="black")
    plt.hist(df_val["timestamp"].astype("int64") // 10**9, bins=60, alpha=0.6, label="val", edgecolor="black")
    plt.hist(df_test["timestamp"].astype("int64") // 10**9, bins=60, alpha=0.6, label="test", edgecolor="black")
    plt.title("Distribucija vremena po splitu (train/val/test)")
    plt.xlabel("timestamp (unix seconds)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_filter_trajectory(traj_df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(12, 7))
    plt.plot(traj_df["iter_no"], traj_df["interactions"], label="interactions")
    plt.plot(traj_df["iter_no"], traj_df["users"], label="users")
    plt.plot(traj_df["iter_no"], traj_df["items"], label="items")
    plt.title("Filter trajectory: kako dataset ‘opada’ po iteracijama")
    plt.xlabel("iteracija")
    plt.ylabel("vrednost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(traj_df["iter_no"], traj_df["density"])
    plt.title("Filter trajectory: gustina (density) po iteracijama")
    plt.xlabel("iteracija")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_density.png"))
    plt.close()


def plot_bar_metrics(table: pd.DataFrame, metric_cols: list[str], title: str, out_path: str) -> None:
    """
    Jednostavan bar plot iz tabele (model x metric).
    """
    models = table["model"].astype(str).to_list()
    x = np.arange(len(models))
    width = 0.8 / max(1, len(metric_cols))

    plt.figure(figsize=(max(12, len(models) * 0.8), 6))
    for j, col in enumerate(metric_cols):
        vals = table[col].to_numpy(dtype=float)
        plt.bar(x + (j - (len(metric_cols)-1)/2)*width, vals, width=width, label=col, edgecolor="black")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Vrednost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_mf_curve(val_curve: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(val_curve["epoch"], val_curve["val_rmse"])
    plt.title("MF: Val RMSE po epohi")
    plt.xlabel("Epoch")
    plt.ylabel("Val RMSE")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_sparsity_heatmap_sample(df_cut: pd.DataFrame, out_path: str, n_users_s: int = 200, n_items_s: int = 200, seed: int = 42) -> None:
    """
    Napravi mali sample user×item matrice (0/1) da vizuelno pokaže sparsity.
    """
    rng = np.random.default_rng(seed)
    users = df_cut["user_index"].unique()
    items = df_cut["item_index"].unique()
    if len(users) == 0 or len(items) == 0:
        return
    users_s = rng.choice(users, size=min(n_users_s, len(users)), replace=False)
    items_s = rng.choice(items, size=min(n_items_s, len(items)), replace=False)

    sub = df_cut[df_cut["user_index"].isin(users_s) & df_cut["item_index"].isin(items_s)][["user_index", "item_index"]].copy()
    u_map = {u: idx for idx, u in enumerate(sorted(users_s))}
    i_map = {i: idx for idx, i in enumerate(sorted(items_s))}
    mat = np.zeros((len(u_map), len(i_map)), dtype=np.uint8)
    for u, i in zip(sub["user_index"].to_numpy(int), sub["item_index"].to_numpy(int)):
        mat[u_map[int(u)], i_map[int(i)]] = 1

    plt.figure(figsize=(8, 7))
    plt.imshow(mat, aspect="auto")
    plt.title("Sparsity sample heatmap (1=interakcija, 0=nema)")
    plt.xlabel("Item (sample)")
    plt.ylabel("User (sample)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# =========================
# 8) EDA pre/posle filtriranja (prošireno)
# =========================

def save_filtering_effect_report_extended(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    out_dir: str,
    traj_df: Optional[pd.DataFrame] = None,
) -> None:
    _ensure_dir(out_dir)

    table = pd.DataFrame([
        compute_dataset_profile(df_before, "before_filter"),
        compute_dataset_profile(df_after, "after_filter"),
    ])
    save_df(table, os.path.join(out_dir, "filtering_effect.csv"))
    print("\n[Filtering Effect] Pre/posle filtriranja:")
    print(table)

    # ratings distribution
    plot_hist_two(
        df_before["rating"], df_after["rating"],
        title="Raspodela ocena: pre vs posle filtriranja",
        xlabel="rating", ylabel="count",
        out_path=os.path.join(out_dir, "ratings_distribution_before_after.png"),
        bins=10, log_y=False,
    )

    # interactions per user/item
    u_before = df_before.groupby("user_id")["rating"].count()
    u_after = df_after.groupby("user_id")["rating"].count()
    plot_hist_two(
        u_before, u_after,
        title="Distribucija #ocena po korisniku: pre vs posle",
        xlabel="# ocena po korisniku", ylabel="# korisnika",
        out_path=os.path.join(out_dir, "ratings_per_user_before_after.png"),
        bins=60, log_y=True,
    )

    i_before = df_before.groupby("item_id")["rating"].count()
    i_after = df_after.groupby("item_id")["rating"].count()
    plot_hist_two(
        i_before, i_after,
        title="Distribucija #ocena po artiklu: pre vs posle",
        xlabel="# ocena po artiklu", ylabel="# artikala",
        out_path=os.path.join(out_dir, "ratings_per_item_before_after.png"),
        bins=60, log_y=True,
    )

    # CDF plots (korisno za diplomski)
    plot_cdf_two(
        u_before.to_numpy(), u_after.to_numpy(),
        title="CDF: #ocena po korisniku (pre vs posle)",
        xlabel="# ocena po korisniku",
        out_path=os.path.join(out_dir, "cdf_ratings_per_user_before_after.png"),
        log_x=True,
    )
    plot_cdf_two(
        i_before.to_numpy(), i_after.to_numpy(),
        title="CDF: #ocena po artiklu (pre vs posle)",
        xlabel="# ocena po artiklu",
        out_path=os.path.join(out_dir, "cdf_ratings_per_item_before_after.png"),
        log_x=True,
    )

    # popularity rank plots (head/tail)
    plot_popularity_rank(
        u_after.to_numpy(),
        title="User activity: rank-frequency (posle filtriranja)",
        out_path=os.path.join(out_dir, "user_activity_rank_frequency_after.png"),
    )
    plot_popularity_rank(
        i_after.to_numpy(),
        title="Item popularity: rank-frequency (posle filtriranja)",
        out_path=os.path.join(out_dir, "item_popularity_rank_frequency_after.png"),
    )

    # Gini + Lorenz
    g_user = gini_coefficient(u_after.to_numpy())
    g_item = gini_coefficient(i_after.to_numpy())
    pd.DataFrame([
        {"entity": "users", "gini": g_user},
        {"entity": "items", "gini": g_item},
    ]).to_csv(os.path.join(out_dir, "gini_after_filter.csv"), index=False)

    plot_lorenz(u_after.to_numpy(), f"Lorenz curve: user activity (Gini={g_user:.3f})", os.path.join(out_dir, "lorenz_users_after.png"))
    plot_lorenz(i_after.to_numpy(), f"Lorenz curve: item popularity (Gini={g_item:.3f})", os.path.join(out_dir, "lorenz_items_after.png"))

    # time series pre i posle (da se vidi da li filtriranje menja temporalnu sliku)
    plot_time_series(df_before, os.path.join(out_dir, "time_series_before"))
    plot_time_series(df_after, os.path.join(out_dir, "time_series_after"))

    # filter trajectory plot
    if traj_df is not None and len(traj_df) > 1:
        traj_df.to_csv(os.path.join(out_dir, "filter_trajectory.csv"), index=False)
        plot_filter_trajectory(traj_df, os.path.join(out_dir, "filter_trajectory.png"))


# =========================
# 9) Eksperiment: full pipeline + svi plotovi
# =========================

@dataclass
class ExperimentConfig:
    min_user: int = 3
    min_item: int = 4
    max_filter_iters: int = 20

    mf_factors: int = 32
    mf_lr: float = 0.02
    mf_reg: float = 0.05
    mf_epochs: int = 10
    seed: int = 42

    knn_topk: int = 50
    knn_block: int = 256
    hybrid_alpha: float = 0.7  # Hybrid Agent = alpha*item-kNN + (1-alpha)*bias

    ranking_k_list: tuple[int, ...] = (5, 10, 20)
    ranking_n_neg: int = 99

    out_dir: str = "eda_outputs"
    exp_name: str = "exp"


def run_single_experiment(df_clean: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, object]:
    """
    Jedan kompletan run:
    - filter + EDA plotovi + trajectory
    - reindex
    - LOO split
    - baselines + bias: RMSE/MAE (val/test) + bar plot
    - MF: trening + kriva + RMSE/MAE + bar plot update
    - ranking: HR/NDCG/MRR za više K + bar plot
    - sparsity heatmap sample
    """
    exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    _ensure_dir(exp_dir)

    # 1) Filter + trajectory
    df_before = df_clean.copy()
    df_cut, traj_df = iterative_filter_min_interactions_with_trajectory(
        df_clean,
        min_user_interactions=cfg.min_user,
        min_item_interactions=cfg.min_item,
        max_iters=cfg.max_filter_iters,
    )
    save_filtering_effect_report_extended(df_before, df_cut, out_dir=exp_dir, traj_df=traj_df)

    # 2) Reindex
    df_cut, idx_maps = add_reindex_columns(df_cut)
    n_users = int(df_cut["user_index"].nunique())
    n_items = int(df_cut["item_index"].nunique())

    # sparsity heatmap sample
    plot_sparsity_heatmap_sample(df_cut, os.path.join(exp_dir, "sparsity_heatmap_sample.png"))

    # 3) LOO split
    df_train, df_val, df_test = build_splits_leave_one_out(df_cut)
    assert df_val["user_index"].isin(df_train["user_index"]).all()
    assert df_test["user_index"].isin(df_train["user_index"]).all()

    plot_split_timestamp_distributions(df_train, df_val, df_test, os.path.join(exp_dir, "split_timestamp_distributions.png"))

    print(f"\n[Data:{cfg.exp_name}] interactions={len(df_cut)} users={n_users} items={n_items}")
    print(f"[Data:{cfg.exp_name}] train/val/test={len(df_train)}/{len(df_val)}/{len(df_test)}")

    # 4) Baselines + Bias
    mu = float(df_train["rating"].mean())

    user_mean_arr = np.full(n_users, mu, dtype=np.float32)
    item_mean_arr = np.full(n_items, mu, dtype=np.float32)
    um = df_train.groupby("user_index")["rating"].mean()
    im = df_train.groupby("item_index")["rating"].mean()
    user_mean_arr[um.index.to_numpy(int)] = um.to_numpy(np.float32)
    item_mean_arr[im.index.to_numpy(int)] = im.to_numpy(np.float32)

    bu, bi = fit_bias_model_by_index(
        df_train, mu=mu, n_users=n_users, n_items=n_items,
        reg=10.0, n_iters=20
    )

    rating_rows = []
    yv = df_val["rating"].to_numpy(float)
    yt = df_test["rating"].to_numpy(float)

    for name, pred_fn in [
        ("GlobalMean", lambda d: pred_global_mean(d, mu)),
        ("UserMean", lambda d: pred_user_mean(d, user_mean_arr)),
        ("ItemMean", lambda d: pred_item_mean(d, item_mean_arr)),
        ("Bias(μ+bu+bi)", lambda d: pred_bias(d, mu, bu, bi)),
    ]:
        pv = pred_fn(df_val)
        pt = pred_fn(df_test)
        mv = eval_rating_metrics(yv, pv)
        mt = eval_rating_metrics(yt, pt)
        rating_rows.append({
            "model": name,
            "val_RMSE": mv["rmse"], "val_MAE": mv["mae"],
            "test_RMSE": mt["rmse"], "test_MAE": mt["mae"],
        })

    # 5) MF
    mf = train_biased_mf_sgd(
        df_train, df_val,
        n_users=n_users, n_items=n_items,
        factors=cfg.mf_factors,
        lr=cfg.mf_lr,
        reg=cfg.mf_reg,
        n_epochs=cfg.mf_epochs,
        seed=cfg.seed,
    )
    plot_mf_curve(mf["val_curve"], os.path.join(exp_dir, "mf_val_curve.png"))
    mf["val_curve"].to_csv(os.path.join(exp_dir, "mf_val_curve.csv"), index=False)

    # MF rating metrics
    u_val = df_val["user_index"].to_numpy(int)
    i_val = df_val["item_index"].to_numpy(int)
    r_val = df_val["rating"].to_numpy(float)
    p_val = np.clip(mf["mu"] + mf["bu"][u_val] + mf["bi"][i_val] + np.sum(mf["P"][u_val] * mf["Q"][i_val], axis=1), 1.0, 5.0)
    mf_val_rmse = float(root_mean_squared_error(r_val, p_val))
    mf_val_mae = float(mean_absolute_error(r_val, p_val))

    u_test = df_test["user_index"].to_numpy(int)
    i_test = df_test["item_index"].to_numpy(int)
    r_test = df_test["rating"].to_numpy(float)
    p_test = np.clip(mf["mu"] + mf["bu"][u_test] + mf["bi"][i_test] + np.sum(mf["P"][u_test] * mf["Q"][i_test], axis=1), 1.0, 5.0)
    mf_test_rmse = float(root_mean_squared_error(r_test, p_test))
    mf_test_mae = float(mean_absolute_error(r_test, p_test))

    rating_rows.append({
        "model": f"MF(Biased SGD, k={cfg.mf_factors})",
        "val_RMSE": mf_val_rmse, "val_MAE": mf_val_mae,
        "test_RMSE": mf_test_rmse, "test_MAE": mf_test_mae,
    })

    rating_table = pd.DataFrame(rating_rows)
    rating_table.to_csv(os.path.join(exp_dir, "rating_metrics_table.csv"), index=False)
    print("\n[Rating Prediction] RMSE + MAE (VAL/TEST):")
    print(rating_table)

    plot_bar_metrics(
        rating_table,
        metric_cols=["val_RMSE", "test_RMSE"],
        title="RMSE (VAL/TEST) po modelu",
        out_path=os.path.join(exp_dir, "bar_rmse_val_test.png"),
    )
    plot_bar_metrics(
        rating_table,
        metric_cols=["val_MAE", "test_MAE"],
        title="MAE (VAL/TEST) po modelu",
        out_path=os.path.join(exp_dir, "bar_mae_val_test.png"),
    )

    # 6) Ranking prep
    all_items = np.arange(n_items, dtype=int)
    user_interactions: dict[int, set[int]] = {}
    for u, i in zip(df_cut["user_index"].to_numpy(int), df_cut["item_index"].to_numpy(int)):
        user_interactions.setdefault(int(u), set()).add(int(i))

    df_val_pos = df_val[["user_index", "item_index"]].copy()
    df_test_pos = df_test[["user_index", "item_index"]].copy()

    def score_global(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return np.full(len(cand), mu, dtype=float)

    def score_usermean(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return np.full(len(cand), float(user_mean_arr[u_idx]), dtype=float)

    def score_itemmean(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return item_mean_arr[cand].astype(float)

    def score_bias(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return (mu + bu[u_idx] + bi[cand]).astype(float)

    def score_mf(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], u_idx, cand)

    # Item-kNN optional
    item_knn_ready = False
    neigh_idx = neigh_sim = user_hist = None
    if csr_matrix is not None:
        dev_mat = build_sparse_deviation_matrix(df_train, n_users=n_users, n_items=n_items, user_mean_arr=user_mean_arr.astype(np.float32))
        neigh_idx, neigh_sim = compute_item_topk_similarities(dev_mat, top_k=cfg.knn_topk, block_size=cfg.knn_block)
        user_hist = build_user_history_ratings(df_train)
        item_knn_ready = True
    else:
        print("\n[CF] Item-kNN preskočen: scipy.sparse nije dostupan.")

    def score_itemknn(u_idx: int, cand: np.ndarray) -> np.ndarray:
        out = np.zeros(len(cand), dtype=float)
        for t, it in enumerate(cand):
            fb = float(mu + bu[u_idx] + bi[int(it)])
            out[t] = predict_item_knn_one(u_idx, int(it), user_hist, user_mean_arr, neigh_idx, neigh_sim, fb)
        return out

    def score_hybrid_agent(u_idx: int, cand: np.ndarray) -> np.ndarray:
        """
        Hybrid Agent scoring used by the recommendation agent:
        blend personalized item-kNN signal with robust bias baseline.
        """
        bias_scores = score_bias(u_idx, cand)
        if not item_knn_ready:
            return bias_scores
        knn_scores = score_itemknn(u_idx, cand)
        a = float(np.clip(cfg.hybrid_alpha, 0.0, 1.0))
        return (a * knn_scores + (1.0 - a) * bias_scores).astype(float)

    # 7) Ranking evaluacija za više K
    models_for_ranking = [
        ("GlobalMean", score_global),
        ("UserMean", score_usermean),
        ("ItemMean", score_itemmean),
        ("Bias(μ+bu+bi)", score_bias),
        (f"MF(Biased SGD, k={cfg.mf_factors})", score_mf),
    ]
    if item_knn_ready:
        models_for_ranking.append((f"Item-kNN(AdjCos, topK={cfg.knn_topk})", score_itemknn))
        models_for_ranking.append((f"Hybrid Agent(alpha={cfg.hybrid_alpha:.2f})", score_hybrid_agent))

    ranking_rows = []
    for model_name, score_fn in models_for_ranking:
        row = {"model": model_name}
        for k in cfg.ranking_k_list:
            val_m = evaluate_ranking(df_val_pos, user_interactions, all_items, score_fn, k=k, n_neg=cfg.ranking_n_neg, seed=cfg.seed)
            test_m = evaluate_ranking(df_test_pos, user_interactions, all_items, score_fn, k=k, n_neg=cfg.ranking_n_neg, seed=cfg.seed + 1)
            row[f"VAL_HR@{k}"] = val_m[f"HR@{k}"]
            row[f"VAL_NDCG@{k}"] = val_m[f"NDCG@{k}"]
            row[f"VAL_MRR@{k}"] = val_m[f"MRR@{k}"]
            row[f"TEST_HR@{k}"] = test_m[f"HR@{k}"]
            row[f"TEST_NDCG@{k}"] = test_m[f"NDCG@{k}"]
            row[f"TEST_MRR@{k}"] = test_m[f"MRR@{k}"]
        ranking_rows.append(row)

    ranking_table = pd.DataFrame(ranking_rows)
    ranking_table.to_csv(os.path.join(exp_dir, "ranking_metrics_table.csv"), index=False)

    print("\n[Top-K Ranking] (VAL/TEST) uz negative sampling:")
    print(ranking_table)

    # Bar plotovi za najčešći K=10 (ako postoji)
    if 10 in cfg.ranking_k_list:
        cols_hr = ["VAL_HR@10", "TEST_HR@10"]
        cols_ndcg = ["VAL_NDCG@10", "TEST_NDCG@10"]
        cols_mrr = ["VAL_MRR@10", "TEST_MRR@10"]
        plot_bar_metrics(ranking_table, cols_hr, "HR@10 (VAL/TEST) po modelu", os.path.join(exp_dir, "bar_hr10_val_test.png"))
        plot_bar_metrics(ranking_table, cols_ndcg, "NDCG@10 (VAL/TEST) po modelu", os.path.join(exp_dir, "bar_ndcg10_val_test.png"))
        plot_bar_metrics(ranking_table, cols_mrr, "MRR@10 (VAL/TEST) po modelu", os.path.join(exp_dir, "bar_mrr10_val_test.png"))

    # Final merge (rating + ranking)
    final_table = rating_table.merge(ranking_table, on="model", how="outer")
    final_table.to_csv(os.path.join(exp_dir, "final_models_comparison.csv"), index=False)
    print("\n[FINAL] Poređenje modela (rating + ranking):")
    print(final_table)

    # Snimi kratki “experiment summary”
    summary = {
        "exp_name": cfg.exp_name,
        "min_user": cfg.min_user,
        "min_item": cfg.min_item,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": int(len(df_cut)),
        "mf_factors": cfg.mf_factors,
        "mf_lr": cfg.mf_lr,
        "mf_reg": cfg.mf_reg,
        "mf_epochs": cfg.mf_epochs,
        "knn_topk": cfg.knn_topk,
        "hybrid_alpha": cfg.hybrid_alpha,
        "ranking_n_neg": cfg.ranking_n_neg,
        "ranking_k_list": list(cfg.ranking_k_list),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(exp_dir, "experiment_summary.csv"), index=False)

    return {
        "exp_dir": exp_dir,
        "summary": summary,
        "rating_table": rating_table,
        "ranking_table": ranking_table,
        "final_table": final_table,
        "df_cut": df_cut,
    }


# =========================
# 10) Sweep (min_user/min_item) + plot rezultata
# =========================

def run_filter_sweep(df_clean: pd.DataFrame, base_cfg: ExperimentConfig, min_user_list: list[int], min_item_list: list[int]) -> None:
    """
    Pokrene više eksperimenata kroz kombinacije min_user/min_item.
    Snimi:
    - sweep_summary.csv
    - plotove: dataset size vs thresholds + metrics vs thresholds (za odabrani model, default MF)
    """
    sweep_dir = os.path.join(base_cfg.out_dir, "SWEEP_filters")
    _ensure_dir(sweep_dir)

    rows = []
    for mu_thr, mi_thr in itertools.product(min_user_list, min_item_list):
        exp_name = f"minU{mu_thr}_minI{mi_thr}"
        cfg = ExperimentConfig(**{**base_cfg.__dict__, "min_user": mu_thr, "min_item": mi_thr, "exp_name": exp_name})
        print(f"\n===== RUN SWEEP: {exp_name} =====")
        res = run_single_experiment(df_clean, cfg)

        # izvuci ključne brojeve + jednu metriku koju želiš da pratiš
        final_table = res["final_table"].copy()
        mf_name = f"MF(Biased SGD, k={cfg.mf_factors})"
        mf_row = final_table[final_table["model"] == mf_name]
        # ako MF row ne postoji iz nekog razloga, uzmi best available
        if mf_row.empty:
            mf_row = final_table.head(1)

        row = {
            "exp_name": exp_name,
            "min_user": mu_thr,
            "min_item": mi_thr,
            "users": res["summary"]["n_users"],
            "items": res["summary"]["n_items"],
            "interactions": res["summary"]["n_interactions"],
        }
        # rating
        for c in ["val_RMSE", "test_RMSE", "val_MAE", "test_MAE"]:
            if c in mf_row.columns:
                row[f"MF_{c}"] = float(mf_row.iloc[0][c]) if pd.notna(mf_row.iloc[0][c]) else np.nan
        # ranking K=10 if exists
        for c in ["VAL_HR@10", "TEST_HR@10", "VAL_NDCG@10", "TEST_NDCG@10", "VAL_MRR@10", "TEST_MRR@10"]:
            if c in mf_row.columns:
                row[f"MF_{c}"] = float(mf_row.iloc[0][c]) if pd.notna(mf_row.iloc[0][c]) else np.nan

        rows.append(row)

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(os.path.join(sweep_dir, "sweep_summary.csv"), index=False)
    print("\n[SWEEP] Sačuvan: sweep_summary.csv")

    # Plot: interactions/users/items vs thresholds (heatmap-like via scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(sweep_df["min_user"], sweep_df["min_item"], s=sweep_df["interactions"] / max(1, sweep_df["interactions"].max()) * 500 + 30)
    plt.title("Sweep: veličina tačke ~ #interactions (posle filtriranja)")
    plt.xlabel("min_user")
    plt.ylabel("min_item")
    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, "sweep_interactions_scatter.png"))
    plt.close()

    # Plot: MF test_RMSE vs thresholds (ako postoji)
    if "MF_test_RMSE" in sweep_df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(sweep_df["min_user"], sweep_df["min_item"], c=sweep_df["MF_test_RMSE"])
        plt.title("Sweep: MF test_RMSE (boja) vs (min_user, min_item)")
        plt.xlabel("min_user")
        plt.ylabel("min_item")
        plt.tight_layout()
        plt.savefig(os.path.join(sweep_dir, "sweep_mf_test_rmse_color.png"))
        plt.close()


# =========================
# 11) Main
# =========================

def parse_int_list(s: str) -> list[int]:
    # "3,4,5" -> [3,4,5]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_k_list(s: str) -> tuple[int, ...]:
    return tuple(parse_int_list(s))


def main():
    parser = argparse.ArgumentParser(description="Enhanced data_loading.py: EDA + plots + models + sweep")
    parser.add_argument("--out_dir", type=str, default="eda_outputs", help="Output folder")
    parser.add_argument("--exp_name", type=str, default="exp_main", help="Experiment name subfolder")

    parser.add_argument("--min_user", type=int, default=3, help="Min interakcija po useru (filter)")
    parser.add_argument("--min_item", type=int, default=4, help="Min interakcija po itemu (filter)")
    parser.add_argument("--max_filter_iters", type=int, default=20, help="Max iteracije filtriranja")

    parser.add_argument("--mf_factors", type=int, default=32, help="MF latent dim")
    parser.add_argument("--mf_lr", type=float, default=0.02, help="MF learning rate")
    parser.add_argument("--mf_reg", type=float, default=0.05, help="MF L2 reg")
    parser.add_argument("--mf_epochs", type=int, default=10, help="MF epohe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--knn_topk", type=int, default=50, help="Item-kNN topK")
    parser.add_argument("--knn_block", type=int, default=256, help="Item-kNN block size")
    parser.add_argument("--hybrid_alpha", type=float, default=0.7, help="Hybrid Agent blend weight for item-kNN (0..1)")

    parser.add_argument("--ranking_ks", type=str, default="5,10,20", help="Lista K za ranking (npr 5,10,20)")
    parser.add_argument("--ranking_n_neg", type=int, default=99, help="Broj negativa")

    # Sweep mode
    parser.add_argument("--sweep_filters", action="store_true", help="Pokreni sweep kroz min_user/min_item")
    parser.add_argument("--sweep_min_user_list", type=str, default="2,3,4,5", help="npr 2,3,4,5")
    parser.add_argument("--sweep_min_item_list", type=str, default="2,3,4,5", help="npr 2,3,4,5")

    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    # Load + clean
    raw = load_raw_reviews()
    df_clean = map_and_clean(raw)

    base_cfg = ExperimentConfig(
        min_user=args.min_user,
        min_item=args.min_item,
        max_filter_iters=args.max_filter_iters,
        mf_factors=args.mf_factors,
        mf_lr=args.mf_lr,
        mf_reg=args.mf_reg,
        mf_epochs=args.mf_epochs,
        seed=args.seed,
        knn_topk=args.knn_topk,
        knn_block=args.knn_block,
        hybrid_alpha=float(np.clip(args.hybrid_alpha, 0.0, 1.0)),
        ranking_k_list=parse_k_list(args.ranking_ks),
        ranking_n_neg=args.ranking_n_neg,
        out_dir=args.out_dir,
        exp_name=args.exp_name,
    )

    if args.sweep_filters:
        mu_list = parse_int_list(args.sweep_min_user_list)
        mi_list = parse_int_list(args.sweep_min_item_list)
        run_filter_sweep(df_clean, base_cfg, mu_list, mi_list)
    else:
        run_single_experiment(df_clean, base_cfg)

    print("\nDONE.")


if __name__ == "__main__":
    main()
