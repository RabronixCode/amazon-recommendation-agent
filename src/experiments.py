"""Full experiment pipeline: EDA, baselines, Bias, MF, Item-kNN, User-kNN and Hybrid.

Figures are written to docs/figures and tables to docs/tables (see config.py).

Usage (from the src folder):
    python experiments.py                    # full pipeline
    python experiments.py --mf-grid          # also run the MF hyperparameter grid and use its winner
    python experiments.py --no-hybrid-grid   # skip the (slow) Hybrid grid search
    python experiments.py --threshold-sweep  # filtering-threshold sensitivity study instead of the pipeline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

import dataprep
import evaluation
import models
import plots
from config import DATA_ERRORS, FIGURES_DIR, MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS, TABLES_DIR

K = 10  # cut-off of the ranking metrics
N_NEG = 99  # sampled negatives per positive item
SEED_VAL = 42  # negative-sampling seed for the validation split
SEED_TEST = 43  # negative-sampling seed for the test split
TOPK_ITEMKNN = 50  # neighbors kept per item
TOPK_USERKNN = 50  # neighbors kept per user
BIAS_LABEL = "Bias(μ+bu+bi)"

# MF hyperparameters used by the pipeline. They are the winner of run_mf_grid_search
# (lowest validation RMSE, see docs/tables/mf_grid_results.csv).
MF_PARAMS = {"factors": 64, "lr": 0.02, "reg": 0.02, "n_epochs": 20, "seed": 42}

# MF hyperparameters used by the filtering-threshold sweep.
MF_SWEEP_PARAMS = {"factors": 32, "lr": 0.02, "reg": 0.02, "n_epochs": 20, "seed": 42}


def mf_label_for(factors: int) -> str:
    return f"MF(Biased SGD, k={factors})"


def save_filtering_effect_report(
    df_before: pd.DataFrame, df_after: pd.DataFrame, fig_dir: str, table_dir: str
) -> None:
    """Save the before/after filtering table and the before/after histograms."""
    os.makedirs(table_dir, exist_ok=True)
    table = dataprep.make_filtering_report_table(df_before, df_after)
    table.to_csv(os.path.join(table_dir, "filtering_effect.csv"), index=False)
    print("\n[Filtering Effect] Before/after filtering:")
    print(table)
    plots.plot_filtering_effects(df_before, df_after, fig_dir)


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF of x (NaNs removed)."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    x_sorted = np.sort(x)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, y


def run_mf_grid_search(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    n_users: int,
    n_items: int,
    table_dir: str,
    factors_list=(16, 32, 64),
    lr_list=(0.01, 0.02),
    reg_list=(0.02, 0.05, 0.1),
    n_epochs_list=(10, 20),
    seeds=(42,),
) -> tuple[dict[str, object], pd.DataFrame]:
    """Try several MF configurations and pick the one with the lowest best_val_rmse.

    Saves mf_grid_results.csv. Returns (best_mf_model_dict, results_table sorted by validation RMSE).
    """
    os.makedirs(table_dir, exist_ok=True)

    rows = []
    best_mf = None
    best_cfg = None
    best_val = float("inf")

    exp_id = 0
    for factors, lr, reg, n_epochs, seed in product(factors_list, lr_list, reg_list, n_epochs_list, seeds):
        exp_id += 1
        print(f"\n[MF-GRID] Exp {exp_id} | factors={factors} lr={lr} reg={reg} epochs={n_epochs} seed={seed}")

        mf = models.train_biased_mf_sgd(
            df_train,
            df_val,
            n_users=n_users,
            n_items=n_items,
            factors=int(factors),
            lr=float(lr),
            reg=float(reg),
            n_epochs=int(n_epochs),
            seed=int(seed),
        )

        row = {
            "exp_id": exp_id,
            "factors": int(factors),
            "lr": float(lr),
            "reg": float(reg),
            "n_epochs": int(n_epochs),
            "seed": int(seed),
            "best_val_rmse": float(mf["best_val_rmse"]),
        }
        rows.append(row)

        if row["best_val_rmse"] < best_val:
            best_val = row["best_val_rmse"]
            best_mf = mf
            best_cfg = row

    results = pd.DataFrame(rows).sort_values("best_val_rmse", ascending=True).reset_index(drop=True)
    results.to_csv(os.path.join(table_dir, "mf_grid_results.csv"), index=False)

    print("\n[MF-GRID] Top 5 configurations by best_val_rmse:")
    print(results.head(5))

    print("\n[MF-GRID] BEST config:")
    print(best_cfg)

    return best_mf, results


def user_knn_coverage_stats(
    df: pd.DataFrame,
    user_hist: dict[int, dict[int, float]],
    neigh_u_idx: np.ndarray,
    neigh_u_sim: np.ndarray,
) -> dict[str, float]:
    """How often User-kNN has at least one neighbor who rated the target item (otherwise it falls back to Bias)."""
    u_arr = df["user_index"].to_numpy(int)
    i_arr = df["item_index"].to_numpy(int)

    covered = 0
    total = len(df)

    for t in range(total):
        u = int(u_arr[t])
        i = int(i_arr[t])
        hit = False
        for v, s in zip(neigh_u_idx[u], neigh_u_sim[u]):
            if not np.isfinite(s):
                continue
            vv = int(v)
            if vv in user_hist and int(i) in user_hist[vv]:
                hit = True
                break
        if hit:
            covered += 1

    cov = float(covered) / float(total) if total else 0.0
    return {"total": float(total), "covered": float(covered), "coverage_rate": float(cov)}


def collect_positive_ranks_and_ties(
    df_pos: pd.DataFrame,
    user_interactions: dict[int, set[int]],
    all_items: np.ndarray,
    score_fn,
    n_neg: int = N_NEG,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ranks, ties) of the positive item.

    ranks: rank of the positive item in [1, 1 + n_neg]
    ties:  number of candidates with the same score as the positive item (including itself)
    """
    rng = np.random.default_rng(seed)
    ranks = np.zeros(len(df_pos), dtype=np.int32)
    ties_arr = np.zeros(len(df_pos), dtype=np.int32)

    for t, (_, row) in enumerate(df_pos.iterrows()):
        u = int(row["user_index"])
        pos_item = int(row["item_index"])
        interacted = user_interactions.get(u, set())

        neg_items = evaluation.sample_negatives_for_user(all_items, interacted, pos_item, n_neg, rng)
        candidates = np.concatenate([[pos_item], neg_items]).astype(int)

        scores = score_fn(u, candidates)
        pos_score = float(scores[0])
        ties_arr[t] = int(np.sum(scores == pos_score))

        # Tie-aware rank, same as in the evaluation.
        ranks[t] = int(evaluation.rank_of_positive_tie_aware(scores, pos_index=0, rng=rng))

    return ranks, ties_arr


def compute_negative_pool_sizes(
    df_pos: pd.DataFrame,
    user_interactions: dict[int, set[int]],
    n_items: int,
) -> np.ndarray:
    """Number of items left to sample negatives from, per positive example.

    user_interactions normally already contains the positive item, so
    pool = n_items - |interacted|. If the positive is not in the set, one more item is excluded.
    """
    pool_sizes = np.zeros(len(df_pos), dtype=np.int32)
    for t, (_, row) in enumerate(df_pos.iterrows()):
        u = int(row["user_index"])
        pos_item = int(row["item_index"])
        interacted = user_interactions.get(u, set())
        if pos_item in interacted:
            pool_sizes[t] = int(n_items - len(interacted))
        else:
            pool_sizes[t] = int(n_items - len(interacted) - 1)
    return pool_sizes


def compute_tie_rate_table(
    df_pos: pd.DataFrame,
    user_interactions: dict[int, set[int]],
    all_items: np.ndarray,
    models_fn: list[tuple[str, object]],
    n_neg: int = N_NEG,
    seed: int = 42,
) -> pd.DataFrame:
    """Tie rate: share of cases where the positive score is shared with another candidate (ties > 1).

    Returns a table with columns model, tie_rate, avg_ties.
    """
    rows = []
    for name, score_fn in models_fn:
        _, ties = collect_positive_ranks_and_ties(
            df_pos=df_pos,
            user_interactions=user_interactions,
            all_items=all_items,
            score_fn=score_fn,
            n_neg=n_neg,
            seed=seed,
        )
        rows.append({"model": name, "tie_rate": float(np.mean(ties > 1)), "avg_ties": float(np.mean(ties))})
    return pd.DataFrame(rows)


def make_baseline_score_fns(
    mu: float,
    user_mean_arr: np.ndarray,
    item_mean_arr: np.ndarray,
    bu: np.ndarray,
    bi: np.ndarray,
) -> list[tuple[str, object]]:
    """Ranking score functions (user, candidates) -> scores of the baselines and the Bias model."""

    def score_global(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return np.full(len(cand), mu, dtype=float)

    def score_usermean(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return np.full(len(cand), float(user_mean_arr[u_idx]), dtype=float)

    def score_itemmean(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return item_mean_arr[cand].astype(float)

    def score_bias(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return (mu + bu[u_idx] + bi[cand]).astype(float)

    return [
        ("GlobalMean", score_global),
        ("UserMean", score_usermean),
        ("ItemMean", score_itemmean),
        (BIAS_LABEL, score_bias),
    ]


def baseline_rating_rows(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    mu: float,
    user_mean_arr: np.ndarray,
    item_mean_arr: np.ndarray,
    bu: np.ndarray,
    bi: np.ndarray,
) -> list[dict[str, object]]:
    """RMSE/MAE rows (validation and test) of GlobalMean, UserMean, ItemMean and Bias."""
    yv = df_val["rating"].to_numpy(float)
    yt = df_test["rating"].to_numpy(float)
    rows = []
    for name, pred_fn in [
        ("GlobalMean", lambda d: models.pred_global_mean(d, mu)),
        ("UserMean", lambda d: models.pred_user_mean(d, user_mean_arr)),
        ("ItemMean", lambda d: models.pred_item_mean(d, item_mean_arr)),
        (BIAS_LABEL, lambda d: models.pred_bias(d, mu, bu, bi)),
    ]:
        mv = evaluation.eval_rating_metrics(yv, pred_fn(df_val))
        mt = evaluation.eval_rating_metrics(yt, pred_fn(df_test))
        rows.append({"model": name, "val_RMSE": mv["rmse"], "val_MAE": mv["mae"], "test_RMSE": mt["rmse"], "test_MAE": mt["mae"]})
    return rows


def mf_rating_row(mf: dict[str, object], label: str, df_val: pd.DataFrame, df_test: pd.DataFrame) -> dict[str, object]:
    """RMSE/MAE row of a trained MF model."""
    args = (mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"])
    mv = evaluation.eval_rating_metrics(df_val["rating"].to_numpy(float), models.mf_predict_ratings(df_val, *args))
    mt = evaluation.eval_rating_metrics(df_test["rating"].to_numpy(float), models.mf_predict_ratings(df_test, *args))
    return {"model": label, "val_RMSE": mv["rmse"], "val_MAE": mv["mae"], "test_RMSE": mt["rmse"], "test_MAE": mt["mae"]}


def ranking_row(
    model_name: str,
    score_fn,
    df_val_pos: pd.DataFrame,
    df_test_pos: pd.DataFrame,
    user_interactions: dict[int, set[int]],
    all_items: np.ndarray,
) -> dict[str, object]:
    """HR/NDCG/MRR@10 on validation and test for one model (one positive + 99 negatives per user)."""
    val_m = evaluation.evaluate_ranking(df_val_pos, user_interactions, all_items, score_fn, k=K, n_neg=N_NEG, seed=SEED_VAL)
    test_m = evaluation.evaluate_ranking(df_test_pos, user_interactions, all_items, score_fn, k=K, n_neg=N_NEG, seed=SEED_TEST)
    return {
        "model": model_name,
        f"VAL_HR@{K}": val_m[f"HR@{K}"],
        f"VAL_NDCG@{K}": val_m[f"NDCG@{K}"],
        f"VAL_MRR@{K}": val_m[f"MRR@{K}"],
        f"TEST_HR@{K}": test_m[f"HR@{K}"],
        f"TEST_NDCG@{K}": test_m[f"NDCG@{K}"],
        f"TEST_MRR@{K}": test_m[f"MRR@{K}"],
    }


def blend_metrics(
    cached: list[tuple[np.ndarray, np.ndarray]], alpha: float, rank_seed: int
) -> dict[str, float]:
    """VAL HR/NDCG/MRR of alpha * MF + (1 - alpha) * Item-kNN, using cached (MF, kNN) score vectors.

    Each score vector has the positive item at index 0.
    """
    rank_rng = np.random.default_rng(rank_seed)
    ranks = [
        evaluation.rank_of_positive_tie_aware(float(alpha) * mf_s + (1.0 - float(alpha)) * knn_s, 0, rank_rng)
        for mf_s, knn_s in cached
    ]
    return {
        f"VAL_HR@{K}": float(np.mean([evaluation.hr_at_k(r, K) for r in ranks])),
        f"VAL_NDCG@{K}": float(np.mean([evaluation.ndcg_at_k(r, K) for r in ranks])),
        f"VAL_MRR@{K}": float(np.mean([evaluation.mrr_at_k(r, K) for r in ranks])),
    }


def topn_summary(ranking_table: pd.DataFrame, metric_col: str, mf_label: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """Top-3 models by a TEST metric plus the MF minus Bias difference.

    Returns (top3, summary_table, mf_value, bias_value, mf_minus_bias).
    """
    top3 = (
        ranking_table[["model", metric_col]]
        .dropna()
        .sort_values(metric_col, ascending=False)
        .head(3)
        .reset_index(drop=True)
    )

    mf_row = ranking_table[ranking_table["model"] == mf_label]
    bias_row = ranking_table[ranking_table["model"] == BIAS_LABEL]
    mf_val = float(mf_row[metric_col].iloc[0]) if len(mf_row) else float("nan")
    bias_val = float(bias_row[metric_col].iloc[0]) if len(bias_row) else float("nan")
    diff = float(mf_val - bias_val) if np.isfinite(mf_val) and np.isfinite(bias_val) else float("nan")

    names, values = [], []
    for rank in range(3):
        names += [f"top{rank + 1}_model", f"top{rank + 1}_{metric_col}"]
        values += [
            str(top3.loc[rank, "model"]) if len(top3) > rank else "",
            float(top3.loc[rank, metric_col]) if len(top3) > rank else float("nan"),
        ]
    names += [f"MF_{metric_col}", f"Bias_{metric_col}", "MF_minus_Bias"]
    values += [mf_val, bias_val, diff]
    return top3, pd.DataFrame({"metric": names, "value": values}), mf_val, bias_val, diff


def run_one_setting(
    df_sorted: pd.DataFrame,
    df_before: pd.DataFrame,
    fig_root: str,
    table_root: str,
    min_u: int,
    min_i: int,
    max_iters: int = 20,
) -> dict[str, object]:
    """Run baselines, Bias and MF for one pair of filtering thresholds.

    Writes to <fig_root>/minU{min_u}_minI{min_i}/ the before/after figures and to
    <table_root>/minU{min_u}_minI{min_i}/ filtering_effect.csv, rating_metrics_table.csv and
    ranking_metrics_table_fixed.csv. MF uses MF_SWEEP_PARAMS, kNN and Hybrid are not evaluated.
    Raises AssertionError when a validation or test user has no training interactions
    (this happens for min_u = 2). Returns a one-row summary dict.
    """
    setting_tag = f"minU{min_u}_minI{min_i}"
    fig_dir = os.path.join(fig_root, setting_tag)
    table_dir = os.path.join(table_root, setting_tag)
    os.makedirs(table_dir, exist_ok=True)

    # Filter.
    df_cut = dataprep.iterative_filter_min_interactions(
        df_sorted,
        min_user_interactions=min_u,
        min_item_interactions=min_i,
        max_iters=max_iters,
    )
    save_filtering_effect_report(df_before, df_cut, fig_dir=fig_dir, table_dir=table_dir)

    # Index and split.
    df_cut, _ = dataprep.reindex_after_filter(df_cut)
    n_users = int(df_cut["user_index"].nunique())
    n_items = int(df_cut["item_index"].nunique())
    df_train, df_val, df_test = dataprep.build_splits_leave_one_out(df_cut)

    assert df_val["user_index"].isin(df_train["user_index"]).all()
    assert df_test["user_index"].isin(df_train["user_index"]).all()

    # Baseline parameters from TRAIN.
    mu = float(df_train["rating"].mean())
    user_mean_arr = models.build_user_mean(df_train, n_users)
    item_mean_arr = np.full(n_items, mu, dtype=np.float32)
    im = df_train.groupby("item_index")["rating"].mean()
    item_mean_arr[im.index.to_numpy(int)] = im.to_numpy(np.float32)
    bu, bi = models.fit_bias_model_by_index(df_train, mu=mu, n_users=n_users, n_items=n_items, reg=10.0, n_iters=20)

    # Rating metrics.
    mf = models.train_biased_mf_sgd(df_train, df_val, n_users=n_users, n_items=n_items, **MF_SWEEP_PARAMS)
    mf_label = mf_label_for(MF_SWEEP_PARAMS["factors"])
    rating_rows = baseline_rating_rows(df_val, df_test, mu, user_mean_arr, item_mean_arr, bu, bi)
    rating_rows.append(mf_rating_row(mf, mf_label, df_val, df_test))
    rating_table = pd.DataFrame(rating_rows)
    rating_table.to_csv(os.path.join(table_dir, "rating_metrics_table.csv"), index=False)

    # Ranking metrics.
    all_items = np.arange(n_items, dtype=int)
    user_interactions = evaluation.build_user_interactions(df_cut)
    df_val_pos = df_val[["user_index", "item_index"]].copy()
    df_test_pos = df_test[["user_index", "item_index"]].copy()

    def score_mf(u_idx, cand):
        return models.mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], u_idx, cand)

    models_for_ranking = make_baseline_score_fns(mu, user_mean_arr, item_mean_arr, bu, bi) + [(mf_label, score_mf)]
    ranking_table = pd.DataFrame(
        [ranking_row(name, fn, df_val_pos, df_test_pos, user_interactions, all_items) for name, fn in models_for_ranking]
    )
    ranking_table.to_csv(os.path.join(table_dir, "ranking_metrics_table_fixed.csv"), index=False)

    # One-row summary with MF as the main model.
    mf_row = ranking_table[ranking_table["model"] == mf_label].iloc[0]
    mf_rating = rating_table[rating_table["model"] == mf_label].iloc[0]
    bias_row = ranking_table[ranking_table["model"] == BIAS_LABEL].iloc[0]
    return {
        "min_user": min_u,
        "min_item": min_i,
        "interactions": int(len(df_cut)),
        "users": int(n_users),
        "items": int(n_items),
        "train": int(len(df_train)),
        "val": int(len(df_val)),
        "test": int(len(df_test)),
        "bias_TEST_NDCG@10": float(bias_row[f"TEST_NDCG@{K}"]),
        "mf_VAL_NDCG@10": float(mf_row[f"VAL_NDCG@{K}"]),
        "mf_TEST_NDCG@10": float(mf_row[f"TEST_NDCG@{K}"]),
        "mf_val_RMSE": float(mf_rating["val_RMSE"]),
        "mf_test_RMSE": float(mf_rating["test_RMSE"]),
    }


def run_threshold_sweep(
    df_raw: pd.DataFrame,
    fig_dir: str = str(FIGURES_DIR),
    table_dir: str = str(TABLES_DIR),
    min_users: tuple[int, ...] = (2, 3, 4, 5),
    min_items: tuple[int, ...] = (2, 3, 4, 5),
) -> pd.DataFrame:
    """Filtering-threshold sensitivity: run_one_setting for every (min_user, min_item) pair.

    Per-setting outputs go to <fig_dir>/threshold_sweep/ and <table_dir>/threshold_sweep/.
    Pairs that cannot be split (min_user = 2) are skipped. Saves threshold_sweep_summary.csv.
    """
    df = dataprep.map_and_clean(df_raw)
    df_sorted = df.sort_values("timestamp", ascending=True)
    df_before = df_sorted.copy()

    fig_root = os.path.join(fig_dir, "threshold_sweep")
    table_root = os.path.join(table_dir, "threshold_sweep")

    rows = []
    for min_u, min_i in product(min_users, min_items):
        print("\n" + "=" * 80)
        print(f"[Sweep] min_user={min_u} min_item={min_i}")
        print("=" * 80)
        try:
            rows.append(run_one_setting(df_sorted, df_before, fig_root, table_root, min_u, min_i))
        except AssertionError:
            print(f"[Sweep] Skipped min_user={min_u} min_item={min_i}: a val/test user has no training interactions.")

    summary = pd.DataFrame(rows)
    os.makedirs(table_root, exist_ok=True)
    summary.to_csv(os.path.join(table_root, "threshold_sweep_summary.csv"), index=False)
    print("\n[Sweep] Summary:")
    print(summary.to_string(index=False))
    return summary


def run_hybrid_grid_search(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_cut: pd.DataFrame,
    n_users: int,
    n_items: int,
    mu: float,
    bu: np.ndarray,
    bi: np.ndarray,
    user_mean_arr: np.ndarray,
    user_hist: dict[int, dict[int, float]],
    table_dir: str,
    mf_factors_grid: tuple[int, ...] = (16, 32, 64),
    mf_reg_grid: tuple[float, ...] = (0.02, 0.05),
    knn_topk_grid: tuple[int, ...] = (20, 50, 100),
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    mf_lr: float = 0.02,
    mf_epochs: int = 10,
    sample_users: int = 500,
    n_neg: int = N_NEG,
    k: int = K,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Hybrid grid search over (mf_factors, mf_reg, knn_topk, alpha) on a sample of validation users.

    Efficient strategy:
      1) Train MF once per (factors, reg).
      2) Precompute Item-kNN once per top_k.
      3) For each (MF config, top_k) cache the (MF score, kNN score) vectors per user.
      4) For each alpha, blend the cached scores and compute VAL HR/NDCG/MRR (cheap).

    Outputs:
      hybrid_grid_results.csv  (all combinations, sorted by VAL_NDCG@k)
      hybrid_best_config.csv   (best combination)
    Returns (df_results, best_config_dict).
    """
    rng = np.random.default_rng(seed)
    n_val_s = min(int(sample_users), len(df_val))
    df_val_s = df_val.sample(n=n_val_s, random_state=seed)

    user_interactions = evaluation.build_user_interactions(df_cut)
    all_items = np.arange(n_items, dtype=int)

    cand_cache: list[tuple[int, np.ndarray]] = []
    for _, row in df_val_s.iterrows():
        u = int(row["user_index"])
        pos = int(row["item_index"])
        interacted = user_interactions.get(u, set())
        neg = evaluation.sample_negatives_for_user(all_items, interacted, pos, n_neg=n_neg, rng=rng)
        cand_cache.append((u, np.concatenate([[pos], neg]).astype(int)))

    print(f"[Hybrid Grid] {len(cand_cache)} (user, candidates) pairs. "
          f"Combos: {len(mf_factors_grid)*len(mf_reg_grid)*len(knn_topk_grid)*len(alpha_grid)} "
          f"(mf x reg x topk x alpha = {len(mf_factors_grid)}x{len(mf_reg_grid)}x{len(knn_topk_grid)}x{len(alpha_grid)})")

    mf_models: dict[tuple[int, float], dict] = {}
    for mf_f in mf_factors_grid:
        for mf_r in mf_reg_grid:
            print(f"[Hybrid Grid] MF train (factors={mf_f}, reg={mf_r})...")
            t0 = time.time()
            mf_models[(int(mf_f), float(mf_r))] = models.train_biased_mf_sgd(
                df_train, df_val, n_users=n_users, n_items=n_items,
                factors=int(mf_f), lr=float(mf_lr), reg=float(mf_r),
                n_epochs=int(mf_epochs), seed=42,
            )
            print(f"  Duration: {time.time() - t0:.1f}s")

    knn_models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    dev_mat = models.build_sparse_deviation_matrix(
        df_train, n_users=n_users, n_items=n_items,
        user_mean_arr=user_mean_arr.astype(np.float32),
    )
    for tk in knn_topk_grid:
        print(f"[Hybrid Grid] Item-kNN precompute (top_k={tk})...")
        t0 = time.time()
        knn_models[int(tk)] = models.compute_item_topk_similarities(dev_mat, top_k=int(tk), block_size=256)
        print(f"  Duration: {time.time() - t0:.1f}s")

    rows: list[dict] = []
    for (mf_f, mf_r), mf in mf_models.items():
        for tk, (ni, nsim) in knn_models.items():
            print(f"[Hybrid Grid] Scoring (factors={mf_f}, reg={mf_r}, top_k={tk})...")
            t0 = time.time()
            scored: list[tuple[np.ndarray, np.ndarray]] = []
            for u, cand in cand_cache:
                mf_s = models.mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], int(u), cand)
                knn_s = np.zeros(len(cand), dtype=float)
                for t, it in enumerate(cand):
                    fb = float(np.clip(mu + bu[int(u)] + bi[int(it)], 1.0, 5.0))
                    knn_s[t] = models.predict_item_knn_one(int(u), int(it), user_hist, user_mean_arr, ni, nsim, fb)
                scored.append((np.asarray(mf_s, dtype=float), knn_s))
            print(f"  Scoring duration: {time.time() - t0:.1f}s")

            for a in alpha_grid:
                rows.append({
                    "mf_factors": int(mf_f),
                    "mf_reg": float(mf_r),
                    "knn_topk": int(tk),
                    "alpha": float(a),
                    **blend_metrics(scored, a, rank_seed=seed + 1000),
                })

    df_grid = pd.DataFrame(rows).sort_values(f"VAL_NDCG@{k}", ascending=False).reset_index(drop=True)
    df_grid.to_csv(os.path.join(table_dir, "hybrid_grid_results.csv"), index=False)
    print(f"[Saved] {os.path.join(table_dir, 'hybrid_grid_results.csv')}")

    best = {kk: (int(v) if isinstance(v, (np.integer, int)) and kk in {"mf_factors", "knn_topk"} else float(v))
            for kk, v in df_grid.iloc[0].to_dict().items()}
    pd.DataFrame([best]).to_csv(os.path.join(table_dir, "hybrid_best_config.csv"), index=False)
    print(f"[Saved] {os.path.join(table_dir, 'hybrid_best_config.csv')}")
    print(f"[Hybrid Grid] Best configuration: {best}")

    return df_grid, best


def prepare_and_run_all(
    df_raw: pd.DataFrame,
    fig_dir: str = str(FIGURES_DIR),
    table_dir: str = str(TABLES_DIR),
    run_hybrid_grid: bool = True,
    run_mf_grid: bool = False,
) -> None:
    """End-to-end pipeline: EDA, filtering, split, baselines, MF, kNN, Hybrid, ranking tables and figures.

    run_mf_grid=False: MF is trained once with MF_PARAMS.
    run_mf_grid=True: run_mf_grid_search tries 36 configurations (factors 16/32/64, lr 0.01/0.02,
      reg 0.02/0.05/0.1, epochs 10/20), picks the lowest validation RMSE and writes
      mf_grid_results.csv. The selected MF is used in every later step (rating metrics, ranking,
      Hybrid), so the MF label and the ranking results can change. This takes much longer than one training.
    run_hybrid_grid=True: also runs the Hybrid grid search at the end (slow).
    """
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # Load, clean and plot the raw data distributions.
    df = dataprep.map_and_clean(df_raw)
    df_sorted = df.sort_values("timestamp", ascending=True)
    df_before = df_sorted.copy()  # snapshot before filtering

    plots.plot_rating_distribution(df["rating"].value_counts().sort_index(), fig_dir)

    ux, uy = ecdf(df.groupby("user_id").size().to_numpy())
    ix, iy = ecdf(df.groupby("item_id").size().to_numpy())
    plots.plot_cdf(ux, uy, ix, iy, fig_dir)

    timeline = df_sorted.set_index("timestamp")["rating"].resample("MS").size()  # MS = month start
    plots.plot_interaction_timeline(timeline, fig_dir)

    # Iterative filtering, indexing and leave-one-out split.
    df_cut = dataprep.iterative_filter_min_interactions(
        df_sorted,
        min_user_interactions=MIN_USER_INTERACTIONS,
        min_item_interactions=MIN_ITEM_INTERACTIONS,
        max_iters=20,
    )
    save_filtering_effect_report(df_before, df_cut, fig_dir=fig_dir, table_dir=table_dir)

    df_cut, _ = dataprep.reindex_after_filter(df_cut)
    n_users = int(df_cut["user_index"].nunique())
    n_items = int(df_cut["item_index"].nunique())
    df_train, df_val, df_test = dataprep.build_splits_leave_one_out(df_cut)

    plots.plot_train_rating_histogram(df_train, out_dir=fig_dir, bins=10)
    plots.plot_train_interaction_distributions(
        df_train,
        out_dir=fig_dir,
        min_user_interactions=MIN_USER_INTERACTIONS,
        min_item_interactions=MIN_ITEM_INTERACTIONS,
        bins=60,
        log_y=True,
    )

    print(f"\n[Data] interactions={len(df_cut)} users={n_users} items={n_items}")
    print(f"[Data] train/val/test={len(df_train)}/{len(df_val)}/{len(df_test)}")

    # Baseline parameters (all from TRAIN) and the Bias model.
    mu = float(df_train["rating"].mean())
    user_mean_arr = models.build_user_mean(df_train, n_users)
    item_mean_arr = np.full(n_items, mu, dtype=np.float32)
    im = df_train.groupby("item_index")["rating"].mean()
    item_mean_arr[im.index.to_numpy(int)] = im.to_numpy(np.float32)
    bu, bi = models.fit_bias_model_by_index(df_train, mu=mu, n_users=n_users, n_items=n_items, reg=10.0, n_iters=20)

    # Matrix factorization: fixed hyperparameters, or the winner of the grid search.
    if run_mf_grid:
        mf, mf_grid_table = run_mf_grid_search(df_train, df_val, n_users, n_items, table_dir=table_dir)
        mf_factors = int(mf_grid_table.iloc[0]["factors"])  # table is sorted by best_val_rmse
    else:
        mf = models.train_biased_mf_sgd(df_train, df_val, n_users=n_users, n_items=n_items, **MF_PARAMS)
        mf_factors = MF_PARAMS["factors"]
    mf_label = mf_label_for(mf_factors)  # model label used in all tables and figures

    # Rating metrics (RMSE and MAE) of the baselines, Bias and MF.
    rating_rows = baseline_rating_rows(df_val, df_test, mu, user_mean_arr, item_mean_arr, bu, bi)
    rating_rows.append(mf_rating_row(mf, mf_label, df_val, df_test))
    rating_table = pd.DataFrame(rating_rows)
    print("\n[Rating Prediction] RMSE + MAE (VAL/TEST):")
    print(rating_table)
    rating_table.to_csv(os.path.join(table_dir, "rating_metrics_table.csv"), index=False)
    plots.plot_baseline_comparison_bars(rating_table=rating_table, out_dir=fig_dir, sort_by="val_RMSE")

    # Ranking setup: all interactions per user (train + val + test) are excluded from negatives.
    all_items = np.arange(n_items, dtype=int)
    user_interactions = evaluation.build_user_interactions(df_cut)
    df_val_pos = df_val[["user_index", "item_index"]].copy()
    df_test_pos = df_test[["user_index", "item_index"]].copy()

    # Training-history statistics.
    user_interactions_train = evaluation.build_user_interactions(df_train)
    hist_u = pd.Series({u: len(items) for u, items in user_interactions_train.items()}, dtype=int)
    hist_stats = {
        "min": int(hist_u.min()),
        "median": float(hist_u.median()),
        "mean": float(hist_u.mean()),
        "p95": float(np.percentile(hist_u.to_numpy(), 95)),
        "max": int(hist_u.max()),
    }
    print("\n[EDA] |history(u)| statistics (TRAIN):")
    print(
        f"min={hist_stats['min']} | median={hist_stats['median']:.2f} | "
        f"mean={hist_stats['mean']:.2f} | p95={hist_stats['p95']:.2f} | max={hist_stats['max']}"
    )
    plots.plot_histogram_user_activity(hist_u, fig_dir)
    plots.plot_histogram_user_logx(hist_u, fig_dir)

    # Ranking score functions: baselines, Bias and MF.
    baseline_fns = make_baseline_score_fns(mu, user_mean_arr, item_mean_arr, bu, bi)
    score_bias = dict(baseline_fns)[BIAS_LABEL]

    def score_mf(u_idx: int, cand: np.ndarray) -> np.ndarray:
        return models.mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], u_idx, cand)

    # Item-kNN (adjusted cosine on a sparse deviation matrix) with a Bias fallback.
    itemknn_meta = {"topK": TOPK_ITEMKNN, "precompute_sec": None, "avg_valid_neighbors": None}

    t0 = time.perf_counter()
    dev_mat = models.build_sparse_deviation_matrix(
        df_train, n_users=n_users, n_items=n_items, user_mean_arr=user_mean_arr.astype(np.float32),
    )
    neigh_idx, neigh_sim = models.compute_item_topk_similarities(dev_mat, top_k=TOPK_ITEMKNN, block_size=256)
    user_hist = models.build_user_history_ratings(df_train)
    itemknn_meta["precompute_sec"] = float(time.perf_counter() - t0)
    print(f"[CF] Item-kNN precompute: {itemknn_meta['precompute_sec']:.3f}s (topK={TOPK_ITEMKNN})")

    # Distribution of the non-zero item-item similarities among the top-K edges.
    sims = neigh_sim.reshape(-1).astype(float)
    sims = sims[np.isfinite(sims)]
    sims = sims[sims != 0.0]
    plots.plot_itemknn_similarity_hist(sims, fig_dir, TOPK_ITEMKNN)

    def score_itemknn(u_idx: int, cand: np.ndarray) -> np.ndarray:
        out = np.zeros(len(cand), dtype=float)
        for t, it in enumerate(cand):
            fb = float(mu + bu[u_idx] + bi[int(it)])  # Bias fallback
            out[t] = models.predict_item_knn_one(u_idx, int(it), user_hist, user_mean_arr, neigh_idx, neigh_sim, fb)
        return out

    # User-kNN (cosine on the same deviation matrix): similarities, diagnostics and coverage.
    t0 = time.time()
    neigh_u_idx, neigh_u_sim = models.compute_user_topk_similarities(dev_mat, top_k=TOPK_USERKNN, block_size=256)
    print(f"\n[CF] User-kNN precompute finished: topK={TOPK_USERKNN}  time={time.time() - t0:.2f}s")

    plots.plot_user_knn_similarity_diagnostics(neigh_u_sim, out_dir=fig_dir, top_k=TOPK_USERKNN)

    # Coverage: how often User-kNN has at least one neighbor who rated the target item.
    cov_tbl = pd.DataFrame([
        {"split": "val", **user_knn_coverage_stats(df_val, user_hist, neigh_u_idx, neigh_u_sim)},
        {"split": "test", **user_knn_coverage_stats(df_test, user_hist, neigh_u_idx, neigh_u_sim)},
    ])
    cov_tbl.to_csv(os.path.join(table_dir, "user_knn_coverage.csv"), index=False)
    print("\n[CF] User-kNN coverage (VAL/TEST):")
    print(cov_tbl)
    print(f"[Saved] {os.path.join(table_dir, 'user_knn_coverage.csv')}")

    def score_userknn(u_idx: int, cand: np.ndarray) -> np.ndarray:
        out = np.zeros(len(cand), dtype=float)
        for t, it in enumerate(cand):
            fb = float(mu + bu[u_idx] + bi[int(it)])  # Bias fallback
            out[t] = models.predict_user_knn_one(int(u_idx), int(it), user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim, fb)
        return out

    # Rating metrics of User-kNN (Bias fallback, consistent with Item-kNN).
    fallback_fn = lambda u, i: (mu + bu[int(u)] + bi[int(i)])
    pv = models.user_knn_predict_df(df_val, user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim, fallback_fn=fallback_fn)
    pt = models.user_knn_predict_df(df_test, user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim, fallback_fn=fallback_fn)
    mv = evaluation.eval_rating_metrics(df_val["rating"].to_numpy(float), pv)
    mt = evaluation.eval_rating_metrics(df_test["rating"].to_numpy(float), pt)
    rating_rows.append({
        "model": f"User-kNN(Cosine, topK={TOPK_USERKNN})",
        "val_RMSE": mv["rmse"],
        "val_MAE": mv["mae"],
        "test_RMSE": mt["rmse"],
        "test_MAE": mt["mae"],
    })

    # Refresh the rating table and chart so that User-kNN is included.
    rating_table = pd.DataFrame(rating_rows)
    print("\n[Rating Prediction] RMSE + MAE (VAL/TEST) + User-kNN added:")
    print(rating_table)
    rating_table.to_csv(os.path.join(table_dir, "rating_metrics_table.csv"), index=False)
    plots.plot_baseline_comparison_bars(rating_table=rating_table, out_dir=fig_dir, sort_by="val_RMSE")

    # Hybrid = alpha * MF + (1 - alpha) * Item-kNN. Alpha is tuned on a sample of 500 validation users,
    # with shared negatives and cached scores (Item-kNN is the slowest part, so it is scored once per user).
    hybrid_alpha_grid = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)

    rng_alpha = np.random.default_rng(42)
    df_val_alpha = df_val.sample(n=min(500, len(df_val)), random_state=42)
    cached_scores: list[tuple[np.ndarray, np.ndarray]] = []
    for u_a, pos_a in zip(df_val_alpha["user_index"].to_numpy(int), df_val_alpha["item_index"].to_numpy(int)):
        neg_a = evaluation.sample_negatives_for_user(
            all_items, user_interactions.get(int(u_a), set()), int(pos_a), n_neg=N_NEG, rng=rng_alpha,
        )
        cand_a = np.concatenate([[pos_a], neg_a]).astype(int)
        cached_scores.append((
            np.asarray(score_mf(int(u_a), cand_a), dtype=float),
            np.asarray(score_itemknn(int(u_a), cand_a), dtype=float),
        ))

    sweep_df = pd.DataFrame(
        [{"alpha": float(a), **blend_metrics(cached_scores, a, rank_seed=43)} for a in hybrid_alpha_grid]
    ).sort_values(f"VAL_NDCG@{K}", ascending=False).reset_index(drop=True)
    sweep_df.to_csv(os.path.join(table_dir, "hybrid_alpha_sweep.csv"), index=False)
    print("\n[Hybrid] Alpha tuning on validation (alpha=1.0 is pure MF, alpha=0.0 is pure Item-kNN):")
    print(sweep_df.to_string(index=False))
    hybrid_alpha = float(sweep_df.iloc[0]["alpha"])
    print(f"[Saved] {os.path.join(table_dir, 'hybrid_alpha_sweep.csv')}")
    print(f"[Hybrid] Selected alpha = {hybrid_alpha:.2f} (max VAL_NDCG@{K}).")

    plots.plot_hybrid_alpha_curve(
        sweep_df, out_dir=fig_dir, k=K,
        filename="hybrid_alpha_sweep.png",
        title="Hybrid α sweep: VAL metrics @10 (default MF & top_k)",
    )

    def score_hybrid(u_idx: int, cand: np.ndarray) -> np.ndarray:
        mf_s = np.asarray(score_mf(int(u_idx), cand), dtype=float)
        knn_s = np.asarray(score_itemknn(int(u_idx), cand), dtype=float)
        return (float(hybrid_alpha) * mf_s + (1.0 - float(hybrid_alpha)) * knn_s).astype(float)

    # Rating metrics of the Hybrid (blend of the MF and Item-kNN rating predictions).
    mf_args = (mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"])
    mf_val_pred = models.mf_predict_ratings(df_val, *mf_args)
    mf_test_pred = models.mf_predict_ratings(df_test, *mf_args)

    def itemknn_rating_predictions(df: pd.DataFrame) -> np.ndarray:
        u_arr = df["user_index"].to_numpy(int)
        i_arr = df["item_index"].to_numpy(int)
        out = np.zeros(len(df), dtype=float)
        for t in range(len(df)):
            uu, ii = int(u_arr[t]), int(i_arr[t])
            fb = float(np.clip(mu + bu[uu] + bi[ii], 1.0, 5.0))
            out[t] = models.predict_item_knn_one(uu, ii, user_hist, user_mean_arr, neigh_idx, neigh_sim, fb)
        return out

    knn_val_pred = itemknn_rating_predictions(df_val)
    knn_test_pred = itemknn_rating_predictions(df_test)
    hybrid_val_pred = float(hybrid_alpha) * mf_val_pred + (1.0 - float(hybrid_alpha)) * knn_val_pred
    hybrid_test_pred = float(hybrid_alpha) * mf_test_pred + (1.0 - float(hybrid_alpha)) * knn_test_pred

    plots.plot_hybrid_pred_distribution(
        mf_pred=mf_test_pred, knn_pred=knn_test_pred, hybrid_pred=hybrid_test_pred,
        out_dir=fig_dir,
        title=f"Prediction distribution per model (TEST, α={hybrid_alpha:.2f})",
    )

    mv_h = evaluation.eval_rating_metrics(df_val["rating"].to_numpy(float), hybrid_val_pred)
    mt_h = evaluation.eval_rating_metrics(df_test["rating"].to_numpy(float), hybrid_test_pred)
    rating_rows.append({
        "model": f"Hybrid(α={hybrid_alpha:.2f}: MF + Item-kNN)",
        "val_RMSE": mv_h["rmse"],
        "val_MAE": mv_h["mae"],
        "test_RMSE": mt_h["rmse"],
        "test_MAE": mt_h["mae"],
    })
    rating_table = pd.DataFrame(rating_rows)
    print("\n[Rating Prediction] RMSE + MAE (VAL/TEST) + Hybrid added:")
    print(rating_table)
    rating_table.to_csv(os.path.join(table_dir, "rating_metrics_table.csv"), index=False)

    # Ranking metrics: HR/NDCG/MRR@10 with 1 positive + 99 sampled negatives, tie-aware ranks.
    models_for_ranking = baseline_fns + [
        (mf_label, score_mf),
        (f"Item-kNN(AdjCos, topK={TOPK_ITEMKNN})", score_itemknn),
        (f"User-kNN(Cosine, topK={TOPK_USERKNN})", score_userknn),
        (f"Hybrid(α={hybrid_alpha:.2f}: MF + Item-kNN)", score_hybrid),
    ]

    ranking_rows = []
    for model_name, score_fn in models_for_ranking:
        is_itemknn = model_name.startswith("Item-kNN(")
        if is_itemknn:
            models.itemknn_reset_neighbor_stats()

        ranking_rows.append(ranking_row(model_name, score_fn, df_val_pos, df_test_pos, user_interactions, all_items))

        if is_itemknn:
            # Average number of neighbors that actually contributed, per scored (user, candidate) pair.
            itemknn_meta["avg_valid_neighbors"] = models.itemknn_get_avg_valid_neighbors()
            print(f"[CF] Item-kNN avg valid neighbors per prediction: {itemknn_meta['avg_valid_neighbors']:.2f}")
            pd.DataFrame([{
                "topK": int(itemknn_meta["topK"]),
                "precompute_sec": float(itemknn_meta["precompute_sec"]),
                "avg_valid_neighbors": float(itemknn_meta["avg_valid_neighbors"]),
            }]).to_csv(os.path.join(table_dir, "item_knn_meta.csv"), index=False)
            print(f"[Saved] {os.path.join(table_dir, 'item_knn_meta.csv')}")

    ranking_table = pd.DataFrame(ranking_rows)
    print("\n[Top-K Ranking] HR@10 / NDCG@10 / MRR@10 (VAL/TEST) with 99 negatives:")
    print(ranking_table)
    ranking_table.to_csv(os.path.join(table_dir, "ranking_metrics_table_fixed.csv"), index=False)

    # Top-3 models and the MF vs Bias difference for the two main TEST metrics.
    top3_ndcg, ndcg_summary, mf_ndcg, bias_ndcg, mf_minus_bias = topn_summary(ranking_table, f"TEST_NDCG@{K}", mf_label)
    ndcg_summary_path = os.path.join(table_dir, "topn_test_ndcg10_summary.csv")
    ndcg_summary.to_csv(ndcg_summary_path, index=False)
    print("\n[Top-N] TEST_NDCG@10 TOP-3:")
    print(top3_ndcg.to_string(index=False))
    print(f"[Top-N] MF vs Bias: {mf_ndcg:.6f} - {bias_ndcg:.6f} = {mf_minus_bias:.6f}")
    print(f"[Saved] {ndcg_summary_path}")

    top3_hr, hr_summary, mf_hr, bias_hr, mf_minus_bias_hr = topn_summary(ranking_table, f"TEST_HR@{K}", mf_label)
    hr_summary_path = os.path.join(table_dir, "topn_test_hr10_summary.csv")
    hr_summary.to_csv(hr_summary_path, index=False)
    print("\n[Top-N] TEST_HR@10 TOP-3:")
    print(top3_hr.to_string(index=False))
    print(f"[Top-N] MF vs Bias (HR@10): {mf_hr:.6f} - {bias_hr:.6f} = {mf_minus_bias_hr:.6f}")
    print(f"[Saved] {hr_summary_path}")

    # Bar charts of the TEST metrics per model.
    plots.plot_topn_bar_test_metric(
        ranking_table=ranking_table,
        out_dir=fig_dir,
        metric_col="TEST_NDCG@10",
        title="Top-N evaluation: TEST_NDCG@10 per model",
        filename="topn_test_ndcg10_bar.png",
    )
    plots.plot_topn_bar_test_metric(
        ranking_table=ranking_table,
        out_dir=fig_dir,
        metric_col="TEST_HR@10",
        title="Top-N evaluation: TEST_HR@10 per model",
        filename="topn_test_hr10_bar.png",
    )

    # Distribution of the positive item's rank (1..100) for Bias vs MF on TEST.
    try:
        ranks_bias, _ = collect_positive_ranks_and_ties(
            df_pos=df_test_pos, user_interactions=user_interactions, all_items=all_items,
            score_fn=score_bias, n_neg=N_NEG, seed=SEED_TEST,
        )
        ranks_mf, _ = collect_positive_ranks_and_ties(
            df_pos=df_test_pos, user_interactions=user_interactions, all_items=all_items,
            score_fn=score_mf, n_neg=N_NEG, seed=SEED_TEST,
        )

        plots.plot_positive_rank_histogram(
            ranks_a=ranks_bias,
            label_a=BIAS_LABEL,
            ranks_b=ranks_mf,
            label_b=mf_label,
            out_dir=fig_dir,
            filename="positive_rank_hist_test_bias_vs_mf.png",
            max_rank=100,
        )

        def rank_stats(ranks: np.ndarray, k_hit: int = 10) -> dict[str, float]:
            r = ranks.astype(float)
            return {
                "median_rank": float(np.median(r)),
                "p90_rank": float(np.percentile(r, 90)),
                "pct_rank_le_k": float(np.mean(r <= float(k_hit))) * 100.0,
            }

        rank_summary = pd.DataFrame([
            {"model": name, "median_rank": s["median_rank"], "p90_rank": s["p90_rank"], "pct_rank_le_10": s["pct_rank_le_k"]}
            for name, s in ((BIAS_LABEL, rank_stats(ranks_bias)), (mf_label, rank_stats(ranks_mf)))
        ])

        rank_summary_path = os.path.join(table_dir, "positive_rank_hist_test_bias_vs_mf_summary.csv")
        rank_summary.to_csv(rank_summary_path, index=False)
        print("\n[RankHist] Summary (TEST):")
        print(rank_summary.to_string(index=False))
        print(f"[Saved] {rank_summary_path}")
    except Exception as e:
        print(f"[WARN] Rank histogram (Bias vs MF) was not saved because of an error: {e}")

    # Negative pool size per user (TEST).
    pool_sizes_test = compute_negative_pool_sizes(df_test_pos, user_interactions, n_items=n_items)
    plots.plot_negative_pool_histogram(pool_sizes=pool_sizes_test, out_dir=fig_dir, filename="neg_pool_size_hist_test.png")

    pool_stats = pd.DataFrame([{
        "split": "test",
        "n": int(len(pool_sizes_test)),
        "min_pool": int(np.min(pool_sizes_test)) if len(pool_sizes_test) else 0,
        "p25_pool": float(np.percentile(pool_sizes_test, 25)) if len(pool_sizes_test) else 0.0,
        "median_pool": float(np.percentile(pool_sizes_test, 50)) if len(pool_sizes_test) else 0.0,
        "mean_pool": float(np.mean(pool_sizes_test)) if len(pool_sizes_test) else 0.0,
        "p75_pool": float(np.percentile(pool_sizes_test, 75)) if len(pool_sizes_test) else 0.0,
        "max_pool": int(np.max(pool_sizes_test)) if len(pool_sizes_test) else 0,
    }])
    pool_stats.to_csv(os.path.join(table_dir, "neg_pool_size_stats_test.csv"), index=False)
    print(f"[Saved] {os.path.join(table_dir, 'neg_pool_size_stats_test.csv')}")

    # Tie rate per model (TEST), for the same models that were evaluated.
    tie_table_test = compute_tie_rate_table(
        df_pos=df_test_pos,
        user_interactions=user_interactions,
        all_items=all_items,
        models_fn=models_for_ranking,
        n_neg=N_NEG,
        seed=SEED_TEST,
    )
    tie_table_test.to_csv(os.path.join(table_dir, "tie_rate_test_table.csv"), index=False)
    print(f"[Saved] {os.path.join(table_dir, 'tie_rate_test_table.csv')}")

    plots.plot_tie_rate_bar(
        tie_table=tie_table_test,
        out_dir=fig_dir,
        filename="tie_rate_test_bar.png",
        title="Tie rate on TEST (ties > 1) per model",
    )

    # Final table: rating and ranking metrics side by side.
    final_table = rating_table.merge(ranking_table, on="model", how="outer")
    print("\n[FINAL] Model comparison (rating + ranking):")
    print(final_table)
    final_table.to_csv(os.path.join(table_dir, "final_models_comparison_fixed.csv"), index=False)

    print(f"\n[Saved] {os.path.join(table_dir, 'rating_metrics_table.csv')}")
    print(f"[Saved] {os.path.join(table_dir, 'ranking_metrics_table_fixed.csv')}")
    print(f"[Saved] {os.path.join(table_dir, 'final_models_comparison_fixed.csv')}")

    # Hybrid grid search over (mf_factors, mf_reg, knn_topk, alpha).
    if run_hybrid_grid:
        print("\n" + "=" * 80)
        print("[Hybrid Grid] Running the Hybrid grid search (expected duration: 15-30 min, depending on the dataset)...")
        print("=" * 80)
        try:
            df_grid, best_cfg = run_hybrid_grid_search(
                df_train=df_train,
                df_val=df_val,
                df_cut=df_cut,
                n_users=n_users,
                n_items=n_items,
                mu=mu,
                bu=bu,
                bi=bi,
                user_mean_arr=user_mean_arr,
                user_hist=user_hist,
                table_dir=table_dir,
            )
            if not df_grid.empty:
                # Alpha curve at the best (mf_factors, mf_reg, knn_topk) slice, plus the grid heatmap.
                sub = df_grid[
                    (df_grid["mf_factors"] == int(best_cfg["mf_factors"])) &
                    (df_grid["mf_reg"] == float(best_cfg["mf_reg"])) &
                    (df_grid["knn_topk"] == int(best_cfg["knn_topk"]))
                ].copy()
                plots.plot_hybrid_alpha_curve(
                    sub, out_dir=fig_dir, k=K,
                    filename="hybrid_alpha_curve_best_config.png",
                    title=(
                        f"Hybrid α sweep at best grid config "
                        f"(factors={best_cfg['mf_factors']}, reg={best_cfg['mf_reg']:.2f}, "
                        f"top_k={best_cfg['knn_topk']})"
                    ),
                )
                plots.plot_hybrid_grid_heatmap(df_grid, out_dir=fig_dir, k=K)
        except Exception as e:
            print(f"[WARN] Hybrid grid search did not finish: {e}")
    else:
        print("\n[Hybrid Grid] Skipped (run_hybrid_grid=False).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recommender experiments.")
    parser.add_argument("--mf-grid", action="store_true",
                        help="run the MF hyperparameter grid search and use its winner for all later steps")
    parser.add_argument("--no-hybrid-grid", action="store_true", help="skip the Hybrid grid search")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="run the filtering-threshold sensitivity study instead of the main pipeline")
    args = parser.parse_args()

    df_raw = dataprep.load_raw_reviews()
    if args.threshold_sweep:
        run_threshold_sweep(df_raw)
    else:
        prepare_and_run_all(df_raw, run_hybrid_grid=not args.no_hybrid_grid, run_mf_grid=args.mf_grid)


if __name__ == "__main__":
    try:
        main()
    except DATA_ERRORS as e:
        sys.exit(f"{type(e).__name__}: {e}")
