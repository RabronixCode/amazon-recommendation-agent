from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def eval_rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE and MAE for rating prediction."""
    return {"rmse": float(root_mean_squared_error(y_true, y_pred)), "mae": float(mean_absolute_error(y_true, y_pred))}


def sample_negatives_for_user(
    all_items: np.ndarray,
    interacted_set: set[int],  # items the user has interacted with
    positive_item: int,
    n_neg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_neg items the user never interacted with (all of them if the pool is smaller)."""
    mask = np.array([(i not in interacted_set) and (i != positive_item) for i in all_items], dtype=bool)
    pool = all_items[mask]
    if len(pool) <= n_neg:
        return pool.astype(int)
    return rng.choice(pool, size=n_neg, replace=False).astype(int)


def rank_of_positive_tie_aware(scores: np.ndarray, pos_index: int, rng: np.random.Generator) -> int:
    """1-based rank of the positive item, breaking ties at random inside the tied block."""
    pos_score = float(scores[pos_index])
    higher = int(np.sum(scores > pos_score))  # strictly better candidates
    ties = int(np.sum(scores == pos_score))  # candidates with the same score (including the positive)
    # The positive is one of the tied elements, so its rank lies in [higher + 1, higher + ties].
    tie_offset = int(rng.integers(low=1, high=ties + 1))
    return higher + tie_offset


def hr_at_k(rank: int, k: int) -> float:
    """Hit rate: 1 if the positive is in the top k, else 0."""
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """NDCG for a single relevant item (the ideal DCG is 1)."""
    return 0.0 if rank > k else 1.0 / math.log2(rank + 1)


def mrr_at_k(rank: int, k: int) -> float:
    """Reciprocal rank, truncated at k."""
    return 0.0 if rank > k else 1.0 / float(rank)


def evaluate_ranking(
    df_pos: pd.DataFrame,  # positive (user_index, item_index) pairs
    user_interactions: dict[int, set[int]],  # user -> all items the user interacted with
    all_items: np.ndarray,
    score_fn: Callable[[int, np.ndarray], np.ndarray],  # score_fn(user, candidates) -> scores
    k: int = 10,
    n_neg: int = 99,
    seed: int = 42,
) -> dict[str, float]:
    """HR@k, NDCG@k and MRR@k with one positive and n_neg sampled negatives per user."""
    rng = np.random.default_rng(seed)
    hrs, ndcgs, mrrs = [], [], []
    for _, row in df_pos.iterrows():
        u = int(row["user_index"])
        pos_item = int(row["item_index"])
        interacted = user_interactions.get(u, set())
        neg_items = sample_negatives_for_user(all_items, interacted, pos_item, n_neg, rng)
        candidates = np.concatenate([[pos_item], neg_items]).astype(int)  # positive is at index 0
        scores = score_fn(u, candidates)
        rnk = rank_of_positive_tie_aware(scores, pos_index=0, rng=rng)
        hrs.append(hr_at_k(rnk, k))
        ndcgs.append(ndcg_at_k(rnk, k))
        mrrs.append(mrr_at_k(rnk, k))
    return {f"HR@{k}": float(np.mean(hrs)), f"NDCG@{k}": float(np.mean(ndcgs)), f"MRR@{k}": float(np.mean(mrrs))}


def build_user_interactions(df: pd.DataFrame) -> dict[int, set[int]]:
    """user_index -> set of item_index values the user interacted with."""
    out: dict[int, set[int]] = {}
    for u, i in zip(df["user_index"].to_numpy(int), df["item_index"].to_numpy(int)):
        out.setdefault(int(u), set()).add(int(i))
    return out
