from __future__ import annotations  # Forward type hints.

import math
from typing import Callable
import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_error  # RMSE.
from sklearn.metrics import mean_absolute_error  # MAE.

def eval_rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:  # RMSE+MAE.
    return {"rmse": float(root_mean_squared_error(y_true, y_pred)), "mae": float(mean_absolute_error(y_true, y_pred))}  # Vrati.

def sample_negatives_for_user(
    all_items: np.ndarray,  # svi itemi.
    interacted_set: set[int],  # user interakcije.
    positive_item: int,  # pozitivan.
    n_neg: int,  # broj negativa.
    rng: np.random.Generator,  # RNG.
) -> np.ndarray:  # negativni itemi.
    mask = np.array([(i not in interacted_set) and (i != positive_item) for i in all_items], dtype=bool)  # filter.
    pool = all_items[mask]  # pool negativa.
    if len(pool) <= n_neg:  # nema dovoljno.
        return pool.astype(int)  # vrati sve.
    return rng.choice(pool, size=n_neg, replace=False).astype(int)  # uzmi n_neg.


def rank_of_positive_tie_aware(scores: np.ndarray, pos_index: int, rng: np.random.Generator) -> int:  # Fer rang u tie slučaju.
    pos_score = float(scores[pos_index])  # skor pozitivnog.
    higher = int(np.sum(scores > pos_score))  # koliko je strogo boljih.
    ties = int(np.sum(scores == pos_score))  # koliko je jednakih.
    # Pozitivan je jedan od tie elemenata: njegov rang je u [higher+1, higher+ties].
    tie_offset = int(rng.integers(low=1, high=ties + 1))  # random offset unutar tie bloka.
    return higher + tie_offset  # 1-based rang.


def hr_at_k(rank: int, k: int) -> float:  # Hit Rate.
    return 1.0 if rank <= k else 0.0  # 1/0.


def ndcg_at_k(rank: int, k: int) -> float:  # NDCG za single-positive.
    return 0.0 if rank > k else 1.0 / math.log2(rank + 1)  # DCG (ideal=1).


def mrr_at_k(rank: int, k: int) -> float:  # MRR.
    return 0.0 if rank > k else 1.0 / float(rank)  # 1/rank.


def evaluate_ranking(
    df_pos: pd.DataFrame,  # pozitivni parovi.
    user_interactions: dict[int, set[int]],  # user->set interakcija.
    all_items: np.ndarray,  # svi itemi.
    score_fn,  # fn(u, candidates)->scores.
    k: int = 10,  # TopK.
    n_neg: int = 99,  # 99 negativa.
    seed: int = 42,  # seed.
) -> dict[str, float]:  # metrike.
    rng = np.random.default_rng(seed)  # RNG.
    hrs, ndcgs, mrrs = [], [], []  # akumulacija.
    for _, row in df_pos.iterrows():  # svaki user pozitivan.
        u = int(row["user_index"])  # user.
        pos_item = int(row["item_index"])  # pozitivan item.
        interacted = user_interactions.get(u, set())  # istorija.
        neg_items = sample_negatives_for_user(all_items, interacted, pos_item, n_neg, rng)  # negative.
        candidates = np.concatenate([[pos_item], neg_items]).astype(int)  # pos na 0.
        scores = score_fn(u, candidates)  # skorovi.
        rnk = rank_of_positive_tie_aware(scores, pos_index=0, rng=rng)  # FIX rang (tie-aware).
        hrs.append(hr_at_k(rnk, k))  # HR.
        ndcgs.append(ndcg_at_k(rnk, k))  # NDCG.
        mrrs.append(mrr_at_k(rnk, k))  # MRR.
    return {f"HR@{k}": float(np.mean(hrs)), f"NDCG@{k}": float(np.mean(ndcgs)), f"MRR@{k}": float(np.mean(mrrs))}  # proseci.
