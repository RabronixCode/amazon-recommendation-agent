"""agent_cli.py

Recommendation agent: trains Bias / MF / Item-kNN / Hybrid, picks the best model by
TEST NDCG@10, saves the artifacts and generates Top-N recommendations with explanations.

All shared logic (data loading, filtering, split, models, evaluation) lives in
dataprep.py, models.py and evaluation.py. This file only contains what is specific to the agent.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import DATA_ERRORS, MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS, resolve_raw_reviews_path
from dataprep import (
    build_splits_leave_one_out,
    iterative_filter_min_interactions,
    load_raw_reviews,
    map_and_clean,
    reindex_after_filter,
)
from evaluation import (
    build_user_interactions,
    evaluate_ranking,
    hr_at_k,
    mrr_at_k,
    ndcg_at_k,
    rank_of_positive_tie_aware,
    sample_negatives_for_user,
)
from models import (
    build_sparse_deviation_matrix,
    build_user_history_ratings,
    build_user_mean,
    compute_item_topk_similarities,
    fit_bias_model_by_index,
    mf_score_candidates,
    predict_item_knn_one,
    train_biased_mf_sgd,
)

Payload = dict[str, object]
ScoreFn = Callable[[int, np.ndarray], np.ndarray]

MODEL_NAMES = ("Bias", "MF", "Item-kNN", "Hybrid")
DEFAULT_MODEL = "Item-kNN"
ALPHA_GRID = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
RATING_MIN, RATING_MAX = 1.0, 5.0

ART_DIR = Path(__file__).resolve().parent / "agent_artifacts"
ART_FILE = ART_DIR / "agent_itemknn.pkl"
REVIEWER_IDS_FILE = ART_DIR / "reviewer_ids.txt"
ITEM_IDS_FILE = ART_DIR / "item_ids.txt"


# Scoring per model: score_fn(user_index, candidates) -> np.ndarray

def _bias_scores(u_idx: int, cand: np.ndarray, payload: Payload) -> np.ndarray:
    """mu + bu[u] + bi[cand], clipped to [1, 5]."""
    raw = float(payload["mu"]) + float(payload["bu"][u_idx]) + payload["bi"][cand]
    return np.clip(raw, RATING_MIN, RATING_MAX).astype(float)


def _mf_scores(u_idx: int, cand: np.ndarray, payload: Payload) -> np.ndarray:
    """mu + bu[u] + bi[i] + P[u] . Q[i]."""
    mf = payload["mf"]
    return mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], u_idx, cand)


def _knn_scores(u_idx: int, cand: np.ndarray, payload: Payload) -> np.ndarray:
    """Item-kNN, with the Bias fallback (mu + bu + bi) when the user has no rated neighbors."""
    fallback = _bias_scores(u_idx, cand, payload)
    out = np.empty(len(cand), dtype=float)
    for t, item in enumerate(cand):
        out[t] = predict_item_knn_one(
            u_idx, int(item), payload["user_hist"], payload["user_mean"],
            payload["neigh_idx"], payload["neigh_sim"], float(fallback[t]),
        )
    return out


def _hybrid_scores(u_idx: int, cand: np.ndarray, payload: Payload) -> np.ndarray:
    """alpha * MF + (1 - alpha) * Item-kNN. alpha=1 is pure MF, alpha=0 is pure Item-kNN."""
    alpha = float(np.clip(payload.get("hybrid_alpha", 0.5), 0.0, 1.0))
    return alpha * _mf_scores(u_idx, cand, payload) + (1.0 - alpha) * _knn_scores(u_idx, cand, payload)


_SCORERS: dict[str, Callable[[int, np.ndarray, Payload], np.ndarray]] = {
    "Bias": _bias_scores,
    "MF": _mf_scores,
    "Item-kNN": _knn_scores,
    "Hybrid": _hybrid_scores,
}


def make_score_fn(model_name: str, payload: Payload) -> ScoreFn:
    """Return score_fn(u_idx, candidates) -> np.ndarray for the given model.

    Used both in the evaluation (evaluate_ranking) and when generating recommendations.
    """
    if model_name not in _SCORERS:
        raise ValueError(f"Unknown model: {model_name}. Allowed: {MODEL_NAMES}")
    scorer = _SCORERS[model_name]
    return lambda u, cand: scorer(int(u), np.asarray(cand, dtype=int), payload)


# Recommendation explanations

def explain_item_knn(u_idx: int, i_idx: int, payload: Payload, top_m: int = 3) -> list[str]:
    user_hist = payload["user_hist"]
    if u_idx not in user_hist:
        return ["We have no rating history for this user, so the recommendation is based on the general model (fallback)."]

    rated = user_hist[u_idx]
    u_mean = float(payload["user_mean"][u_idx])
    item_ids = payload["idx_maps"]["item_index_to_id"]
    contribs: list[tuple[float, str]] = []

    for j, s in zip(payload["neigh_idx"][i_idx], payload["neigh_sim"][i_idx]):
        sim = float(s)
        if not np.isfinite(sim) or sim == 0.0 or int(j) not in rated:
            continue
        weight = abs(sim * (rated[int(j)] - u_mean))
        contribs.append((weight, f"Because you rated '{item_ids[int(j)]}' highly (similarity={sim:.3f})."))

    contribs.sort(key=lambda x: x[0], reverse=True)
    msgs = [m for _, m in contribs[:top_m]]
    return msgs or [
        "The recommendation is based on similar products, but we found no directly similar items "
        "in your history (fallback to the general signal)."
    ]


def _explain_for_model(model_name: str, u_idx: int, i_idx: int, payload: Payload) -> list[str]:
    """List of explanations of why the item was recommended, adapted to the model."""
    if model_name == "Item-kNN":
        return explain_item_knn(u_idx, i_idx, payload)

    mf = payload["mf"]
    mu_v = float(payload["mu"])
    bu_v = float(payload["bu"][u_idx])
    bi_v = float(payload["bi"][i_idx])
    dot = float(np.dot(mf["P"][u_idx], mf["Q"][i_idx]))

    if model_name == "MF":
        pred = float(mf["mu"]) + float(mf["bu"][u_idx]) + float(mf["bi"][i_idx]) + dot
        return [
            f"MF (Matrix Factorization): predicted rating = mu + bu + bi + P_u.Q_i = "
            f"{mu_v:.3f} + {bu_v:+.3f} + {bi_v:+.3f} + {dot:+.3f} ~ {pred:.3f}.",
            f"Latent factor contribution P_u.Q_i = {dot:+.3f} (positive = your profile points in the same direction as the item).",
            f"Item bias bi = {bi_v:+.3f} (how much this item deviates from the global mean in general).",
        ]

    if model_name == "Bias":
        pred = mu_v + bu_v + bi_v
        return [
            f"Bias model: prediction = mu + bu + bi = {mu_v:.3f} + {bu_v:+.3f} + {bi_v:+.3f} ~ {pred:.3f}.",
            f"Item bias bi = {bi_v:+.3f} (positive = generally rated above the global mean).",
            f"Your user bias bu = {bu_v:+.3f} (your tendency to rate above or below mu).",
            "Note: the Bias model is not personalized by content, it ranks all users by the same "
            "globally best items.",
        ]

    if model_name == "Hybrid":
        alpha = float(payload.get("hybrid_alpha", 0.5))
        cand = np.array([i_idx], dtype=int)
        mf_pred = float(_mf_scores(u_idx, cand, payload)[0])
        knn_pred = float(_knn_scores(u_idx, cand, payload)[0])
        blend = alpha * mf_pred + (1.0 - alpha) * knn_pred
        return [
            f"Hybrid model: alpha={alpha:.2f} * MF + (1 - alpha) * Item-kNN.",
            f"  MF prediction = {mf_pred:.3f} (latent factors P_u.Q_i = {dot:+.3f}).",
            f"  Item-kNN prediction = {knn_pred:.3f} (neighbor-based or bias fallback).",
            f"  Combined prediction = {alpha:.2f}*{mf_pred:.3f} + {1.0 - alpha:.2f}*{knn_pred:.3f} ~ {blend:.3f}.",
        ]

    return ["The recommendation was generated by the selected model."]


_MODEL_BANNERS = {
    "Item-kNN": "User found: Item-kNN (Adjusted Cosine) with a bias fallback.",
    "MF": "User found: Matrix Factorization (Biased SGD), prediction from the latent factors P_u and Q_i.",
    "Bias": "User found: Bias model (mu + bu + bi). Note: it ranks all users by the same globally best items.",
}


# Top-N recommendations for a user

def ensure_lookups(payload: Payload) -> Payload:
    """Add a fast user_id -> user_index map (also for old pickle files)."""
    if "user_id_to_index" not in payload:
        ids = payload["idx_maps"]["user_index_to_id"]
        payload["user_id_to_index"] = {uid: idx for idx, uid in enumerate(ids)}
    return payload


def recommend_top_n_for_user(
    user_id: str,
    payload: Payload,
    model_name: Optional[str] = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Top-N recommendations for a user with the given (or the best) model; an unknown user gets the cold-start flow."""
    ensure_lookups(payload)
    model_name = model_name or str(payload.get("best_model", DEFAULT_MODEL))
    item_ids = payload["idx_maps"]["item_index_to_id"]
    n_items = len(item_ids)

    u_idx = payload["user_id_to_index"].get(user_id)
    known_user = u_idx is not None

    candidates = np.arange(n_items, dtype=int)
    if known_user:
        seen = payload["user_hist"].get(u_idx)
        if seen:
            mask = np.ones(n_items, dtype=bool)
            mask[list(seen)] = False
            candidates = candidates[mask]
        scores = make_score_fn(model_name, payload)(u_idx, candidates)
    else:
        # Cold-start: there is no user-specific signal, so use mu + item bias.
        raw = float(payload["mu"]) + payload["bi"][candidates]
        scores = np.clip(raw, RATING_MIN, RATING_MAX).astype(float)

    order = np.argsort(-scores, kind="stable")[:top_n]
    results = [
        {"asin": item_ids[int(candidates[o])], "score": float(scores[o]), "item_index": int(candidates[o])}
        for o in order
    ]

    explanation = [f"Model used: {model_name}"]
    if not known_user:
        explanation.append(
            "User not found in the dataset, cold-start: recommendations are based on the general signal (mu + item bias)."
        )
    elif model_name == "Hybrid":
        a = float(payload.get("hybrid_alpha", 0.5))
        explanation.append(
            f"User found: Hybrid (alpha={a:.2f}*MF + (1-alpha)*Item-kNN), a combination of global and local signal."
        )
    elif model_name in _MODEL_BANNERS:
        explanation.append(_MODEL_BANNERS[model_name])

    n_explain = min(5, len(results))
    if known_user:
        item_explanations = {
            r["asin"]: _explain_for_model(model_name, u_idx, r["item_index"], payload) for r in results[:n_explain]
        }
    else:
        item_explanations = {
            r["asin"]: ["The recommendation was chosen from general popularity and ratings (cold-start fallback)."]
            for r in results[:n_explain]
        }

    return {
        "user_id": user_id,
        "model": model_name,
        "known_user": known_user,
        "results": results,
        "explanation": explanation,
        "item_explanations": item_explanations,
    }


# Model evaluation (Top-N ranking)

def tune_hybrid_alpha(
    payload: Payload,
    df_val: pd.DataFrame,
    n_items: int,
    alphas: tuple[float, ...] = ALPHA_GRID,
    sample_users: int = 500,
    n_neg: int = 99,
    k: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """Pick alpha on the validation split, with shared negatives for all alpha values.

    1) Take a sample of validation pairs (user, positive item).
    2) For each user, draw 99 negatives once and compute the MF and Item-kNN scores of the 100 candidates.
    3) For each alpha, only blend the already computed scores, so the whole search costs
       roughly as much as a single Item-kNN pass.
    """
    rng = np.random.default_rng(seed)
    df_val_s = df_val.sample(n=min(int(sample_users), len(df_val)), random_state=seed)

    user_interactions = build_user_interactions(payload["df_cut"])
    all_items = np.arange(n_items, dtype=int)
    mf_fn = make_score_fn("MF", payload)
    knn_fn = make_score_fn("Item-kNN", payload)

    cached: list[tuple[np.ndarray, np.ndarray]] = []
    for u, pos_item in zip(df_val_s["user_index"].to_numpy(int), df_val_s["item_index"].to_numpy(int)):
        negatives = sample_negatives_for_user(all_items, user_interactions.get(int(u), set()), int(pos_item), n_neg, rng)
        cand = np.concatenate([[pos_item], negatives]).astype(int)
        cached.append((np.asarray(mf_fn(u, cand), dtype=float), np.asarray(knn_fn(u, cand), dtype=float)))

    rows = []
    for a in alphas:
        rank_rng = np.random.default_rng(seed + 1000)  # same tie-breaking for every alpha
        ranks = [
            rank_of_positive_tie_aware(float(a) * mf_s + (1.0 - float(a)) * knn_s, 0, rank_rng)
            for mf_s, knn_s in cached
        ]
        rows.append({
            "alpha": float(a),
            f"VAL_HR@{k}": float(np.mean([hr_at_k(r, k) for r in ranks])),
            f"VAL_NDCG@{k}": float(np.mean([ndcg_at_k(r, k) for r in ranks])),
            f"VAL_MRR@{k}": float(np.mean([mrr_at_k(r, k) for r in ranks])),
        })

    df_alpha = pd.DataFrame(rows).sort_values(f"VAL_NDCG@{k}", ascending=False).reset_index(drop=True)
    return df_alpha, float(df_alpha.iloc[0]["alpha"])


def _sample_rows(df: pd.DataFrame, sample_users: int | None, seed: int) -> pd.DataFrame:
    if sample_users is None or int(sample_users) >= len(df):
        return df
    return df.sample(n=int(sample_users), random_state=seed)


def evaluate_models(
    payload: Payload,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    n_items: int,
    sample_users: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """HR@10 / NDCG@10 / MRR@10 for all models (1 positive + 99 negatives per user)."""
    val_pos = _sample_rows(df_val, sample_users, seed)[["user_index", "item_index"]]
    test_pos = _sample_rows(df_test, sample_users, seed + 1)[["user_index", "item_index"]]
    user_interactions = build_user_interactions(payload["df_cut"])
    all_items = np.arange(n_items, dtype=int)

    rows = []
    for model_name in MODEL_NAMES:
        score_fn = make_score_fn(model_name, payload)
        val_m = evaluate_ranking(val_pos, user_interactions, all_items, score_fn, k=10, n_neg=99, seed=seed)
        test_m = evaluate_ranking(test_pos, user_interactions, all_items, score_fn, k=10, n_neg=99, seed=seed + 1)
        rows.append({
            "model": model_name,
            "VAL_HR@10": val_m["HR@10"], "VAL_NDCG@10": val_m["NDCG@10"], "VAL_MRR@10": val_m["MRR@10"],
            "TEST_HR@10": test_m["HR@10"], "TEST_NDCG@10": test_m["NDCG@10"], "TEST_MRR@10": test_m["MRR@10"],
        })
    return pd.DataFrame(rows)


def print_model_comparison(payload: Payload) -> None:
    """Print the metrics table, the selected model and the alpha search."""
    metrics = payload.get("metrics")
    print("\n" + "=" * 80)
    print("Model comparison (Top-N ranking metrics, sample of validation and test users)")
    print("=" * 80)
    if metrics is None or len(metrics) == 0:
        print("(Metrics are not available. Run again with --retrain.)")
        print("=" * 80)
        return

    cols = ["model", "VAL_HR@10", "VAL_NDCG@10", "VAL_MRR@10", "TEST_HR@10", "TEST_NDCG@10", "TEST_MRR@10"]
    df_show = metrics[cols].copy()
    for c in cols[1:]:
        df_show[c] = df_show[c].map(lambda x: f"{float(x):.4f}")
    print(df_show.to_string(index=False))
    print("-" * 80)

    best = str(payload.get("best_model", ""))
    if best in set(metrics["model"].astype(str)):
        best_row = metrics[metrics["model"] == best].iloc[0]
        print(f"Best model by TEST_NDCG@10: {best} (TEST_NDCG@10={float(best_row['TEST_NDCG@10']):.4f})")
        print(
            "Reason for the choice: NDCG@10 measures how high the positive items are ranked among 99 random "
            "negatives, which is a direct metric of Top-N recommendation quality, unlike RMSE, which measures "
            "the accuracy of the rating prediction."
        )
    print("=" * 80)

    alpha_sweep = payload.get("hybrid_alpha_sweep")
    chosen_alpha = payload.get("hybrid_alpha")
    if isinstance(alpha_sweep, pd.DataFrame) and len(alpha_sweep) > 0:
        print("\nHybrid alpha tuning (val): alpha=1.0 is pure MF, alpha=0.0 is pure Item-kNN")
        df_a = alpha_sweep.copy()
        for c in df_a.columns:
            df_a[c] = df_a[c].map(lambda x, c=c: f"{float(x):.2f}" if c == "alpha" else f"{float(x):.4f}")
        print(df_a.to_string(index=False))
        if chosen_alpha is not None:
            print(f"Selected alpha for Hybrid = {float(chosen_alpha):.2f} (max VAL_NDCG@10).")
        print("=" * 80)


# Artifacts: save / load

def dump_reviewer_ids(payload: Payload) -> None:
    """Save all reviewerIDs and ASINs of the filtered dataset to text files."""
    ART_DIR.mkdir(parents=True, exist_ok=True)
    for path, key, label in (
        (REVIEWER_IDS_FILE, "user_index_to_id", "reviewerIDs"),
        (ITEM_IDS_FILE, "item_index_to_id", "item ASINs"),
    ):
        ids = list(payload["idx_maps"][key])
        path.write_text("".join(f"{x}\n" for x in ids), encoding="utf-8")
        print(f"[Saved] {len(ids)} {label} to {path}")


def save_artifacts(payload: Payload) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(ART_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"[Saved] Artifacts saved to: {ART_FILE}")
    dump_reviewer_ids(payload)


def load_artifacts() -> Optional[Payload]:
    if not ART_FILE.exists():
        return None
    try:
        with open(ART_FILE, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"[Warn] Failed to load the artifacts ({exc}). Retraining.")
        return None
    print(f"[Loaded] Artifacts loaded from: {ART_FILE}")
    if not REVIEWER_IDS_FILE.exists():  # backfill for an old pickle
        try:
            dump_reviewer_ids(payload)
        except OSError as exc:
            print(f"[Warn] Dumping the reviewerIDs failed: {exc}")
    return payload


# Training

def prepare_agent_model(
    min_user: int = MIN_USER_INTERACTIONS,
    min_item: int = MIN_ITEM_INTERACTIONS,
    top_k: int = 50,
    mf_factors: int = 64,
    mf_epochs: int = 20,
    mf_lr: float = 0.02,
    mf_reg: float = 0.02,
    eval_sample_users: int | None = None,
) -> Payload:
    """Train Bias / MF / Item-kNN, pick the Hybrid alpha, evaluate and pick the best model."""
    df = map_and_clean(load_raw_reviews()).sort_values("timestamp", ascending=True)
    df_cut = iterative_filter_min_interactions(
        df, min_user_interactions=min_user, min_item_interactions=min_item, max_iters=20,
    )
    df_cut, idx_maps = reindex_after_filter(df_cut)

    df_train, df_val, df_test = build_splits_leave_one_out(df_cut)
    assert df_val["user_index"].isin(df_train["user_index"]).all()
    assert df_test["user_index"].isin(df_train["user_index"]).all()

    n_users = int(df_cut["user_index"].nunique())
    n_items = int(df_cut["item_index"].nunique())
    print(f"[Data] users={n_users}, items={n_items}, train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    print("[Train] Bias model (mu + bu + bi)...")
    mu = float(df_train["rating"].mean())
    bu, bi = fit_bias_model_by_index(df_train, mu, n_users, n_items, reg=10.0, n_iters=20)
    user_mean = build_user_mean(df_train, n_users)
    user_hist = build_user_history_ratings(df_train)

    print(f"[Train] Item-kNN (Adjusted Cosine, top_k={top_k})...")
    dev_mat = build_sparse_deviation_matrix(df_train, n_users, n_items, user_mean)
    neigh_idx, neigh_sim = compute_item_topk_similarities(dev_mat, top_k=top_k, block_size=256)

    print(f"[Train] MF (Biased SGD, factors={mf_factors}, epochs={mf_epochs})...")
    mf = train_biased_mf_sgd(
        df_train, df_val, n_users=n_users, n_items=n_items,
        factors=mf_factors, lr=mf_lr, reg=mf_reg, n_epochs=mf_epochs, seed=42,
    )

    payload: Payload = {
        "df_cut": df_cut, "idx_maps": idx_maps,
        "mu": mu, "bu": bu, "bi": bi,
        "user_mean": user_mean, "user_hist": user_hist,
        "neigh_idx": neigh_idx, "neigh_sim": neigh_sim,
        "mf": mf,
        "min_user": min_user, "min_item": min_item, "top_k": top_k,
        "mf_factors": mf_factors, "mf_epochs": mf_epochs,
    }

    print(f"[Tune] Hybrid alpha on validation (grid={list(ALPHA_GRID)})...")
    alpha_table, best_alpha = tune_hybrid_alpha(payload, df_val, n_items=n_items, sample_users=500, seed=42)
    payload["hybrid_alpha"] = best_alpha
    payload["hybrid_alpha_sweep"] = alpha_table
    print(alpha_table.to_string(index=False))
    print(f"[Tune] Selected alpha (Hybrid) = {best_alpha:.2f} (by VAL_NDCG@10).")

    n_eval = "all" if eval_sample_users is None else f"~{eval_sample_users}"
    print(f"[Eval] Comparing models on {n_eval} val/test users (HR@10, NDCG@10, MRR@10)...")
    metrics = evaluate_models(payload, df_val, df_test, n_items=n_items, sample_users=eval_sample_users, seed=42)
    payload["metrics"] = metrics

    best_model = str(metrics.loc[int(metrics["TEST_NDCG@10"].astype(float).idxmax()), "model"])
    payload["best_model"] = best_model
    print(f"[Eval] Best model by TEST_NDCG@10: {best_model}")
    return payload


# CLI

def print_recommendation_block(resp: dict[str, object], top_n: int) -> None:
    print("\n" + "=" * 80)
    print(f"USER: {resp['user_id']}  |  MODEL: {resp.get('model', '?')}")
    for line in resp["explanation"]:
        print(f"- {line}")
    print("-" * 80)

    results = resp["results"]
    if not results:
        print("No recommendations for this user.")
        return

    print(f"TOP-{min(top_n, len(results))} recommendations (by predicted rating):")
    for rank, r in enumerate(results[:top_n], start=1):
        print(f"{rank:02d}. ASIN={r['asin']} | predicted_rating={r['score']:.3f}")
        for ex in resp["item_explanations"].get(r["asin"], [])[:3]:
            print(f"    - {ex}")
    print("=" * 80)


def run_single_recommendation(
    payload: Payload, user_id: str, top_n: int, model_name: Optional[str] = None,
) -> None:
    resp = recommend_top_n_for_user(str(user_id), payload, model_name=model_name, top_n=int(top_n))
    print_recommendation_block(resp, top_n=int(top_n))


def run_demo(payload: Payload, top_n: int = 10, model_name: Optional[str] = None) -> None:
    """Top-N for the first few known users and one cold-start user."""
    sample_users = [str(u) for u in payload["idx_maps"]["user_index_to_id"][:4]] + ["NEW_USER_123"]
    for u in sample_users:
        run_single_recommendation(payload, user_id=u, top_n=top_n, model_name=model_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Amazon Recommender (Bias / MF / Item-kNN / Hybrid)")
    parser.add_argument("--retrain", action="store_true", help="Retrain and save the artifacts")
    parser.add_argument("--demo", action="store_true", help="Top-N for a few example users")
    parser.add_argument("--user", type=str, default=None, help="user_id (reviewerID) to recommend for")
    parser.add_argument("--topn", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_NAMES),
                        help="Model to use (default: best by TEST_NDCG@10)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Hybrid weight alpha in [0, 1] (default: selected on validation)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.topn <= 0:
        raise ValueError("--topn must be a positive number.")
    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in the range [0, 1].")

    payload = None if args.retrain else load_artifacts()
    if payload is None:
        raw_path = Path(resolve_raw_reviews_path())
        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {raw_path}")
        print("[Info] Preparing the agent model (this can take a while on the first run)...")
        payload = prepare_agent_model()
        save_artifacts(payload)

    if "metrics" not in payload or "best_model" not in payload:
        print("[Warn] The loaded artifacts have no metrics. Run with --retrain for a new evaluation.")
        payload.setdefault("best_model", DEFAULT_MODEL)

    if args.alpha is not None:
        payload["hybrid_alpha"] = float(args.alpha)
        print(f"[Info] Hybrid alpha override: {args.alpha:.2f}")

    print_model_comparison(payload)

    best = payload.get("best_model")
    chosen_model = args.model or str(best or DEFAULT_MODEL)
    if args.model and args.model != best:
        print(f"\n[Info] Override: using model '{chosen_model}' instead of best='{best}'.")
    else:
        print(f"\n[Info] Active model for recommendations: {chosen_model}")

    if args.demo:
        run_demo(payload, top_n=args.topn, model_name=chosen_model)
        return
    if args.user:
        run_single_recommendation(payload, user_id=args.user, top_n=args.topn, model_name=chosen_model)
        return

    print("\nCommands:")
    print("  - Enter a user_id (reviewerID) to get Top-N recommendations.")
    print("  - An empty user_id or 'exit' quits.")
    while True:
        u = input("\nEnter user_id (or 'exit'): ").strip()
        if not u or u.lower() in ("exit", "quit", "q"):
            break
        run_single_recommendation(payload, user_id=u, top_n=args.topn, model_name=chosen_model)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except DATA_ERRORS as e:
        sys.exit(f"{type(e).__name__}: {e}")
