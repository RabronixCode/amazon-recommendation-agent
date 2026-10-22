"""agent_cli.py

Agent za preporuke: trenira Bias / MF / Item-kNN / Hybrid, bira najbolji model po
TEST NDCG@10, snima artefakte i generise Top-N preporuke sa objasnjenjima.

Sva zajednicka logika (učitavanje podataka, filtriranje, split, modeli, evaluacija)
zivi u dataprep.py, models.py i evaluation.py. Ovde je samo ono što je specificno za agenta.
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


# =========================
# 1) Skoring po modelu: score_fn(user_index, candidates) -> np.ndarray
# =========================

def _bias_scores(u_idx: int, cand: np.ndarray, payload: Payload) -> np.ndarray:
    """mu + bu[u] + bi[cand], ogranicen na [1, 5]."""
    raw = float(payload["mu"]) + float(payload["bu"][u_idx]) + payload["bi"][cand]
    return np.clip(raw, RATING_MIN, RATING_MAX).astype(float)


def _mf_scores(u_idx: int, cand: np.ndarray, payload: Payload) -> np.ndarray:
    """mu + bu[u] + bi[i] + P[u] . Q[i]."""
    mf = payload["mf"]
    return mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], u_idx, cand)


def _knn_scores(u_idx: int, cand: np.ndarray, payload: Payload) -> np.ndarray:
    """Item-kNN, uz Bias fallback (mu + bu + bi) kada korisnik nema ocenjenih suseda."""
    fallback = _bias_scores(u_idx, cand, payload)
    out = np.empty(len(cand), dtype=float)
    for t, item in enumerate(cand):
        out[t] = predict_item_knn_one(
            u_idx, int(item), payload["user_hist"], payload["user_mean"],
            payload["neigh_idx"], payload["neigh_sim"], float(fallback[t]),
        )
    return out


def _hybrid_scores(u_idx: int, cand: np.ndarray, payload: Payload) -> np.ndarray:
    """alpha * MF + (1 - alpha) * Item-kNN. alpha=1 je čist MF, alpha=0 čist Item-kNN."""
    alpha = float(np.clip(payload.get("hybrid_alpha", 0.5), 0.0, 1.0))
    return alpha * _mf_scores(u_idx, cand, payload) + (1.0 - alpha) * _knn_scores(u_idx, cand, payload)


_SCORERS: dict[str, Callable[[int, np.ndarray, Payload], np.ndarray]] = {
    "Bias": _bias_scores,
    "MF": _mf_scores,
    "Item-kNN": _knn_scores,
    "Hybrid": _hybrid_scores,
}


def make_score_fn(model_name: str, payload: Payload) -> ScoreFn:
    """Vraca score_fn(u_idx, candidates) -> np.ndarray za dati model.

    Koristi se i u evaluaciji (evaluate_ranking) i pri generisanju preporuka.
    """
    if model_name not in _SCORERS:
        raise ValueError(f"Nepoznat model: {model_name}. Dozvoljeno: {MODEL_NAMES}")
    scorer = _SCORERS[model_name]
    return lambda u, cand: scorer(int(u), np.asarray(cand, dtype=int), payload)


# =========================
# 2) Objasnjenja preporuka
# =========================

def explain_item_knn(u_idx: int, i_idx: int, payload: Payload, top_m: int = 3) -> list[str]:
    user_hist = payload["user_hist"]
    if u_idx not in user_hist:
        return ["Nemamo istoriju ocena za ovog korisnika, pa je preporuka zasnovana na opštem modelu (fallback)."]

    rated = user_hist[u_idx]
    u_mean = float(payload["user_mean"][u_idx])
    item_ids = payload["idx_maps"]["item_index_to_id"]
    contribs: list[tuple[float, str]] = []

    for j, s in zip(payload["neigh_idx"][i_idx], payload["neigh_sim"][i_idx]):
        sim = float(s)
        if not np.isfinite(sim) or sim == 0.0 or int(j) not in rated:
            continue
        weight = abs(sim * (rated[int(j)] - u_mean))
        contribs.append((weight, f"Zato što ste visoko ocenili '{item_ids[int(j)]}' (sličnost={sim:.3f})."))

    contribs.sort(key=lambda x: x[0], reverse=True)
    msgs = [m for _, m in contribs[:top_m]]
    return msgs or [
        "Preporuka je zasnovana na sličnim proizvodima, ali nismo našli direktne slične stavke "
        "iz vaše istorije (fallback na opšti signal)."
    ]


def _explain_for_model(model_name: str, u_idx: int, i_idx: int, payload: Payload) -> list[str]:
    """Lista objasnjenja zašto je item preporucen, prilagodjena modelu."""
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
            f"MF (Matrix Factorization): predviđena ocena = mu + bu + bi + P_u.Q_i = "
            f"{mu_v:.3f} + {bu_v:+.3f} + {bi_v:+.3f} + {dot:+.3f} ~ {pred:.3f}.",
            f"Doprinos latentnih faktora P_u.Q_i = {dot:+.3f} (pozitivno = vaš profil leži u istom pravcu kao item).",
            f"Item bias bi = {bi_v:+.3f} (koliko ovaj item generalno odstupa od globalnog proseka).",
        ]

    if model_name == "Bias":
        pred = mu_v + bu_v + bi_v
        return [
            f"Bias model: predikcija = mu + bu + bi = {mu_v:.3f} + {bu_v:+.3f} + {bi_v:+.3f} ~ {pred:.3f}.",
            f"Item bias bi = {bi_v:+.3f} (pozitivno = generalno ocenjen iznad globalnog proseka).",
            f"Vaš user bias bu = {bu_v:+.3f} (vaša tendencija da ocenjujete iznad ili ispod mu).",
            "Napomena: Bias model nije personalizovan po sadržaju, rangira sve korisnike po istim "
            "globalno najboljim itemima.",
        ]

    if model_name == "Hybrid":
        alpha = float(payload.get("hybrid_alpha", 0.5))
        cand = np.array([i_idx], dtype=int)
        mf_pred = float(_mf_scores(u_idx, cand, payload)[0])
        knn_pred = float(_knn_scores(u_idx, cand, payload)[0])
        blend = alpha * mf_pred + (1.0 - alpha) * knn_pred
        return [
            f"Hybrid model: alpha={alpha:.2f} * MF + (1 - alpha) * Item-kNN.",
            f"  MF predikcija = {mf_pred:.3f} (latentni faktori P_u.Q_i = {dot:+.3f}).",
            f"  Item-kNN predikcija = {knn_pred:.3f} (sused-based ili bias fallback).",
            f"  Kombinovana predikcija = {alpha:.2f}*{mf_pred:.3f} + {1.0 - alpha:.2f}*{knn_pred:.3f} ~ {blend:.3f}.",
        ]

    return ["Preporuka je generisana izabranim modelom."]


_MODEL_BANNERS = {
    "Item-kNN": "Korisnik pronađen: Item-kNN (Adjusted Cosine) sa bias fallback-om.",
    "MF": "Korisnik pronađen: Matrix Factorization (Biased SGD), predikcija iz latentnih faktora P_u i Q_i.",
    "Bias": "Korisnik pronađen: Bias model (mu + bu + bi). Napomena: rangira sve korisnike po istim globalno najboljim itemima.",
}


# =========================
# 3) Top-N preporuke za korisnika
# =========================

def ensure_lookups(payload: Payload) -> Payload:
    """Dodaje brzu mapu user_id -> user_index (i za stare pickle fajlove)."""
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
    """Top-N preporuke za korisnika zadatim (ili najboljim) modelom; nepoznat korisnik dobija cold-start."""
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
        # Cold-start: nema user-specificnog signala, koristimo mu + item bias.
        raw = float(payload["mu"]) + payload["bi"][candidates]
        scores = np.clip(raw, RATING_MIN, RATING_MAX).astype(float)

    order = np.argsort(-scores, kind="stable")[:top_n]
    results = [
        {"asin": item_ids[int(candidates[o])], "score": float(scores[o]), "item_index": int(candidates[o])}
        for o in order
    ]

    explanation = [f"Korišćen model: {model_name}"]
    if not known_user:
        explanation.append(
            "Korisnik nije pronađen u datasetu, cold-start: preporuke su zasnovane na opštem signalu (mu + item bias)."
        )
    elif model_name == "Hybrid":
        a = float(payload.get("hybrid_alpha", 0.5))
        explanation.append(
            f"Korisnik pronađen: Hybrid (alpha={a:.2f}*MF + (1-alpha)*Item-kNN), kombinovani globalni i lokalni signal."
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
            r["asin"]: ["Preporuka je izabrana na osnovu opšte popularnosti i ocena (cold-start fallback)."]
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


# =========================
# 4) Evaluacija modela (Top-N ranking)
# =========================

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
    """Bira alpha na validaciji, uz zajednicke negative za sve alpha vrednosti.

    1) Uzme uzorak validacionih parova (user, pozitivan item).
    2) Za svakog korisnika jednom izvuce 99 negativa i izračuna MF i Item-kNN skorove za 100 kandidata.
    3) Za svaku alpha samo meša vec izračunate skorove, pa cela pretraga košta
       otprilike kao jedan Item-kNN prolaz.
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
        rank_rng = np.random.default_rng(seed + 1000)  # isti tie-breaking za svaku alpha
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
    """HR@10 / NDCG@10 / MRR@10 za sve modele (1 pozitivan + 99 negativa po korisniku)."""
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
    """Ispis tabele metrika, izabranog modela i alpha pretrage."""
    metrics = payload.get("metrics")
    print("\n" + "=" * 80)
    print("Poređenje modela (Top-N ranking metrike, uzorak validacionih i test korisnika)")
    print("=" * 80)
    if metrics is None or len(metrics) == 0:
        print("(Metrike nisu dostupne. Pokrenite ponovo sa --retrain.)")
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
        print(f"Najbolji model po TEST_NDCG@10: {best} (TEST_NDCG@10={float(best_row['TEST_NDCG@10']):.4f})")
        print(
            "Razlog izbora: NDCG@10 meri koliko su pozitivni itemi rangirani visoko među 99 nasumičnih "
            "negativa, što je direktna metrika kvaliteta Top-N preporuka, za razliku od RMSE koji meri "
            "tačnost predikcije ocene."
        )
    print("=" * 80)

    alpha_sweep = payload.get("hybrid_alpha_sweep")
    chosen_alpha = payload.get("hybrid_alpha")
    if isinstance(alpha_sweep, pd.DataFrame) and len(alpha_sweep) > 0:
        print("\nHybrid alpha tuning (val): alpha=1.0 je čist MF, alpha=0.0 je čist Item-kNN")
        df_a = alpha_sweep.copy()
        for c in df_a.columns:
            df_a[c] = df_a[c].map(lambda x, c=c: f"{float(x):.2f}" if c == "alpha" else f"{float(x):.4f}")
        print(df_a.to_string(index=False))
        if chosen_alpha is not None:
            print(f"Izabrana alpha za Hybrid = {float(chosen_alpha):.2f} (max VAL_NDCG@10).")
        print("=" * 80)


# =========================
# 5) Artefakti: save / load
# =========================

def dump_reviewer_ids(payload: Payload) -> None:
    """Snima sve reviewerID i ASIN iz filtriranog dataseta u tekstualne fajlove."""
    ART_DIR.mkdir(parents=True, exist_ok=True)
    for path, key, label in (
        (REVIEWER_IDS_FILE, "user_index_to_id", "reviewerID-eva"),
        (ITEM_IDS_FILE, "item_index_to_id", "item ASIN-ova"),
    ):
        ids = list(payload["idx_maps"][key])
        path.write_text("".join(f"{x}\n" for x in ids), encoding="utf-8")
        print(f"[Saved] {len(ids)} {label} u {path}")


def save_artifacts(payload: Payload) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(ART_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"[Saved] Artefakti snimljeni u: {ART_FILE}")
    dump_reviewer_ids(payload)


def load_artifacts() -> Optional[Payload]:
    if not ART_FILE.exists():
        return None
    try:
        with open(ART_FILE, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"[Warn] Neuspešno učitavanje artefakata ({exc}). Radi se retrain.")
        return None
    print(f"[Loaded] Artefakti učitani iz: {ART_FILE}")
    if not REVIEWER_IDS_FILE.exists():  # backfill za stari pickle
        try:
            dump_reviewer_ids(payload)
        except OSError as exc:
            print(f"[Warn] Nije uspeo dump reviewerID-eva: {exc}")
    return payload


# =========================
# 6) Trening
# =========================

def prepare_agent_model(
    min_user: int = MIN_USER_INTERACTIONS,
    min_item: int = MIN_ITEM_INTERACTIONS,
    top_k: int = 50,
    mf_factors: int = 32,
    mf_epochs: int = 20,
    mf_lr: float = 0.02,
    mf_reg: float = 0.02,
    eval_sample_users: int | None = None,
) -> Payload:
    """Trenira Bias / MF / Item-kNN, bira Hybrid alpha, evaluira i bira najbolji model."""
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

    print(f"[Tune] Hybrid alpha na validaciji (grid={list(ALPHA_GRID)})...")
    alpha_table, best_alpha = tune_hybrid_alpha(payload, df_val, n_items=n_items, sample_users=500, seed=42)
    payload["hybrid_alpha"] = best_alpha
    payload["hybrid_alpha_sweep"] = alpha_table
    print(alpha_table.to_string(index=False))
    print(f"[Tune] Izabrana alpha (Hybrid) = {best_alpha:.2f} (po VAL_NDCG@10).")

    n_eval = "svim" if eval_sample_users is None else f"~{eval_sample_users}"
    print(f"[Eval] Poređenje modela na {n_eval} val/test korisnika (HR@10, NDCG@10, MRR@10)...")
    metrics = evaluate_models(payload, df_val, df_test, n_items=n_items, sample_users=eval_sample_users, seed=42)
    payload["metrics"] = metrics

    best_model = str(metrics.loc[int(metrics["TEST_NDCG@10"].astype(float).idxmax()), "model"])
    payload["best_model"] = best_model
    print(f"[Eval] Najbolji model po TEST_NDCG@10: {best_model}")
    return payload


# =========================
# 7) CLI
# =========================

def print_recommendation_block(resp: dict[str, object], top_n: int) -> None:
    print("\n" + "=" * 80)
    print(f"USER: {resp['user_id']}  |  MODEL: {resp.get('model', '?')}")
    for line in resp["explanation"]:
        print(f"- {line}")
    print("-" * 80)

    results = resp["results"]
    if not results:
        print("Nema preporuka za ovog korisnika.")
        return

    print(f"TOP-{min(top_n, len(results))} preporuka (po predviđenoj oceni):")
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
    """Top-N za prvih nekoliko poznatih korisnika i jednog cold-start korisnika."""
    sample_users = [str(u) for u in payload["idx_maps"]["user_index_to_id"][:4]] + ["NEW_USER_123"]
    for u in sample_users:
        run_single_recommendation(payload, user_id=u, top_n=top_n, model_name=model_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Amazon Recommender (Bias / MF / Item-kNN / Hybrid)")
    parser.add_argument("--retrain", action="store_true", help="Ponovo istreniraj i snimi artefakte")
    parser.add_argument("--demo", action="store_true", help="Top-N za nekoliko primernih korisnika")
    parser.add_argument("--user", type=str, default=None, help="user_id (reviewerID) za preporuke")
    parser.add_argument("--topn", type=int, default=10, help="Broj preporuka")
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_NAMES),
                        help="Izbor modela (podrazumevano: najbolji po TEST_NDCG@10)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Hybrid težina alpha iz [0, 1] (podrazumevano: izabrana na validaciji)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.topn <= 0:
        raise ValueError("--topn mora biti pozitivan broj.")
    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha mora biti u opsegu [0, 1].")

    payload = None if args.retrain else load_artifacts()
    if payload is None:
        raw_path = Path(resolve_raw_reviews_path())
        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset ne postoji: {raw_path}")
        print("[Info] Pripremam agent model (može potrajati pri prvom pokretanju)...")
        payload = prepare_agent_model()
        save_artifacts(payload)

    if "metrics" not in payload or "best_model" not in payload:
        print("[Warn] Učitani artefakti nemaju metrike. Pokrenite sa --retrain za novu evaluaciju.")
        payload.setdefault("best_model", DEFAULT_MODEL)

    if args.alpha is not None:
        payload["hybrid_alpha"] = float(args.alpha)
        print(f"[Info] Hybrid alpha override: {args.alpha:.2f}")

    print_model_comparison(payload)

    best = payload.get("best_model")
    chosen_model = args.model or str(best or DEFAULT_MODEL)
    if args.model and args.model != best:
        print(f"\n[Info] Override: koristim model '{chosen_model}' umesto best='{best}'.")
    else:
        print(f"\n[Info] Aktivni model za preporuke: {chosen_model}")

    if args.demo:
        run_demo(payload, top_n=args.topn, model_name=chosen_model)
        return
    if args.user:
        run_single_recommendation(payload, user_id=args.user, top_n=args.topn, model_name=chosen_model)
        return

    print("\nKomande:")
    print("  - Unesi user_id (reviewerID) za Top-N preporuka.")
    print("  - Prazan user_id ili 'exit' prekida.")
    while True:
        u = input("\nUnesi user_id (ili 'exit'): ").strip()
        if not u or u.lower() in ("exit", "quit", "q"):
            break
        run_single_recommendation(payload, user_id=u, top_n=args.topn, model_name=chosen_model)
    print("\nKraj.")


if __name__ == "__main__":
    try:
        main()
    except DATA_ERRORS as e:
        sys.exit(f"{type(e).__name__}: {e}")
