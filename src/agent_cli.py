from __future__ import annotations  # Omogućava forward type hints.

import sys
import os  # Rad sa env var i putanjama.
import math  # log2 za NDCG u alpha sweep-u.
import gzip  # Rad sa .gz fajlovima.
import json  # Robustan JSON parser linija.
import ast  # Fallback za stari Amazon dict-literal format.
import pickle  # Snimanje artefakata modela.
import argparse  # CLI argumenti.
from pathlib import Path
from typing import Optional  # Optional tipovi.

import numpy as np  # Numerika.
import pandas as pd  # DataFrame rad.

from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity za item-kNN.

from config import resolve_raw_reviews_path, DATA_ERRORS  # Putanja do reviews JSONL.GZ (kao i kod tebe).

# MF + ranking eval su ponovo iskorišćeni iz data_loading.py da ne bismo duplirali implementaciju.
from data_loading import (
    train_biased_mf_sgd,
    mf_score_candidates,
    evaluate_ranking,
    sample_negatives_for_user,
)


# =========================
# 1) Data pipeline (isti protokol: map, clean, filter, split, reindex)
# =========================

def load_raw_reviews() -> pd.DataFrame:  # Učitava raw dataset (robustno).
    # Tolerantni line-by-line parser: pandas + ujson padaju na neispravnim linijama
    # (npr. neescape-ovani navodnici u Amazon review tekstovima). Ovde preskačemo
    # loše linije i kao fallback probamo ast.literal_eval za stari Amazon
    # format gde su redovi Python dict literali (single quotes).
    path = str(resolve_raw_reviews_path())
    open_fn = gzip.open if path.endswith(".gz") else open

    records: list[dict] = []
    bad = 0
    with open_fn(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
                continue
            except Exception:
                pass
            try:
                obj = ast.literal_eval(line)
                if isinstance(obj, dict):
                    records.append(obj)
                    continue
            except Exception:
                pass
            bad += 1

    if bad:
        print(f"[Warn] Preskočeno {bad} loših linija pri učitavanju {path}.")
    print(f"[Info] Učitano {len(records)} redova iz {path}.")
    return pd.DataFrame.from_records(records)


def map_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:  # Mapira i čisti.
    column_mapping = {  # Amazon format.
        "reviewerID": "user_id",  # user.
        "asin": "item_id",  # item.
        "overall": "rating",  # rating.
        "unixReviewTime": "timestamp",  # time.
    }  # kraj mape.
    missing = [c for c in column_mapping if c not in df_raw.columns]  # Provera da li kolone postoje.
    if missing:  # Ako fali nešto.
        raise ValueError(f"Nedostaju kolone {missing}. Dostupno: {list(df_raw.columns)}")  # Stop.

    df = df_raw[list(column_mapping.keys())].copy()  # Uzmi samo potrebne kolone.
    df.rename(columns=column_mapping, inplace=True)  # Rename u standardna imena.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # UNIX -> datetime.
    df["rating"] = df["rating"].astype(float)  # Rating u float.

    df.dropna(subset=["user_id", "item_id", "rating", "timestamp"], inplace=True)  # Ukloni null.
    df.drop_duplicates(inplace=True)  # Ukloni duplikate.
    return df  # Vrati očišćen DF.


def iterative_filter_min_interactions(  # Stabilno filtriranje.
    df: pd.DataFrame,  # Interakcije.
    min_user: int = 3,  # Min po useru.
    min_item: int = 4,  # Min po itemu.
    max_iters: int = 20,  # Maks iteracija.
) -> pd.DataFrame:  # Vraća filtriran DF.
    out = df.copy()  # Kopija.
    for _ in range(max_iters):  # Iteracije.
        before = len(out)  # Pre.
        out = out.groupby("user_id").filter(lambda x: len(x) >= min_user)  # User filter.
        out = out.groupby("item_id").filter(lambda x: len(x) >= min_item)  # Item filter.
        after = len(out)  # Posle.
        if after == before:  # Stabilno.
            break  # Stop.
    return out  # Vrati.


def build_leave_one_out_splits(df_cut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # LOO split.
    df_cut = df_cut.sort_values(["user_id", "timestamp"], ascending=[True, True]).copy()  # Sort po user+time.
    n = df_cut.groupby("user_id")["rating"].transform("size")  # Broj interakcija po useru.
    r = df_cut.groupby("user_id").cumcount()  # Redni broj u grupi.
    df_cut["split"] = "train"  # Default train.
    df_cut.loc[r == (n - 1), "split"] = "test"  # Poslednja -> test.
    df_cut.loc[r == (n - 2), "split"] = "val"  # Pretposlednja -> val.
    df_train = df_cut[df_cut["split"] == "train"].reset_index(drop=True)  # Train.
    df_val = df_cut[df_cut["split"] == "val"].reset_index(drop=True)  # Val.
    df_test = df_cut[df_cut["split"] == "test"].reset_index(drop=True)  # Test.
    return df_train, df_val, df_test  # Vrati tri DF-a.


def reindex_after_filter(df_cut: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:  # Reindeksiranje.
    df_cut = df_cut.copy()  # Copy.
    user_cats = df_cut["user_id"].astype("category")  # Category za user.
    item_cats = df_cut["item_id"].astype("category")  # Category za item.
    df_cut["user_index"] = user_cats.cat.codes.astype(int)  # user_id -> user_index.
    df_cut["item_index"] = item_cats.cat.codes.astype(int)  # item_id -> item_index.

    idx_maps = {  # Mapiranja za kasniji povrat.
        "user_index_to_id": list(user_cats.cat.categories),  # index -> user_id.
        "item_index_to_id": list(item_cats.cat.categories),  # index -> item_id.
    }  # kraj.
    return df_cut, idx_maps  # Vrati.


# =========================
# 2) Item-kNN (Adjusted Cosine) + Bias fallback (μ + bu + bi)
# =========================

def fit_bias(df_train: pd.DataFrame, n_users: int, n_items: int, reg: float = 10.0, n_iters: int = 20) -> tuple[float, np.ndarray, np.ndarray]:  # Bias fit.
    mu = float(df_train["rating"].mean())  # Global mean.
    bu = np.zeros(n_users, dtype=np.float32)  # user bias.
    bi = np.zeros(n_items, dtype=np.float32)  # item bias.
    ug = df_train.groupby("user_index")  # po useru.
    ig = df_train.groupby("item_index")  # po itemu.

    for _ in range(n_iters):  # iteracije.
        for u, g in ug:  # update bu.
            items = g["item_index"].to_numpy(dtype=int)  # item indeksi.
            resid = g["rating"].to_numpy(dtype=float) - mu - bi[items]  # residual.
            bu[int(u)] = float(resid.sum() / (reg + len(g)))  # closed form.
        for i, g in ig:  # update bi.
            users = g["user_index"].to_numpy(dtype=int)  # user indeksi.
            resid = g["rating"].to_numpy(dtype=float) - mu - bu[users]  # residual.
            bi[int(i)] = float(resid.sum() / (reg + len(g)))  # closed form.

    return mu, bu, bi  # Vrati.


def build_user_mean(df_train: pd.DataFrame, n_users: int) -> np.ndarray:  # User mean za adj cosine.
    mu = float(df_train["rating"].mean())  # Global mean.
    arr = np.full(n_users, mu, dtype=np.float32)  # Default.
    um = df_train.groupby("user_index")["rating"].mean()  # user mean.
    arr[um.index.to_numpy(int)] = um.to_numpy(np.float32)  # Fill.
    return arr  # Vrati.


def build_user_history(df_train: pd.DataFrame) -> dict[int, dict[int, float]]:  # user->(item->rating).
    hist: dict[int, dict[int, float]] = {}  # init.
    for u, i, r in zip(  # iter kroz train.
        df_train["user_index"].to_numpy(int),
        df_train["item_index"].to_numpy(int),
        df_train["rating"].to_numpy(float),
    ):
        hist.setdefault(int(u), {})[int(i)] = float(r)  # upiši.
    return hist  # Vrati.


def compute_item_topk_adjcos(
    df_train: pd.DataFrame,
    n_users: int,
    n_items: int,
    user_mean: np.ndarray,
    top_k: int = 50,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:  # Precompute topK suseda po itemu.
    # Pivot user×item za devijacije (rating - user_mean), pa transponuj u item×user.
    # Ovo je najjednostavnije za implementaciju; za veoma velike matrice može biti teško,
    # ali posle filtriranja kod tebe je n_items ~ nekoliko hiljada, pa je izvodljivo.
    df_train_local = df_train[["user_index", "item_index", "rating"]].copy()  # kopija.
    df_train_local["dev"] = df_train_local["rating"].to_numpy(float) - user_mean[df_train_local["user_index"].to_numpy(int)]  # dev.

    # Napravi user×item dev matricu.
    user_item = df_train_local.pivot_table(index="user_index", columns="item_index", values="dev", aggfunc="mean")  # dev pivot.
    user_item = user_item.reindex(index=np.arange(n_users), columns=np.arange(n_items))  # poravnaj indekse.
    user_item_filled = user_item.fillna(0.0).to_numpy(dtype=np.float32)  # NaN->0.

    # Item×user matrica za cosine.
    item_user = user_item_filled.T  # transpose.

    neigh_idx = np.zeros((n_items, top_k), dtype=np.int32)  # sused indeksi.
    neigh_sim = np.zeros((n_items, top_k), dtype=np.float32)  # sused sim.

    for start in range(0, n_items, block_size):  # blokovi itema.
        end = min(start + block_size, n_items)  # kraj.
        sims_block = cosine_similarity(item_user[start:end], item_user)  # cosine sim (block×all).
        for r in range(sims_block.shape[0]):  # svaki red u bloku.
            item_i = start + r  # global item.
            sims_row = sims_block[r]  # sim prema svim itemima.
            sims_row[item_i] = -np.inf  # ukloni self.
            top = np.argpartition(-sims_row, top_k)[:top_k]  # topK kandidati (ne-sort).
            top = top[np.argsort(-sims_row[top])]  # sortiraj opadajuće.
            neigh_idx[item_i, :] = top.astype(np.int32)  # snimi indekse.
            neigh_sim[item_i, :] = sims_row[top].astype(np.float32)  # snimi sim.

    return neigh_idx, neigh_sim  # Vrati topK po itemu.


def predict_item_knn(
    u_idx: int,
    i_idx: int,
    user_hist: dict[int, dict[int, float]],
    user_mean: np.ndarray,
    neigh_idx: np.ndarray,
    neigh_sim: np.ndarray,
    fallback: float,
) -> float:  # Predikcija rating-a za (u,i).
    if u_idx not in user_hist:  # Cold user.
        return float(fallback)  # fallback.
    rated = user_hist[u_idx]  # item->rating.
    u_mean = float(user_mean[u_idx])  # user mean.
    num = 0.0  # brojilac.
    den = 0.0  # imenilac.
    for j, s in zip(neigh_idx[i_idx], neigh_sim[i_idx]):  # susedi.
        jj = int(j)  # sused item id.
        ss = float(s)  # sim.
        if not np.isfinite(ss):  # preskoči -inf.
            continue  # next.
        if jj in rated:  # ako user ocenio sused.
            num += ss * (float(rated[jj]) - u_mean)  # ponder dev.
            den += abs(ss)  # sum težina.
    if den == 0.0:  # nema poklapanja.
        return float(fallback)  # fallback.
    pred = u_mean + (num / den)  # u_mean + weighted dev.
    return float(np.clip(pred, 1.0, 5.0))  # clip u [1,5].


# =========================
# 3) Score funkcije po modelu (Bias / MF / Item-kNN)
# =========================

def _score_bias(u_idx: int, cand: np.ndarray, mu: float, bu: np.ndarray, bi: np.ndarray) -> np.ndarray:
    """Score = μ + bu[u] + bi[cand] (clipped to [1, 5])."""
    raw = float(mu) + float(bu[u_idx]) + bi[np.asarray(cand, dtype=int)]
    return np.clip(raw, 1.0, 5.0).astype(float)


def _score_mf(u_idx: int, cand: np.ndarray, mf: dict[str, object]) -> np.ndarray:
    """MF score = μ + bu[u] + bi[i] + P[u] · Q[i]."""
    return mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], int(u_idx), np.asarray(cand, dtype=int))


def _score_itemknn(
    u_idx: int,
    cand: np.ndarray,
    mu: float,
    bu: np.ndarray,
    bi: np.ndarray,
    user_mean: np.ndarray,
    user_hist: dict[int, dict[int, float]],
    neigh_idx: np.ndarray,
    neigh_sim: np.ndarray,
) -> np.ndarray:
    """Item-kNN score sa bias fallback-om kada nema overlap-a sa korisnikom."""
    out = np.zeros(len(cand), dtype=float)
    for t, it in enumerate(cand):
        ii = int(it)
        fb = float(np.clip(mu + bu[u_idx] + bi[ii], 1.0, 5.0))
        out[t] = predict_item_knn(int(u_idx), ii, user_hist, user_mean, neigh_idx, neigh_sim, fb)
    return out


def _score_hybrid(
    u_idx: int,
    cand: np.ndarray,
    alpha: float,
    mu: float,
    bu: np.ndarray,
    bi: np.ndarray,
    mf: dict[str, object],
    user_mean: np.ndarray,
    user_hist: dict[int, dict[int, float]],
    neigh_idx: np.ndarray,
    neigh_sim: np.ndarray,
) -> np.ndarray:
    """Hybrid score = α * MF + (1 − α) * Item-kNN.

    α=1.0 → čist MF, α=0.0 → čist Item-kNN. MF nosi globalne (latentne) obrasce,
    Item-kNN lokalne (item-item) obrasce; linearni blend često ulovi oba signala.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    mf_scores = _score_mf(u_idx, cand, mf)
    knn_scores = _score_itemknn(u_idx, cand, mu, bu, bi, user_mean, user_hist, neigh_idx, neigh_sim)
    return (a * mf_scores + (1.0 - a) * knn_scores).astype(float)


def make_score_fn(model_name: str, payload: dict[str, object]):
    """Vraća score_fn(u_idx, cand_array) -> np.ndarray za zadati model.

    Koristi se i za ranking evaluaciju (evaluate_ranking) i za skoringe pri preporuci.
    """
    mu = payload["mu"]
    bu = payload["bu"]
    bi = payload["bi"]
    if model_name == "Bias":
        return lambda u, cand: _score_bias(int(u), cand, mu, bu, bi)
    if model_name == "MF":
        mf = payload["mf"]
        return lambda u, cand: _score_mf(int(u), cand, mf)
    if model_name == "Item-kNN":
        user_mean = payload["user_mean"]
        user_hist = payload["user_hist"]
        neigh_idx = payload["neigh_idx"]
        neigh_sim = payload["neigh_sim"]
        return lambda u, cand: _score_itemknn(int(u), cand, mu, bu, bi, user_mean, user_hist, neigh_idx, neigh_sim)
    if model_name == "Hybrid":
        mf = payload["mf"]
        user_mean = payload["user_mean"]
        user_hist = payload["user_hist"]
        neigh_idx = payload["neigh_idx"]
        neigh_sim = payload["neigh_sim"]
        alpha = float(payload.get("hybrid_alpha", 0.5))
        return lambda u, cand: _score_hybrid(
            int(u), cand, alpha, mu, bu, bi, mf, user_mean, user_hist, neigh_idx, neigh_sim,
        )
    raise ValueError(f"Unknown model: {model_name}")


# =========================
# 4) Objašnjenja preporuka po modelu
# =========================

def explain_item_knn(
    u_idx: int,
    i_idx: int,
    user_hist: dict[int, dict[int, float]],
    user_mean: np.ndarray,
    neigh_idx: np.ndarray,
    neigh_sim: np.ndarray,
    item_index_to_asin: list[str],
    top_m: int = 3,
) -> list[str]:
    if u_idx not in user_hist:
        return ["Nemamo istoriju ocena za ovog korisnika, pa je preporuka zasnovana na opštem modelu (fallback)."]

    rated = user_hist[u_idx]
    u_mean = float(user_mean[u_idx])
    contribs: list[tuple[float, str]] = []

    for j, s in zip(neigh_idx[i_idx], neigh_sim[i_idx]):
        jj = int(j)
        ss = float(s)
        if not np.isfinite(ss):
            continue
        if jj in rated:
            dev = float(rated[jj]) - u_mean
            weight = ss * dev
            asin_j = item_index_to_asin[jj]
            contribs.append((abs(weight), f"Zato što ste visoko ocenili '{asin_j}' (sličnost={ss:.3f})."))

    contribs.sort(key=lambda x: x[0], reverse=True)
    msgs = [m for _, m in contribs[:top_m]]
    if not msgs:
        msgs = ["Preporuka je zasnovana na sličnim proizvodima, ali nismo našli direktne slične stavke iz vaše istorije (fallback na opšti signal)."]
    return msgs


def _explain_for_model(model_name: str, u_idx: int, i_idx: int, payload: dict[str, object]) -> list[str]:
    """Vrati listu objašnjenja zašto je item preporučen, prilagođenu modelu."""
    if model_name == "Item-kNN":
        return explain_item_knn(
            u_idx,
            i_idx,
            payload["user_hist"],
            payload["user_mean"],
            payload["neigh_idx"],
            payload["neigh_sim"],
            payload["idx_maps"]["item_index_to_id"],
            top_m=3,
        )

    if model_name == "MF":
        mf = payload["mf"]
        p_u = mf["P"][u_idx]
        q_i = mf["Q"][i_idx]
        dot = float(np.dot(p_u, q_i))
        mu_v = float(mf["mu"])
        bu_v = float(mf["bu"][u_idx])
        bi_v = float(mf["bi"][i_idx])
        pred = mu_v + bu_v + bi_v + dot
        return [
            f"MF (Matrix Factorization): predviđena ocena = μ + bu + bi + P_u·Q_i = {mu_v:.3f} + {bu_v:+.3f} + {bi_v:+.3f} + {dot:+.3f} ≈ {pred:.3f}.",
            f"Doprinos latentnih faktora P_u · Q_i = {dot:+.3f} (pozitivno = vaš profil 'leži' u istom pravcu kao item).",
            f"Item bias bi = {bi_v:+.3f} (koliko ovaj item generalno odstupa od globalnog proseka).",
        ]

    if model_name == "Bias":
        mu_v = float(payload["mu"])
        bu_v = float(payload["bu"][u_idx])
        bi_v = float(payload["bi"][i_idx])
        pred = mu_v + bu_v + bi_v
        return [
            f"Bias model: predikcija = μ + bu + bi = {mu_v:.3f} + {bu_v:+.3f} + {bi_v:+.3f} ≈ {pred:.3f}.",
            f"Item ima visok item bias bi = {bi_v:+.3f} (generalno ocenjen iznad globalnog proseka).",
            f"Vaš user bias bu = {bu_v:+.3f} (vaša tendencija da ocenjujete iznad/ispod μ).",
            "Napomena: Bias model nije personalizovan po sadržaju — rangira sve korisnike po istim 'globalno najboljim' itemima.",
        ]

    if model_name == "Hybrid":
        alpha = float(payload.get("hybrid_alpha", 0.5))
        mf = payload["mf"]
        # MF doprinos
        p_u = mf["P"][u_idx]
        q_i = mf["Q"][i_idx]
        dot = float(np.dot(p_u, q_i))
        mf_pred = float(mf["mu"]) + float(mf["bu"][u_idx]) + float(mf["bi"][i_idx]) + dot
        # Item-kNN doprinos (sa bias fallback-om — kao u skoringu)
        mu_v = float(payload["mu"])
        bu_v = float(payload["bu"][u_idx])
        bi_v = float(payload["bi"][i_idx])
        fb = float(np.clip(mu_v + bu_v + bi_v, 1.0, 5.0))
        knn_pred = float(predict_item_knn(
            int(u_idx), int(i_idx),
            payload["user_hist"], payload["user_mean"],
            payload["neigh_idx"], payload["neigh_sim"],
            fb,
        ))
        blend = alpha * mf_pred + (1.0 - alpha) * knn_pred
        return [
            f"Hybrid model: α={alpha:.2f} · MF + (1−α) · Item-kNN.",
            f"  MF predikcija = {mf_pred:.3f} (latentni faktori P_u·Q_i = {dot:+.3f}).",
            f"  Item-kNN predikcija = {knn_pred:.3f} (sused-based ili bias fallback).",
            f"  Kombinovana predikcija = {alpha:.2f}·{mf_pred:.3f} + {1.0-alpha:.2f}·{knn_pred:.3f} ≈ {blend:.3f}.",
        ]

    return ["Preporuka je generisana izabranim modelom."]


# =========================
# 5) Top-N preporuke za korisnika (model-agnostic)
# =========================

def recommend_top_n_for_user(
    user_id: str,
    payload: dict[str, object],
    model_name: Optional[str] = None,
    top_n: int = 10,
) -> dict[str, object]:
    """Računa Top-N preporuka za korisnika koristeći zadati (ili 'best') model."""
    if model_name is None:
        model_name = str(payload.get("best_model", "Item-kNN"))

    idx_maps = payload["idx_maps"]
    user_index_to_id = idx_maps["user_index_to_id"]
    item_index_to_asin = idx_maps["item_index_to_id"]
    user_hist = payload.get("user_hist", {})

    try:
        u_idx = user_index_to_id.index(user_id)
        known_user = True
    except ValueError:
        u_idx = -1
        known_user = False

    n_items = len(item_index_to_asin)

    if known_user and u_idx in user_hist:
        rated_items = set(user_hist[u_idx].keys())
        candidates = np.array([i for i in range(n_items) if i not in rated_items], dtype=int)
    else:
        candidates = np.arange(n_items, dtype=int)

    if known_user:
        score_fn = make_score_fn(model_name, payload)
        scores = np.asarray(score_fn(u_idx, candidates), dtype=float)
    else:
        # Cold-start: bez user-specific signala koristimo μ + item bias.
        mu = float(payload["mu"])
        bi = payload["bi"]
        scores = np.clip(mu + bi[candidates], 1.0, 5.0).astype(float)

    order = np.argsort(-scores, kind="stable")[:top_n]
    top_items = candidates[order]
    top_scores = scores[order]

    results = [
        {"asin": item_index_to_asin[int(i)], "score": float(s), "item_index": int(i)}
        for i, s in zip(top_items, top_scores)
    ]

    explanation: list[str] = [f"Korišćeni model: {model_name}"]
    if not known_user:
        explanation.append("Korisnik nije pronađen u datasetu → cold-start: preporuke su zasnovane na opštem signalu (μ + item bias).")
    elif model_name == "Item-kNN":
        explanation.append("Korisnik pronađen → Item-kNN (Adjusted Cosine) sa bias fallback-om.")
    elif model_name == "MF":
        explanation.append("Korisnik pronađen → Matrix Factorization (Biased SGD): predikcija na osnovu latentnih faktora P_u, Q_i.")
    elif model_name == "Bias":
        explanation.append("Korisnik pronađen → Bias model (μ + bu + bi). Napomena: rangira sve korisnike po istim 'globalno najboljim' itemima.")
    elif model_name == "Hybrid":
        a = float(payload.get("hybrid_alpha", 0.5))
        explanation.append(f"Korisnik pronađen → Hybrid (α={a:.2f}·MF + (1−α)·Item-kNN): kombinovani globalni i lokalni signal.")

    item_explanations: dict[str, list[str]] = {}
    n_explain = min(5, len(results))
    if known_user:
        for r in results[:n_explain]:
            i_idx = int(r["item_index"])
            item_explanations[r["asin"]] = _explain_for_model(model_name, u_idx, i_idx, payload)
    else:
        for r in results[:n_explain]:
            item_explanations[r["asin"]] = ["Preporuka je izabrana na osnovu opšte popularnosti/ocena (cold-start fallback)."]

    return {
        "user_id": user_id,
        "model": model_name,
        "known_user": known_user,
        "results": results,
        "explanation": explanation,
        "item_explanations": item_explanations,
    }


# =========================
# 6) Evaluacija modela (Top-N ranking)
# =========================

EVAL_MODELS = ("Bias", "MF", "Item-kNN", "Hybrid")
ALPHA_GRID = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)


def tune_hybrid_alpha(
    payload: dict[str, object],
    df_val: pd.DataFrame,
    n_items: int,
    alphas: tuple[float, ...] = ALPHA_GRID,
    sample_users: int = 500,
    n_neg: int = 99,
    k: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """Tune α na validacionom setu sa zajedničkim negativima (efikasno).

    Strategija:
    1) Uzmi uzorak od `sample_users` validacionih (user, pos_item) parova.
    2) Za svakog usera, sample-uj 99 negativa JEDNOM, zatim izračunaj MF i Item-kNN
       skorove za tih 100 kandidata samo jednom.
    3) Za svaki α u gridu, samo blendiraj već keširane skorove i izračunaj NDCG@k.

    Tako alpha sweep košta otprilike kao jedan Item-kNN ranking pass.
    """
    rng = np.random.default_rng(seed)
    n_val = min(int(sample_users), len(df_val))
    df_val_s = df_val.sample(n=n_val, random_state=seed)

    df_cut = payload["df_cut"]
    user_interactions: dict[int, set[int]] = {}
    for u, i in zip(df_cut["user_index"].to_numpy(int), df_cut["item_index"].to_numpy(int)):
        user_interactions.setdefault(int(u), set()).add(int(i))

    all_items = np.arange(n_items, dtype=int)
    score_mf_fn = make_score_fn("MF", payload)
    score_knn_fn = make_score_fn("Item-kNN", payload)

    cached: list[tuple[np.ndarray, np.ndarray]] = []
    for _, row in df_val_s.iterrows():
        u = int(row["user_index"])
        pos_item = int(row["item_index"])
        interacted = user_interactions.get(u, set())
        neg_items = sample_negatives_for_user(all_items, interacted, pos_item, n_neg, rng)
        cand = np.concatenate([[pos_item], neg_items]).astype(int)
        mf_scores = np.asarray(score_mf_fn(u, cand), dtype=float)
        knn_scores = np.asarray(score_knn_fn(u, cand), dtype=float)
        cached.append((mf_scores, knn_scores))

    rows = []
    for a in alphas:
        ndcgs: list[float] = []
        hrs: list[float] = []
        mrrs: list[float] = []
        rank_rng = np.random.default_rng(seed + 1000)  # za tie-breaking determinizam
        for mf_s, knn_s in cached:
            scores = float(a) * mf_s + (1.0 - float(a)) * knn_s
            pos_score = float(scores[0])
            higher = int(np.sum(scores > pos_score))
            ties = int(np.sum(scores == pos_score))
            tie_offset = int(rank_rng.integers(low=1, high=ties + 1))
            rnk = higher + tie_offset
            hrs.append(1.0 if rnk <= k else 0.0)
            ndcgs.append(0.0 if rnk > k else 1.0 / math.log2(rnk + 1))
            mrrs.append(0.0 if rnk > k else 1.0 / float(rnk))
        rows.append({
            "alpha": float(a),
            f"VAL_HR@{k}": float(np.mean(hrs)),
            f"VAL_NDCG@{k}": float(np.mean(ndcgs)),
            f"VAL_MRR@{k}": float(np.mean(mrrs)),
        })

    df_alpha = pd.DataFrame(rows).sort_values(f"VAL_NDCG@{k}", ascending=False).reset_index(drop=True)
    best_alpha = float(df_alpha.iloc[0]["alpha"])
    return df_alpha, best_alpha


def evaluate_models(
    payload: dict[str, object],
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    n_items: int,
    sample_users: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Poredi modele po HR@10 / NDCG@10 / MRR@10 (uzorak val/test korisnika).

    Koristi 1 pozitivan + 99 negativa po korisniku (standardni LOO Top-N protokol).
    Uzorkujemo `sample_users` redova iz val/test radi brzine (Item-kNN je spor).
    """
    if sample_users is None or int(sample_users) >= len(df_val):
        df_val_s = df_val
    else:
        df_val_s = df_val.sample(n=int(sample_users), random_state=seed)
    if sample_users is None or int(sample_users) >= len(df_test):
        df_test_s = df_test
    else:
        df_test_s = df_test.sample(n=int(sample_users), random_state=seed + 1)

    df_val_pos = df_val_s[["user_index", "item_index"]].copy()
    df_test_pos = df_test_s[["user_index", "item_index"]].copy()

    df_cut = payload["df_cut"]
    user_interactions: dict[int, set[int]] = {}
    for u, i in zip(df_cut["user_index"].to_numpy(int), df_cut["item_index"].to_numpy(int)):
        user_interactions.setdefault(int(u), set()).add(int(i))

    all_items = np.arange(n_items, dtype=int)

    rows = []
    for model_name in EVAL_MODELS:
        score_fn = make_score_fn(model_name, payload)
        val_m = evaluate_ranking(df_val_pos, user_interactions, all_items, score_fn, k=10, n_neg=99, seed=seed)
        test_m = evaluate_ranking(df_test_pos, user_interactions, all_items, score_fn, k=10, n_neg=99, seed=seed + 1)
        rows.append({
            "model": model_name,
            "VAL_HR@10": val_m["HR@10"],
            "VAL_NDCG@10": val_m["NDCG@10"],
            "VAL_MRR@10": val_m["MRR@10"],
            "TEST_HR@10": test_m["HR@10"],
            "TEST_NDCG@10": test_m["NDCG@10"],
            "TEST_MRR@10": test_m["MRR@10"],
        })

    return pd.DataFrame(rows)


def print_model_comparison(payload: dict[str, object]) -> None:
    """Pretty-print metrics table + best-model line."""
    metrics = payload.get("metrics")
    print("\n" + "=" * 80)
    print("Poređenje modela (Top-N ranking metrike, uzorak validacionih i test korisnika)")
    print("=" * 80)
    if metrics is None or len(metrics) == 0:
        print("(Metrike nisu dostupne — pokrenite ponovo sa --retrain.)")
        print("=" * 80)
        return

    cols = ["model", "VAL_HR@10", "VAL_NDCG@10", "VAL_MRR@10", "TEST_HR@10", "TEST_NDCG@10", "TEST_MRR@10"]
    df_show = metrics[cols].copy()
    for c in cols[1:]:
        df_show[c] = df_show[c].map(lambda x: f"{float(x):.4f}")
    print(df_show.to_string(index=False))
    print("-" * 80)

    best = str(payload.get("best_model", ""))
    model_set = set(metrics["model"].astype(str))
    if best and best in model_set:
        best_row = metrics[metrics["model"] == best].iloc[0]
        print(f"Najbolji model po TEST_NDCG@10: {best} (TEST_NDCG@10={float(best_row['TEST_NDCG@10']):.4f})")
        print(
            "Razlog izbora: NDCG@10 meri koliko su pozitivni iteme rangirani visoko među 99 nasumičnih "
            "negativa — to je direktna metrika kvaliteta Top-N preporuka, za razliku od RMSE koji meri "
            "tačnost predikcije skalarne ocene i obično favorizuje konzervativne (bias-based) modele."
        )
    print("=" * 80)

    alpha_sweep = payload.get("hybrid_alpha_sweep")
    chosen_alpha = payload.get("hybrid_alpha")
    if isinstance(alpha_sweep, pd.DataFrame) and len(alpha_sweep) > 0:
        print("\nHybrid α tuning (val): α=1.0 → čisti MF, α=0.0 → čisti Item-kNN")
        df_a = alpha_sweep.copy()
        for c in df_a.columns:
            if c != "alpha":
                df_a[c] = df_a[c].map(lambda x: f"{float(x):.4f}")
            else:
                df_a[c] = df_a[c].map(lambda x: f"{float(x):.2f}")
        print(df_a.to_string(index=False))
        if chosen_alpha is not None:
            print(f"Izabrana α za Hybrid = {float(chosen_alpha):.2f} (max VAL_NDCG@10).")
        print("=" * 80)


# =========================
# 7) Artefakti: save/load (da ne treniraš svaki put)
# =========================

ART_DIR = Path(__file__).resolve().parent / "agent_artifacts"  # Folder za artefakte.
ART_FILE = ART_DIR / "agent_itemknn.pkl"  # Pickle fajl.
REVIEWER_IDS_FILE = ART_DIR / "reviewer_ids.txt"  # Spisak svih reviewerID-eva u modelu.
ITEM_IDS_FILE = ART_DIR / "item_ids.txt"  # Spisak svih ASIN-ova u modelu.


def dump_reviewer_ids(payload: dict[str, object]) -> None:
    """Snima sve reviewerID iz aktivnog (filtriranog) dataset-a u tekstualni fajl.

    Korisno za: brzu proveru da li je neki user u modelu i kao izvor user_id-eva
    za testiranje preporuka (random sample, demo skripte, itd).
    """
    ART_DIR.mkdir(parents=True, exist_ok=True)
    user_ids = list(payload["idx_maps"]["user_index_to_id"])
    item_ids = list(payload["idx_maps"]["item_index_to_id"])

    with open(REVIEWER_IDS_FILE, "w", encoding="utf-8") as f:
        for uid in user_ids:
            f.write(f"{uid}\n")
    print(f"[Saved] {len(user_ids)} reviewerID-eva → {REVIEWER_IDS_FILE}")

    with open(ITEM_IDS_FILE, "w", encoding="utf-8") as f:
        for iid in item_ids:
            f.write(f"{iid}\n")
    print(f"[Saved] {len(item_ids)} item ASIN-ova → {ITEM_IDS_FILE}")


def save_artifacts(payload: dict[str, object]) -> None:  # Snimanje.
    ART_DIR.mkdir(parents=True, exist_ok=True)  # Kreiraj folder.
    with open(ART_FILE, "wb") as f:  # Otvori binarno.
        pickle.dump(payload, f)  # Dump.
    print(f"[Saved] Artefakti snimljeni u: {ART_FILE}")  # Log.
    dump_reviewer_ids(payload)  # Spisak reviewerID-eva za brzu proveru / testiranje.

def load_artifacts() -> Optional[dict[str, object]]:  # Učitavanje.
    if not ART_FILE.exists():  # Ako ne postoji.
        return None  # Nema.
    try:
        with open(ART_FILE, "rb") as f:  # Otvori.
            payload = pickle.load(f)  # Load.
    except Exception as exc:
        print(f"[Warn] Neuspešno učitavanje artefakata ({exc}). Radi se retrain.")  # Log.
        return None
    print(f"[Loaded] Artefakti učitani iz: {ART_FILE}")  # Log.
    if not REVIEWER_IDS_FILE.exists():  # Backfill ako stari pickle nema dump.
        try:
            dump_reviewer_ids(payload)
        except Exception as exc:
            print(f"[Warn] Nije uspeo dump reviewerID-eva: {exc}")
    return payload  # Vrati.


# =========================
# 8) Trening / priprema modela za agent
# =========================

def prepare_agent_model(
    min_user: int = 3,
    min_item: int = 4,
    top_k: int = 50,
    mf_factors: int = 32,
    mf_epochs: int = 20,
    mf_lr: float = 0.02,
    mf_reg: float = 0.02,
    eval_sample_users: int | None = None,
) -> dict[str, object]:
    """Trenira Bias / MF / Item-kNN, evaluira ih, bira najbolji za Top-N preporuke."""
    df_raw = load_raw_reviews()
    df = map_and_clean(df_raw)
    df = df.sort_values("timestamp", ascending=True)

    df_cut = iterative_filter_min_interactions(df, min_user=min_user, min_item=min_item, max_iters=20)
    df_cut, idx_maps = reindex_after_filter(df_cut)

    df_train, df_val, df_test = build_leave_one_out_splits(df_cut)
    assert df_val["user_index"].isin(df_train["user_index"]).all()
    assert df_test["user_index"].isin(df_train["user_index"]).all()

    n_users = int(df_cut["user_index"].nunique())
    n_items = int(df_cut["item_index"].nunique())
    print(f"[Data] users={n_users}, items={n_items}, train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    print("[Train] Bias model (μ + bu + bi)...")
    mu, bu, bi = fit_bias(df_train, n_users=n_users, n_items=n_items, reg=10.0, n_iters=20)
    user_mean = build_user_mean(df_train, n_users=n_users)
    user_hist = build_user_history(df_train)

    print(f"[Train] Item-kNN (Adjusted Cosine, top_k={top_k})...")
    neigh_idx, neigh_sim = compute_item_topk_adjcos(
        df_train, n_users=n_users, n_items=n_items, user_mean=user_mean, top_k=top_k, block_size=256,
    )

    print(f"[Train] MF (Biased SGD, factors={mf_factors}, epochs={mf_epochs})...")
    mf = train_biased_mf_sgd(
        df_train, df_val, n_users=n_users, n_items=n_items,
        factors=mf_factors, lr=mf_lr, reg=mf_reg, n_epochs=mf_epochs, seed=42,
    )

    payload: dict[str, object] = {
        "df_cut": df_cut,
        "idx_maps": idx_maps,
        "mu": mu,
        "bu": bu,
        "bi": bi,
        "user_mean": user_mean,
        "user_hist": user_hist,
        "neigh_idx": neigh_idx,
        "neigh_sim": neigh_sim,
        "mf": mf,
        "min_user": min_user,
        "min_item": min_item,
        "top_k": top_k,
        "mf_factors": mf_factors,
        "mf_epochs": mf_epochs,
    }

    print(f"[Tune] Hybrid α sweep na validaciji (alpha grid={list(ALPHA_GRID)})...")
    alpha_table, best_alpha = tune_hybrid_alpha(
        payload, df_val, n_items=n_items, alphas=ALPHA_GRID, sample_users=500, seed=42,
    )
    payload["hybrid_alpha"] = best_alpha
    payload["hybrid_alpha_sweep"] = alpha_table
    print(alpha_table.to_string(index=False))
    print(f"[Tune] Izabrana α (Hybrid) = {best_alpha:.2f} (po VAL_NDCG@10).")

    n_eval = "svim" if eval_sample_users is None else f"~{eval_sample_users}"
    print(f"[Eval] Poređenje modela na {n_eval} val/test korisnika (HR@10, NDCG@10, MRR@10)...")
    metrics = evaluate_models(payload, df_val, df_test, n_items=n_items, sample_users=eval_sample_users, seed=42)
    payload["metrics"] = metrics

    best_idx = int(metrics["TEST_NDCG@10"].astype(float).idxmax())
    best_model = str(metrics.loc[best_idx, "model"])
    payload["best_model"] = best_model
    print(f"[Eval] Najbolji model po TEST_NDCG@10: {best_model}")

    return payload


# =========================
# 9) CLI
# =========================

def run_single_recommendation(
    payload: dict[str, object],
    user_id: str,
    top_n: int,
    model_name: Optional[str] = None,
) -> None:
    """Compute Top-N for one user and print a formatted block."""
    resp = recommend_top_n_for_user(
        user_id=str(user_id),
        payload=payload,
        model_name=model_name,
        top_n=int(top_n),
    )
    print_recommendation_block(resp, top_n=int(top_n))


def print_recommendation_block(resp: dict[str, object], top_n: int) -> None:
    print("\n" + "=" * 80)
    print(f"USER: {resp['user_id']}  |  MODEL: {resp.get('model', '?')}")
    for line in resp["explanation"]:
        print(f"- {line}")
    print("-" * 80)

    results = resp["results"]
    item_expl = resp["item_explanations"]

    if not results:
        print("Nema preporuka za ovog korisnika.")
        return

    print(f"TOP-{min(top_n, len(results))} preporuka (po predviđenoj oceni):")
    for idx, r in enumerate(results[:top_n], start=1):
        print(f"{idx:02d}. ASIN={r['asin']} | predicted_rating={r['score']:.3f}")
        for ex in item_expl.get(r["asin"], [])[:3]:
            print(f"    • {ex}")

    print("=" * 80)


def run_demo(
    agent_payload: dict[str, object],
    top_n: int = 10,
    model_name: Optional[str] = None,
) -> None:
    """Sample Top-N for the first few known users + one cold-start user."""
    user_ids = agent_payload["idx_maps"]["user_index_to_id"]
    sample_users = [str(u) for u in user_ids[:4]] if user_ids else []
    sample_users.append("NEW_USER_123")
    for u in sample_users:
        run_single_recommendation(agent_payload, user_id=u, top_n=top_n, model_name=model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon Recommender (multi-model: Bias / MF / Item-kNN)")
    parser.add_argument("--retrain", action="store_true", help="Force retrain and save artifacts")
    parser.add_argument("--demo", action="store_true", help="Run Top-N for a few example users")
    parser.add_argument("--user", type=str, default=None, help="user_id (reviewerID) for recommendations")
    parser.add_argument("--topn", type=int, default=10, help="Top-N recommendations")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[None, "Bias", "MF", "Item-kNN", "Hybrid"],
        help="Override active model (default: best by TEST_NDCG@10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override Hybrid blend weight α∈[0,1] (default: auto-tuned on validation)",
    )
    args = parser.parse_args()

    if args.topn <= 0:
        raise ValueError("--topn mora biti pozitivan broj.")
    if args.alpha is not None and not (0.0 <= float(args.alpha) <= 1.0):
        raise ValueError("--alpha mora biti u opsegu [0, 1].")

    if not Path(resolve_raw_reviews_path()).exists() and args.retrain:
        raise FileNotFoundError(f"RAW_REVIEWS_PATH ne postoji: {resolve_raw_reviews_path()}")

    payload = None if args.retrain else load_artifacts()
    if payload is None:
        print("[Info] Pripremam agent model (može potrajati pri prvom pokretanju)...")
        payload = prepare_agent_model(min_user=4, min_item=2, top_k=50)
        save_artifacts(payload)

    # Bezbedno: ako neko učita stari pickle bez metrika, ne raspadaj.
    if "metrics" not in payload or "best_model" not in payload:
        print("[Warn] Učitani artefakti nemaju metrike za poređenje modela. Pokrenite sa --retrain za novu evaluaciju.")
        payload.setdefault("best_model", "Item-kNN")

    # Override hybrid α ako je zadat.
    if args.alpha is not None:
        payload["hybrid_alpha"] = float(args.alpha)
        print(f"[Info] Hybrid α override: {float(args.alpha):.2f}")

    print_model_comparison(payload)

    chosen_model = args.model or str(payload.get("best_model", "Item-kNN"))
    if args.model and args.model != payload.get("best_model"):
        print(f"\n[Info] Override: koristim model '{chosen_model}' umesto best='{payload.get('best_model')}'.")
    else:
        print(f"\n[Info] Aktivni model za preporuke: {chosen_model}")

    if args.demo:
        run_demo(payload, top_n=int(args.topn), model_name=chosen_model)
        return

    if args.user:
        run_single_recommendation(payload, user_id=str(args.user), top_n=int(args.topn), model_name=chosen_model)
        return

    print("\nKomande:")
    print("  - Unesi user_id (reviewerID) za Top-N preporuka.")
    print("  - Prazan user_id ili 'exit' prekida.")

    while True:
        u = input("\nUnesi user_id (ili 'exit'): ").strip()
        if not u or u.lower() in ["exit", "quit", "q"]:
            break
        run_single_recommendation(payload, user_id=str(u), top_n=int(args.topn), model_name=chosen_model)

    print("\nKraj.")


if __name__ == "__main__":
    try:
        main()
    except DATA_ERRORS as e:
        sys.exit(f"{type(e).__name__}: {e}")

