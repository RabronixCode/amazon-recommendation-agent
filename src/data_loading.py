

import sys
import os  # Rad sa fajlovima/folderima.
import math
import time  # log2 za NDCG.
import numpy as np  # Numerika + sampling + SGD.
import pandas as pd  # DataFrame pipeline.

from sklearn.metrics import root_mean_squared_error  # RMSE.
from sklearn.metrics import mean_absolute_error  # MAE.

from config import DATA_ERRORS

import dataprep
import plots
import evaluation
import models


try:  # Sparse podrška (za item-kNN).
    from scipy.sparse import csr_matrix  # CSR matrice.
except ImportError:  # Ako scipy nije instaliran.
    csr_matrix = None  # Obeleži da nema sparse.

pd.set_option("display.max_rows", None)  # Debug prikaz više redova.


def save_filtering_effect_report(df_before: pd.DataFrame, df_after: pd.DataFrame, out_dir: str = "eda_outputs") -> None:  # Snima pre/posle izveštaj.
    os.makedirs(out_dir, exist_ok=True)  # Kreiraj folder.
    table = dataprep.make_filtering_report_table(df_before, df_after)
    table.to_csv(os.path.join(out_dir, "filtering_effect.csv"), index=False)  # Snimi CSV.
    print("\n[Filtering Effect] Pre/posle filtriranja:")  # Naslov.
    print(table)  # Ispis.
    plots.plot_filtering_effects(df_before, df_after, out_dir)
    

from itertools import product

def run_mf_grid_search(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    n_users: int,
    n_items: int,
    out_dir: str,
    factors_list=(16, 32, 64),
    lr_list=(0.01, 0.02),
    reg_list=(0.02, 0.05, 0.1),
    n_epochs_list=(10, 20),
    seeds=(42,),
) -> tuple[dict[str, object], pd.DataFrame]:
    """
    Ispituje više MF konfiguracija i bira najbolju po best_val_rmse.
    Snima CSV: mf_grid_results.csv
    Vraća: (best_mf_model_dict, results_table)
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    best_mf = None
    best_cfg = None
    best_val = float("inf")

    exp_id = 0
    for factors, lr, reg, n_epochs, seed in product(factors_list, lr_list, reg_list, n_epochs_list, seeds):
        exp_id += 1
        print(
            f"\n[MF-GRID] Exp {exp_id} | factors={factors} lr={lr} reg={reg} epochs={n_epochs} seed={seed}"
        )

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
    results.to_csv(os.path.join(out_dir, "mf_grid_results.csv"), index=False)

    print("\n[MF-GRID] Top 5 konfiguracija po best_val_rmse:")
    print(results.head(5))

    print("\n[MF-GRID] BEST config:")
    print(best_cfg)

    return best_mf, results

def build_user_history_ratings(df_train: pd.DataFrame) -> dict[int, dict[int, float]]:  # user->(item->rating).
    hist: dict[int, dict[int, float]] = {}  # init.
    for u, i, r in zip(df_train["user_index"].to_numpy(int), df_train["item_index"].to_numpy(int), df_train["rating"].to_numpy(float)):  # iter.
        if int(u) not in hist:  # ako nema.
            hist[int(u)] = {}  # init dict.
        hist[int(u)][int(i)] = float(r)  # upiši.
    return hist  # vrati.

def user_knn_coverage_stats(
    df: pd.DataFrame,
    user_hist: dict[int, dict[int, float]],
    user_mean_arr: np.ndarray,
    neigh_u_idx: np.ndarray,
    neigh_u_sim: np.ndarray,
) -> dict[str, float]:
    """
    Koliko često user-kNN ima barem jednog suseda koji je ocenio ciljni item.
    (Za diplomski: 'coverage' i razlog fallback-a.)
    """
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
    n_neg: int = 99,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vraća:
      ranks: rang pozitivnog itema u [1..(1+n_neg)]
      ties:  broj tie elemenata (koliko kandidata ima isti skor kao pozitivan)
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
        ties = int(np.sum(scores == pos_score))
        ties_arr[t] = ties

        # tie-aware rang (kao u evaluaciji)
        rnk = evaluation.rank_of_positive_tie_aware(scores, pos_index=0, rng=rng)
        ranks[t] = int(rnk)

    return ranks, ties_arr

def compute_negative_pool_sizes(
    df_pos: pd.DataFrame,
    user_interactions: dict[int, set[int]],
    n_items: int,
) -> np.ndarray:
    """
    Pool size = koliko itema ostaje nakon filtera 'already interacted' (+ izbacivanje pozitivnog).
    U tvom setup-u interacted_set uglavnom već sadrži positive_item, pa je:
      pool = n_items - |interacted_set|
    (ako iz nekog razloga positive nije u set-u, onda još -1)
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
    n_neg: int = 99,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tie-rate: procenat slučajeva gde ties > 1 (pozitivan skor deljen sa još nekim kandidatom).
    Vraća tabelu: model, tie_rate, avg_ties
    """
    rows = []
    for name, score_fn in models_fn:
        ranks, ties = collect_positive_ranks_and_ties(
            df_pos=df_pos,
            user_interactions=user_interactions,
            all_items=all_items,
            score_fn=score_fn,
            n_neg=n_neg,
            seed=seed,
        )
        tie_rate = float(np.mean(ties > 1))
        avg_ties = float(np.mean(ties))
        rows.append({"model": name, "tie_rate": tie_rate, "avg_ties": avg_ties})
    return pd.DataFrame(rows)




def run_one_setting(df_sorted, df_before, out_root, min_u, min_i, max_iters=20, seed_val=42, seed_test=43):
    setting_tag = f"minU{min_u}_minI{min_i}"
    out_dir = os.path.join(out_root, setting_tag)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Filter
    df_cut = dataprep.iterative_filter_min_interactions(
        df_sorted,
        min_user_interactions=min_u,
        min_item_interactions=min_i,
        max_iters=max_iters,
    )
    save_filtering_effect_report(df_before, df_cut, out_dir=out_dir)

    # 2) Indexing
    df_cut = df_cut.copy()
    df_cut["user_index"] = df_cut["user_id"].astype("category").cat.codes.astype(int)
    df_cut["item_index"] = df_cut["item_id"].astype("category").cat.codes.astype(int)
    n_users = int(df_cut["user_index"].nunique())
    n_items = int(df_cut["item_index"].nunique())

    # 3) Split
    df_train, df_val, df_test = dataprep.build_splits_leave_one_out(df_cut)

    assert df_val["user_index"].isin(df_train["user_index"]).all()
    assert df_test["user_index"].isin(df_train["user_index"]).all()

    # 4) Baseline parametri (iz TRAIN)
    mu = float(df_train["rating"].mean())
    user_mean_arr = np.full(n_users, mu, dtype=np.float32)
    item_mean_arr = np.full(n_items, mu, dtype=np.float32)

    um = df_train.groupby("user_index")["rating"].mean()
    im = df_train.groupby("item_index")["rating"].mean()
    user_mean_arr[um.index.to_numpy(int)] = um.to_numpy(np.float32)
    item_mean_arr[im.index.to_numpy(int)] = im.to_numpy(np.float32)

    bu, bi = models.fit_bias_model_by_index(df_train, mu=mu, n_users=n_users, n_items=n_items, reg=10.0, n_iters=20)

    # 5) Rating evaluacija (baselines + bias)
    rating_rows = []
    for name, pred_fn in [
        ("GlobalMean", lambda d: models.pred_global_mean(d, mu)),
        ("UserMean", lambda d: models.pred_user_mean(d, user_mean_arr)),
        ("ItemMean", lambda d: models.pred_item_mean(d, item_mean_arr)),
        ("Bias(μ+bu+bi)", lambda d: models.pred_bias(d, mu, bu, bi)),
    ]:
        yv = df_val["rating"].to_numpy(float)
        yt = df_test["rating"].to_numpy(float)
        pv = pred_fn(df_val)
        pt = pred_fn(df_test)
        mv = evaluation.eval_rating_metrics(yv, pv)
        mt = evaluation.eval_rating_metrics(yt, pt)
        rating_rows.append({"model": name, "val_RMSE": mv["rmse"], "val_MAE": mv["mae"], "test_RMSE": mt["rmse"], "test_MAE": mt["mae"]})

    # 6) MF (isti hiperparametri kao kod tebe)
    mf = models.train_biased_mf_sgd(df_train, df_val, n_users=n_users, n_items=n_items, factors=32, lr=0.02, reg=0.02, n_epochs=20, seed=42)

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

    rating_rows.append({"model": "MF(Biased SGD, k=32)", "val_RMSE": mf_val_rmse, "val_MAE": mf_val_mae, "test_RMSE": mf_test_rmse, "test_MAE": mf_test_mae})
    rating_table = pd.DataFrame(rating_rows)
    rating_table.to_csv(os.path.join(out_dir, "rating_metrics_table.csv"), index=False)

    # 7) Ranking setup (kao kod tebe, samo MF + Bias + baselines)
    all_items = np.arange(n_items, dtype=int)
    user_interactions = {}
    for u, i in zip(df_cut["user_index"].to_numpy(int), df_cut["item_index"].to_numpy(int)):
        user_interactions.setdefault(int(u), set()).add(int(i))

    df_val_pos = df_val[["user_index", "item_index"]].copy()
    df_test_pos = df_test[["user_index", "item_index"]].copy()

    def score_global(u_idx, cand): return np.full(len(cand), mu, dtype=float)
    def score_usermean(u_idx, cand): return np.full(len(cand), float(user_mean_arr[u_idx]), dtype=float)
    def score_itemmean(u_idx, cand): return item_mean_arr[cand].astype(float)
    def score_bias(u_idx, cand): return (mu + bu[u_idx] + bi[cand]).astype(float)
    def score_mf(u_idx, cand): return models.mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], u_idx, cand)

    models_for_ranking = [
        ("GlobalMean", score_global),
        ("UserMean", score_usermean),
        ("ItemMean", score_itemmean),
        ("Bias(μ+bu+bi)", score_bias),
        ("MF(Biased SGD, k=32)", score_mf),
    ]

    k = 10
    n_neg = 99

    ranking_rows = []
    for model_name, score_fn in models_for_ranking:
        val_m = evaluation.evaluate_ranking(df_val_pos, user_interactions, all_items, score_fn, k=k, n_neg=n_neg, seed=seed_val)
        test_m = evaluation.evaluate_ranking(df_test_pos, user_interactions, all_items, score_fn, k=k, n_neg=n_neg, seed=seed_test)
        ranking_rows.append({
            "model": model_name,
            "VAL_NDCG@10": val_m["NDCG@10"],
            "TEST_NDCG@10": test_m["NDCG@10"],
            "VAL_HR@10": val_m["HR@10"],
            "TEST_HR@10": test_m["HR@10"],
            "VAL_MRR@10": val_m["MRR@10"],
            "TEST_MRR@10": test_m["MRR@10"],
        })

    ranking_table = pd.DataFrame(ranking_rows)
    ranking_table.to_csv(os.path.join(out_dir, "ranking_metrics_table_fixed.csv"), index=False)

    # 8) Kratak “summary” red za grid rezultate (uz MF kao glavni)
    mf_row = ranking_table[ranking_table["model"] == "MF(Biased SGD, k=32)"].iloc[0]
    return {
        "min_user": min_u,
        "min_item": min_i,
        "interactions": int(len(df_cut)),
        "users": int(n_users),
        "items": int(n_items),
        "train": int(len(df_train)),
        "val": int(len(df_val)),
        "test": int(len(df_test)),
        "mf_VAL_NDCG@10": float(mf_row["VAL_NDCG@10"]),
        "mf_TEST_NDCG@10": float(mf_row["TEST_NDCG@10"]),
        "mf_val_RMSE": float(rating_table[rating_table["model"] == "MF(Biased SGD, k=32)"]["val_RMSE"].iloc[0]),
        "mf_test_RMSE": float(rating_table[rating_table["model"] == "MF(Biased SGD, k=32)"]["test_RMSE"].iloc[0]),
        "out_dir": out_dir,
    }


# =========================
# 7.X) Hybrid grid search + dedicated EDA plots
# =========================

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
    out_dir: str,
    mf_factors_grid: tuple[int, ...] = (16, 32, 64),
    mf_reg_grid: tuple[float, ...] = (0.02, 0.05),
    knn_topk_grid: tuple[int, ...] = (20, 50, 100),
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    mf_lr: float = 0.02,
    mf_epochs: int = 10,
    sample_users: int = 500,
    n_neg: int = 99,
    k: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Hybrid grid search nad (mf_factors, mf_reg, knn_topk, α).

    Strategija (efikasna):
      1) Treniraj MF jednom po (factors, reg).
      2) Precompute Item-kNN jednom po top_k.
      3) Za svaku (mf_cfg, top_k) keširaj (MF score, kNN score) po useru.
      4) Za svaki α blendiraj keširane skorove i izračunaj VAL HR/NDCG/MRR (jeftino).

    Outputs:
      - hybrid_grid_results.csv  (sve kombinacije, sortirane po VAL_NDCG@k)
      - hybrid_best_config.csv   (najbolja kombinacija)
    Returns: (df_results, best_config_dict)
    """
    if csr_matrix is None:
        print("[Hybrid Grid] WARN: scipy.sparse nije dostupan — preskačem grid (Item-kNN nemoguć).")
        return pd.DataFrame(), {}

    rng = np.random.default_rng(seed)
    n_val_s = min(int(sample_users), len(df_val))
    df_val_s = df_val.sample(n=n_val_s, random_state=seed)

    user_interactions: dict[int, set[int]] = {}
    for u, i in zip(df_cut["user_index"].to_numpy(int), df_cut["item_index"].to_numpy(int)):
        user_interactions.setdefault(int(u), set()).add(int(i))
    all_items = np.arange(n_items, dtype=int)

    cand_cache: list[tuple[int, np.ndarray]] = []
    for _, row in df_val_s.iterrows():
        u = int(row["user_index"])
        pos = int(row["item_index"])
        interacted = user_interactions.get(u, set())
        neg = evaluation.sample_negatives_for_user(all_items, interacted, pos, n_neg=n_neg, rng=rng)
        cand = np.concatenate([[pos], neg]).astype(int)
        cand_cache.append((u, cand))

    print(f"[Hybrid Grid] {len(cand_cache)} (user, kandidati) parova. "
          f"Combos: {len(mf_factors_grid)*len(mf_reg_grid)*len(knn_topk_grid)*len(alpha_grid)} "
          f"(mf×reg×topk×α = {len(mf_factors_grid)}×{len(mf_reg_grid)}×{len(knn_topk_grid)}×{len(alpha_grid)})")

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
            print(f"  Trajanje: {time.time() - t0:.1f}s")

    knn_models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    dev_mat = models.build_sparse_deviation_matrix(
        df_train, n_users=n_users, n_items=n_items,
        user_mean_arr=user_mean_arr.astype(np.float32),
    )
    for tk in knn_topk_grid:
        print(f"[Hybrid Grid] Item-kNN precompute (top_k={tk})...")
        t0 = time.time()
        ni, nsim = models.compute_item_topk_similarities(dev_mat, top_k=int(tk), block_size=256)
        knn_models[int(tk)] = (ni, nsim)
        print(f"  Trajanje: {time.time() - t0:.1f}s")

    rows: list[dict] = []
    for (mf_f, mf_r), mf in mf_models.items():
        for tk, (ni, nsim) in knn_models.items():
            print(f"[Hybrid Grid] Skoring (factors={mf_f}, reg={mf_r}, top_k={tk})...")
            t0 = time.time()
            scored: list[tuple[np.ndarray, np.ndarray]] = []
            for u, cand in cand_cache:
                mf_s = models.mf_score_candidates(
                    mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], int(u), cand,
                )
                knn_s = np.zeros(len(cand), dtype=float)
                for t, it in enumerate(cand):
                    fb = float(np.clip(mu + bu[int(u)] + bi[int(it)], 1.0, 5.0))
                    knn_s[t] = models.predict_item_knn_one(
                        int(u), int(it), user_hist, user_mean_arr, ni, nsim, fb,
                    )
                scored.append((np.asarray(mf_s, dtype=float), knn_s))
            print(f"  Skoring trajanje: {time.time() - t0:.1f}s")

            for a in alpha_grid:
                tie_rng = np.random.default_rng(seed + 1000)
                hrs: list[float] = []
                ndcgs: list[float] = []
                mrrs: list[float] = []
                for mf_s, knn_s in scored:
                    blend = float(a) * mf_s + (1.0 - float(a)) * knn_s
                    pos_score = float(blend[0])
                    higher = int(np.sum(blend > pos_score))
                    ties = int(np.sum(blend == pos_score))
                    tie_off = int(tie_rng.integers(low=1, high=ties + 1))
                    rnk = higher + tie_off
                    hrs.append(1.0 if rnk <= k else 0.0)
                    ndcgs.append(0.0 if rnk > k else 1.0 / math.log2(rnk + 1))
                    mrrs.append(0.0 if rnk > k else 1.0 / float(rnk))
                rows.append({
                    "mf_factors": int(mf_f),
                    "mf_reg": float(mf_r),
                    "knn_topk": int(tk),
                    "alpha": float(a),
                    f"VAL_HR@{k}": float(np.mean(hrs)),
                    f"VAL_NDCG@{k}": float(np.mean(ndcgs)),
                    f"VAL_MRR@{k}": float(np.mean(mrrs)),
                })

    df_grid = pd.DataFrame(rows).sort_values(f"VAL_NDCG@{k}", ascending=False).reset_index(drop=True)
    df_grid.to_csv(os.path.join(out_dir, "hybrid_grid_results.csv"), index=False)
    print(f"[Saved] {os.path.join(out_dir, 'hybrid_grid_results.csv')}")

    best = {kk: (int(v) if isinstance(v, (np.integer, int)) and kk in {"mf_factors", "knn_topk"} else float(v))
            for kk, v in df_grid.iloc[0].to_dict().items()}
    pd.DataFrame([best]).to_csv(os.path.join(out_dir, "hybrid_best_config.csv"), index=False)
    print(f"[Saved] {os.path.join(out_dir, 'hybrid_best_config.csv')}")
    print(f"[Hybrid Grid] Najbolja konfiguracija: {best}")

    return df_grid, best




# =========================
# 8) Glavni pipeline: data + baselines + MF + item-kNN + ranking tabela
# =========================

def prepare_and_run_all(
    df_raw: pd.DataFrame,
    out_dir: str = "eda_outputs",
    run_hybrid_grid: bool = True,
) -> None:  # End-to-end.
    os.makedirs(out_dir, exist_ok=True)  # Folder.

    column_mapping = {  # Mapiranje Amazon kolona.
        "reviewerID": "user_id",  # user.
        "asin": "item_id",  # item.
        "overall": "rating",  # rating.
        "unixReviewTime": "timestamp",  # timestamp.
    }

    missing = [c for c in column_mapping if c not in df_raw.columns]  # Provera kolona.
    if missing:  # Ako fali.
        raise ValueError(f"Nedostaju kolone: {missing}. Dostupne: {list(df_raw.columns)}")  # Stop.

    df = df_raw[list(column_mapping.keys())].copy()  # Selektuj potrebne kolone.
    df.rename(columns=column_mapping, inplace=True)  # Rename u standard.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  # UNIX->datetime.
    df["rating"] = df["rating"].astype(float)  # float rating.

    df.dropna(subset=["user_id", "item_id", "rating", "timestamp"], inplace=True)  # Ukloni null.
    df.drop_duplicates(inplace=True)

    df_sorted = df.sort_values("timestamp", ascending=True)  # Sort po vremenu.
    df_before = df_sorted.copy()  # Snapshot pre filtriranja.

    rating_counts = df["rating"].value_counts().sort_index()
    plots.plot_rating_distribution(rating_counts, out_dir)    


    user_activity = df.groupby("user_id").size().to_numpy()
    item_activity = df.groupby("item_id").size().to_numpy()

    def ecdf(x: np.ndarray):
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        x_sorted = np.sort(x)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    ux, uy = ecdf(user_activity)
    ix, iy = ecdf(item_activity)

    plots.plot_cdf(ux, uy, ix, iy, out_dir)
    
    
    ts = df_sorted.set_index("timestamp")
    timeline = ts["rating"].resample("MS").size()  # MS = month start
    plots.plot_interaction_timeline(timeline, out_dir)

    df_cut = dataprep.iterative_filter_min_interactions(df_sorted, min_user_interactions=4, min_item_interactions=2, max_iters=20)  # Stabilno filtriranje.
    save_filtering_effect_report(df_before, df_cut, out_dir=out_dir)  # EDA pre/posle.

    df_cut = df_cut.copy()  # Copy.
    df_cut["user_index"] = df_cut["user_id"].astype("category").cat.codes.astype(int)  # user_id->index.
    df_cut["item_index"] = df_cut["item_id"].astype("category").cat.codes.astype(int)  # item_id->index.

    n_users = int(df_cut["user_index"].nunique())  # #users.
    n_items = int(df_cut["item_index"].nunique())  # #items.

    df_train, df_val, df_test = dataprep.build_splits_leave_one_out(df_cut)  # LOO split.

    plots.plot_train_rating_histogram(df_train, out_dir=out_dir, bins=10)

    plots.plot_train_interaction_distributions(
        df_train,
        out_dir=out_dir,
        min_user_interactions=4,
        min_item_interactions=2,
        bins=60,
        log_y=True,
    )

    print(f"\n[Data] interactions={len(df_cut)} users={n_users} items={n_items}")  # Pregled.
    print(f"[Data] train/val/test={len(df_train)}/{len(df_val)}/{len(df_test)}")  # Pregled.

    mu = float(df_train["rating"].mean())  # Global mean iz train-a.
    print(mu)  # Ispis global mean.

    user_mean_arr = np.full(n_users, mu, dtype=np.float32)  # Default user mean = mu.
    item_mean_arr = np.full(n_items, mu, dtype=np.float32)  # Default item mean = mu.

    um = df_train.groupby("user_index")["rating"].mean()  # user mean.
    im = df_train.groupby("item_index")["rating"].mean()  # item mean.

    user_mean_arr[um.index.to_numpy(int)] = um.to_numpy(np.float32)  # fill user mean.
    item_mean_arr[im.index.to_numpy(int)] = im.to_numpy(np.float32)  # fill item mean.

    bu, bi = models.fit_bias_model_by_index(df_train, mu=mu, n_users=n_users, n_items=n_items, reg=10.0, n_iters=20)  # Bias model.

    # Rating evaluacija (RMSE+MAE) za baseline/bias (val+test)
    rating_rows = []  # Lista redova.
    for name, pred_fn in [  # Modeli.
        ("GlobalMean", lambda d: models.pred_global_mean(d, mu)),  # Global.
        ("UserMean", lambda d: models.pred_user_mean(d, user_mean_arr)),  # User mean.
        ("ItemMean", lambda d: models.pred_item_mean(d, item_mean_arr)),  # Item mean.
        ("Bias(μ+bu+bi)", lambda d: models.pred_bias(d, mu, bu, bi)),  # Bias.
    ]:
        yv = df_val["rating"].to_numpy(float)  # true val.
        yt = df_test["rating"].to_numpy(float)  # true test.
        pv = pred_fn(df_val)  # pred val.
        pt = pred_fn(df_test)  # pred test.
        mv = evaluation.eval_rating_metrics(yv, pv)  # metrics val.
        mt = evaluation.eval_rating_metrics(yt, pt)  # metrics test.
        rating_rows.append({"model": name, "val_RMSE": mv["rmse"], "val_MAE": mv["mae"], "test_RMSE": mt["rmse"], "test_MAE": mt["mae"]})  # append.

    mf = models.train_biased_mf_sgd(df_train, df_val, n_users=n_users, n_items=n_items, factors=32, lr=0.02, reg=0.02, n_epochs=20, seed=42)

    u_val = df_val["user_index"].to_numpy(int)  # u val.
    i_val = df_val["item_index"].to_numpy(int)  # i val.
    r_val = df_val["rating"].to_numpy(float)  # r val.
    p_val = np.clip(mf["mu"] + mf["bu"][u_val] + mf["bi"][i_val] + np.sum(mf["P"][u_val] * mf["Q"][i_val], axis=1), 1.0, 5.0)  # pred val.
    mf_val_rmse = float(root_mean_squared_error(r_val, p_val))  # val rmse.
    mf_val_mae = float(mean_absolute_error(r_val, p_val))  # val mae.

    u_test = df_test["user_index"].to_numpy(int)  # u test.
    i_test = df_test["item_index"].to_numpy(int)  # i test.
    r_test = df_test["rating"].to_numpy(float)  # r test.
    p_test = np.clip(mf["mu"] + mf["bu"][u_test] + mf["bi"][i_test] + np.sum(mf["P"][u_test] * mf["Q"][i_test], axis=1), 1.0, 5.0)  # pred test.
    mf_test_rmse = float(root_mean_squared_error(r_test, p_test))  # test rmse.
    mf_test_mae = float(mean_absolute_error(r_test, p_test))  # test mae.

    rating_rows.append({"model": "MF(Biased SGD, k=32)", "val_RMSE": mf_val_rmse, "val_MAE": mf_val_mae, "test_RMSE": mf_test_rmse, "test_MAE": mf_test_mae})  # dodaj MF.

    rating_table = pd.DataFrame(rating_rows)  # tabela rating metrika.
    print("\n[Rating Prediction] RMSE + MAE (VAL/TEST):")  # naslov.
    print(rating_table)  # ispis.
    rating_table.to_csv(os.path.join(out_dir, "rating_metrics_table.csv"), index=False)  # snimi.

    plots.plot_baseline_comparison_bars(
        rating_table=rating_table,
        out_dir=out_dir,
        sort_by="val_RMSE",
    )

    # Priprema interakcija po useru za negative sampling (train+val+test)
    all_items = np.arange(n_items, dtype=int)  # svi item-i.
    user_interactions: dict[int, set[int]] = {}  # user->set items.
    for u, i in zip(df_cut["user_index"].to_numpy(int), df_cut["item_index"].to_numpy(int)):  # iter.
        user_interactions.setdefault(int(u), set()).add(int(i))  # upiši.

    df_val_pos = df_val[["user_index", "item_index"]].copy()  # val positives.
    df_test_pos = df_test[["user_index", "item_index"]].copy()  # test positives.

    user_interactions_train: dict[int, set[int]] = {}
    for u, i in zip(df_train["user_index"].to_numpy(int), df_train["item_index"].to_numpy(int)):
        user_interactions_train.setdefault(int(u), set()).add(int(i))
    hist_u = pd.Series({u: len(items) for u, items in user_interactions_train.items()}, dtype=int)  # istorija po useru.

    # Statistike
    hist_stats = {
        "min": int(hist_u.min()),
        "median": float(hist_u.median()),
        "mean": float(hist_u.mean()),
        "p95": float(np.percentile(hist_u.to_numpy(), 95)),
        "max": int(hist_u.max()),
    }

    print("\n[EDA] |history(u)| statistike (TRAIN):")
    print(
        f"min={hist_stats['min']} | median={hist_stats['median']:.2f} | "
        f"mean={hist_stats['mean']:.2f} | p95={hist_stats['p95']:.2f} | max={hist_stats['max']}"
    )

    plots.plot_histogram_user_activity(hist_u, out_dir)

    plots.plot_histogram_user_logx(hist_u, out_dir)

    # Score funkcije (za ranking)
    def score_global(u_idx: int, cand: np.ndarray) -> np.ndarray:  # Global mean scoring.
        return np.full(len(cand), mu, dtype=float)  # svi isti.

    def score_usermean(u_idx: int, cand: np.ndarray) -> np.ndarray:  # User mean scoring.
        return np.full(len(cand), float(user_mean_arr[u_idx]), dtype=float)  # svi isti po useru.

    def score_itemmean(u_idx: int, cand: np.ndarray) -> np.ndarray:  # Item mean scoring.
        return item_mean_arr[cand].astype(float)  # po itemu.

    def score_bias(u_idx: int, cand: np.ndarray) -> np.ndarray:  # Bias scoring.
        return (mu + bu[u_idx] + bi[cand]).astype(float)  # μ+bu+bi.

    def score_mf(u_idx: int, cand: np.ndarray) -> np.ndarray:  # MF scoring.
        return models.mf_score_candidates(mf["mu"], mf["bu"], mf["bi"], mf["P"], mf["Q"], u_idx, cand)  # MF.
    print("TU SAM")
    
    TOPK_ITEMKNN = 50  # metodologija: topK=50

    item_knn_ready = False
    itemknn_meta = {
        "topK": TOPK_ITEMKNN,
        "scipy_available": (csr_matrix is not None),
        "precompute_sec": None,
        "avg_valid_neighbors": None,
    }

    if csr_matrix is not None:  # ako imamo scipy.
        t0 = time.perf_counter()

        dev_mat = models.build_sparse_deviation_matrix(
            df_train,
            n_users=n_users,
            n_items=n_items,
            user_mean_arr=user_mean_arr.astype(np.float32),
        )

        neigh_idx, neigh_sim = models.compute_item_topk_similarities(dev_mat, top_k=TOPK_ITEMKNN, block_size=256)
        user_hist = build_user_history_ratings(df_train)

        t1 = time.perf_counter()
        itemknn_meta["precompute_sec"] = float(t1 - t0)

        item_knn_ready = True
        print(f"[CF] Item-kNN precompute: {itemknn_meta['precompute_sec']:.3f}s (topK={TOPK_ITEMKNN})")

        # Opciono: distribucija item-item sličnosti (topK ivice)
        sims = neigh_sim.reshape(-1).astype(float)
        sims = sims[np.isfinite(sims)]
        sims = sims[sims != 0.0]

        plots.plot_itemknn_similarity_hist(sims, out_dir, TOPK_ITEMKNN)

        print(f"[Saved] {os.path.join(out_dir, 'item_knn_similarity_hist.png')}")

    else:
        print("\n[CF] Item-kNN preskočen: scipy.sparse nije dostupan.")

    def score_itemknn(u_idx: int, cand: np.ndarray) -> np.ndarray:  # Item-kNN scoring.
        out = np.zeros(len(cand), dtype=float)  # skorovi.
        for t, it in enumerate(cand):  # svaki kandidat.
            fb = float(mu + bu[u_idx] + bi[int(it)])  # fallback bias.
            out[t] = models.predict_item_knn_one(u_idx, int(it), user_hist, user_mean_arr, neigh_idx, neigh_sim, fb)  # pred.
        return out  # vrati.

    # =========================
    # 8.1) DODATO: User-based kNN CF (user-user sim + predikcija + EDA grafike)
    # =========================
    user_knn_ready = False  # flag.
    if csr_matrix is not None:  # ako imamo scipy.
        # Reuse user_hist ako već postoji (item-kNN ga pravi), inače napravi.
        if "user_hist" not in locals():
            user_hist = build_user_history_ratings(df_train)  # istorija.

        # Reuse dev_mat ako već postoji (item-kNN ga pravi), inače izgradi.
        if "dev_mat" in locals():
            dev_mat_user = dev_mat
        else:
            dev_mat_user = models.build_sparse_deviation_matrix(df_train, n_users=n_users, n_items=n_items, user_mean_arr=user_mean_arr.astype(np.float32))

        # Precompute user-user topK sličnosti
        t0 = time.time()
        neigh_u_idx, neigh_u_sim = models.compute_user_topk_similarities(dev_mat_user, top_k=50, block_size=256)
        t1 = time.time()
        print(f"\n[CF] User-kNN precompute finished: topK=50  time={t1 - t0:.2f}s")

        # EDA: sličnosti (hist, mean/max po useru)
        plots.plot_user_knn_similarity_diagnostics(neigh_u_sim, out_dir=out_dir, top_k=50)

        # Coverage statistike (koliko puta user-kNN ima bar jednog suseda koji je ocenio ciljni item)
        cov_val = user_knn_coverage_stats(df_val, user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim)
        cov_test = user_knn_coverage_stats(df_test, user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim)
        cov_tbl = pd.DataFrame([
            {"split": "val", **cov_val},
            {"split": "test", **cov_test},
        ])
        cov_tbl.to_csv(os.path.join(out_dir, "user_knn_coverage.csv"), index=False)
        print("\n[CF] User-kNN coverage (VAL/TEST):")
        print(cov_tbl)
        print(f"[Saved] {os.path.join(out_dir, 'user_knn_coverage.csv')}")

        user_knn_ready = True
    else:
        print("\n[CF] User-kNN preskočen: scipy.sparse nije dostupan.")  # info.

    def score_userknn(u_idx: int, cand: np.ndarray) -> np.ndarray:  # User-kNN scoring.
        out = np.zeros(len(cand), dtype=float)
        for t, it in enumerate(cand):
            fb = float(mu + bu[u_idx] + bi[int(it)])  # fallback bias.
            out[t] = models.predict_user_knn_one(int(u_idx), int(it), user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim, fb)
        return out

    # DODATO: Rating evaluacija za User-kNN (RMSE+MAE) — tek kada su pripreme gotove
    if user_knn_ready:
        yv = df_val["rating"].to_numpy(float)
        yt = df_test["rating"].to_numpy(float)

        fallback_fn = lambda u, i: (mu + bu[int(u)] + bi[int(i)])  # fallback bias (konzistentno sa item-kNN)
        pv = models.user_knn_predict_df(df_val, user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim, fallback_fn=fallback_fn)
        pt = models.user_knn_predict_df(df_test, user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim, fallback_fn=fallback_fn)

        mv = evaluation.eval_rating_metrics(yv, pv)
        mt = evaluation.eval_rating_metrics(yt, pt)

        rating_rows.append({
            "model": "User-kNN(Cosine, topK=50)",
            "val_RMSE": mv["rmse"],
            "val_MAE": mv["mae"],
            "test_RMSE": mt["rmse"],
            "test_MAE": mt["mae"],
        })

        # Refresh tabela + snimanje (da User-kNN uđe u poređenja i bar chart)
        rating_table = pd.DataFrame(rating_rows)
        print("\n[Rating Prediction] RMSE + MAE (VAL/TEST) + User-kNN dodat:")
        print(rating_table)
        rating_table.to_csv(os.path.join(out_dir, "rating_metrics_table.csv"), index=False)

        plots.plot_baseline_comparison_bars(
            rating_table=rating_table,
            out_dir=out_dir,
            sort_by="val_RMSE",
        )

    # =========================
    # 8.3) DODATO: Hybrid model (α·MF + (1−α)·Item-kNN) sa alpha sweep-om
    # =========================
    hybrid_ready = False
    HYBRID_ALPHA_GRID = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
    hybrid_alpha = 0.5  # default; biće prepisana ako item-kNN postoji.

    if item_knn_ready:
        # Brzo tuniranje α na validaciji koristeći zajedničke negative i kešovane skorove.
        # (Item-kNN je ovde najsporiji deo — keširamo ga jednom po useru i jeftino blendujemo.)
        rng_alpha = np.random.default_rng(42)
        df_val_alpha = df_val.sample(n=min(500, len(df_val)), random_state=42)
        cached_scores: list[tuple[np.ndarray, np.ndarray]] = []
        for _, row in df_val_alpha.iterrows():
            u_a = int(row["user_index"])
            pos_a = int(row["item_index"])
            interacted_a = user_interactions.get(u_a, set())
            neg_a = models.sample_negatives_for_user(all_items, interacted_a, pos_a, n_neg=99, rng=rng_alpha)
            cand_a = np.concatenate([[pos_a], neg_a]).astype(int)
            mf_s = np.asarray(score_mf(u_a, cand_a), dtype=float)
            knn_s = np.asarray(score_itemknn(u_a, cand_a), dtype=float)
            cached_scores.append((mf_s, knn_s))

        sweep_rows = []
        rank_rng = np.random.default_rng(43)
        k_alpha = 10
        for a in HYBRID_ALPHA_GRID:
            ndcgs_a, hrs_a, mrrs_a = [], [], []
            for mf_s, knn_s in cached_scores:
                blended = float(a) * mf_s + (1.0 - float(a)) * knn_s
                pos_score = float(blended[0])
                higher = int(np.sum(blended > pos_score))
                ties = int(np.sum(blended == pos_score))
                tie_offset = int(rank_rng.integers(low=1, high=ties + 1))
                rnk = higher + tie_offset
                hrs_a.append(1.0 if rnk <= k_alpha else 0.0)
                ndcgs_a.append(0.0 if rnk > k_alpha else 1.0 / math.log2(rnk + 1))
                mrrs_a.append(0.0 if rnk > k_alpha else 1.0 / float(rnk))
            sweep_rows.append({
                "alpha": float(a),
                f"VAL_HR@{k_alpha}": float(np.mean(hrs_a)),
                f"VAL_NDCG@{k_alpha}": float(np.mean(ndcgs_a)),
                f"VAL_MRR@{k_alpha}": float(np.mean(mrrs_a)),
            })

        sweep_df = pd.DataFrame(sweep_rows).sort_values(f"VAL_NDCG@{k_alpha}", ascending=False).reset_index(drop=True)
        sweep_df.to_csv(os.path.join(out_dir, "hybrid_alpha_sweep.csv"), index=False)
        print("\n[Hybrid] α tuning na validaciji (α=1.0 → čist MF, α=0.0 → čist Item-kNN):")
        print(sweep_df.to_string(index=False))
        hybrid_alpha = float(sweep_df.iloc[0]["alpha"])
        print(f"[Saved] {os.path.join(out_dir, 'hybrid_alpha_sweep.csv')}")
        print(f"[Hybrid] Izabrana α = {hybrid_alpha:.2f} (max VAL_NDCG@{k_alpha}).")

        plots.plot_hybrid_alpha_curve(
            sweep_df, out_dir=out_dir, k=k_alpha,
            filename="hybrid_alpha_sweep.png",
            title="Hybrid α sweep — VAL metrics @10 (default MF & top_k)",
        )

        def score_hybrid(u_idx: int, cand: np.ndarray) -> np.ndarray:
            """Hybrid: α·MF + (1−α)·Item-kNN scoring."""
            mf_s = np.asarray(score_mf(int(u_idx), cand), dtype=float)
            knn_s = np.asarray(score_itemknn(int(u_idx), cand), dtype=float)
            return (float(hybrid_alpha) * mf_s + (1.0 - float(hybrid_alpha)) * knn_s).astype(float)

        hybrid_ready = True

    # Hybrid rating evaluacija (RMSE+MAE) — koristi α blend predikcija na df_val/df_test
    if hybrid_ready:
        u_v = df_val["user_index"].to_numpy(int)
        i_v = df_val["item_index"].to_numpy(int)
        u_t = df_test["user_index"].to_numpy(int)
        i_t = df_test["item_index"].to_numpy(int)

        # MF predikcije (vektorizovane)
        mf_val_pred = np.clip(mf["mu"] + mf["bu"][u_v] + mf["bi"][i_v] + np.sum(mf["P"][u_v] * mf["Q"][i_v], axis=1), 1.0, 5.0)
        mf_test_pred = np.clip(mf["mu"] + mf["bu"][u_t] + mf["bi"][i_t] + np.sum(mf["P"][u_t] * mf["Q"][i_t], axis=1), 1.0, 5.0)

        # Item-kNN predikcije (Python petlja, ali samo na val/test redovima — ne na svim itemima)
        knn_val_pred = np.zeros(len(df_val), dtype=float)
        knn_test_pred = np.zeros(len(df_test), dtype=float)
        for t in range(len(df_val)):
            uu = int(u_v[t]); ii = int(i_v[t])
            fb = float(np.clip(mu + bu[uu] + bi[ii], 1.0, 5.0))
            knn_val_pred[t] = models.predict_item_knn_one(uu, ii, user_hist, user_mean_arr, neigh_idx, neigh_sim, fb)
        for t in range(len(df_test)):
            uu = int(u_t[t]); ii = int(i_t[t])
            fb = float(np.clip(mu + bu[uu] + bi[ii], 1.0, 5.0))
            knn_test_pred[t] = models.predict_item_knn_one(uu, ii, user_hist, user_mean_arr, neigh_idx, neigh_sim, fb)

        hybrid_val_pred = float(hybrid_alpha) * mf_val_pred + (1.0 - float(hybrid_alpha)) * knn_val_pred
        hybrid_test_pred = float(hybrid_alpha) * mf_test_pred + (1.0 - float(hybrid_alpha)) * knn_test_pred

        plots.plot_hybrid_pred_distribution(
            mf_pred=mf_test_pred, knn_pred=knn_test_pred, hybrid_pred=hybrid_test_pred,
            out_dir=out_dir,
            title=f"Distribucija predikcija po modelu (TEST, α={hybrid_alpha:.2f})",
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
        print("\n[Rating Prediction] RMSE + MAE (VAL/TEST) + Hybrid dodat:")
        print(rating_table)
        rating_table.to_csv(os.path.join(out_dir, "rating_metrics_table.csv"), index=False)

    # Ranking evaluacija (FIX)
    k = 10  # Top-10.
    n_neg = 99  # 99 negativa.

    ranking_rows = []  # redovi.
    models_for_ranking = [  # lista modela.
        ("GlobalMean", score_global),
        ("UserMean", score_usermean),
        ("ItemMean", score_itemmean),
        ("Bias(μ+bu+bi)", score_bias),
        ("MF(Biased SGD, k=32)", score_mf),
    ]  # kraj.

    if item_knn_ready:  # ako item-kNN spreman.
        models_for_ranking.append(("Item-kNN(AdjCos, topK=50)", score_itemknn))  # dodaj.

    if user_knn_ready:
        models_for_ranking.append(("User-kNN(Cosine, topK=50)", score_userknn))

    if hybrid_ready:
        models_for_ranking.append((f"Hybrid(α={hybrid_alpha:.2f}: MF + Item-kNN)", score_hybrid))

    for model_name, score_fn in models_for_ranking:  # svaki model.
        if model_name.startswith("Item-kNN("):
            models.itemknn_reset_neighbor_stats()

        val_m = evaluation.evaluate_ranking(df_val_pos, user_interactions, all_items, score_fn, k=k, n_neg=n_neg, seed=42)  # val.
        test_m = evaluation.evaluate_ranking(df_test_pos, user_interactions, all_items, score_fn, k=k, n_neg=n_neg, seed=43)  # test.

        if model_name.startswith("Item-kNN("):
            itemknn_meta["avg_valid_neighbors"] = models.itemknn_get_avg_valid_neighbors()
            print(f"[CF] Item-kNN avg valid neighbors per prediction: {itemknn_meta['avg_valid_neighbors']:.2f}")

            pd.DataFrame([{
                "topK": int(itemknn_meta["topK"]),
                "scipy_available": bool(itemknn_meta["scipy_available"]),
                "precompute_sec": float(itemknn_meta["precompute_sec"]) if itemknn_meta["precompute_sec"] is not None else None,
                "avg_valid_neighbors": float(itemknn_meta["avg_valid_neighbors"]) if itemknn_meta["avg_valid_neighbors"] is not None else None,
            }]).to_csv(os.path.join(out_dir, "item_knn_meta.csv"), index=False)
            print(f"[Saved] {os.path.join(out_dir, 'item_knn_meta.csv')}")

        ranking_rows.append({  # upiši.
            "model": model_name,
            "VAL_HR@10": val_m["HR@10"],
            "VAL_NDCG@10": val_m["NDCG@10"],
            "VAL_MRR@10": val_m["MRR@10"],
            "TEST_HR@10": test_m["HR@10"],
            "TEST_NDCG@10": test_m["NDCG@10"],
            "TEST_MRR@10": test_m["MRR@10"],
        })

    ranking_table = pd.DataFrame(ranking_rows)  # tabela.
    print("\n[Top-K Ranking FIXED] HR@10 / NDCG@10 / MRR@10 (VAL/TEST) uz 99 negativa:")  # naslov.
    print(ranking_table)  # ispis.
    ranking_table.to_csv(os.path.join(out_dir, "ranking_metrics_table_fixed.csv"), index=False)  # snimi.
    
    # Top 3 modeli po TEST_NDCG@10
    top3_ndcg = (
        ranking_table[["model", "TEST_NDCG@10"]]
        .dropna()
        .sort_values("TEST_NDCG@10", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )

    # Razlika MF vs Bias (MF - Bias)
    mf_name = "MF(Biased SGD, k=32)"
    bias_name = "Bias(μ+bu+bi)"

    mf_row = ranking_table[ranking_table["model"] == mf_name]
    bias_row = ranking_table[ranking_table["model"] == bias_name]

    mf_ndcg = float(mf_row["TEST_NDCG@10"].iloc[0]) if len(mf_row) else float("nan")
    bias_ndcg = float(bias_row["TEST_NDCG@10"].iloc[0]) if len(bias_row) else float("nan")
    mf_minus_bias = float(mf_ndcg - bias_ndcg) if np.isfinite(mf_ndcg) and np.isfinite(bias_ndcg) else float("nan")

    # Snimi brojke da ih direktno koristiš u tekstu
    ndcg_summary = pd.DataFrame({
        "metric": ["top1_model", "top1_TEST_NDCG@10",
                   "top2_model", "top2_TEST_NDCG@10",
                   "top3_model", "top3_TEST_NDCG@10",
                   "MF_TEST_NDCG@10", "Bias_TEST_NDCG@10", "MF_minus_Bias"],
        "value": [
            str(top3_ndcg.loc[0, "model"]) if len(top3_ndcg) > 0 else "",
            float(top3_ndcg.loc[0, "TEST_NDCG@10"]) if len(top3_ndcg) > 0 else float("nan"),
            str(top3_ndcg.loc[1, "model"]) if len(top3_ndcg) > 1 else "",
            float(top3_ndcg.loc[1, "TEST_NDCG@10"]) if len(top3_ndcg) > 1 else float("nan"),
            str(top3_ndcg.loc[2, "model"]) if len(top3_ndcg) > 2 else "",
            float(top3_ndcg.loc[2, "TEST_NDCG@10"]) if len(top3_ndcg) > 2 else float("nan"),
            mf_ndcg,
            bias_ndcg,
            mf_minus_bias,
        ]
    })

    ndcg_summary_path = os.path.join(out_dir, "topn_test_ndcg10_summary.csv")
    ndcg_summary.to_csv(ndcg_summary_path, index=False)

    print("\n[Top-N] TEST_NDCG@10 TOP-3:")
    print(top3_ndcg.to_string(index=False))
    print(f"[Top-N] MF vs Bias: {mf_ndcg:.6f} - {bias_ndcg:.6f} = {mf_minus_bias:.6f}")
    print(f"[Saved] {ndcg_summary_path}")

    # Top 3 modeli po TEST_HR@10
    top3_hr = (
        ranking_table[["model", "TEST_HR@10"]]
        .dropna()
        .sort_values("TEST_HR@10", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )

    mf_name = "MF(Biased SGD, k=32)"
    bias_name = "Bias(μ+bu+bi)"

    mf_row = ranking_table[ranking_table["model"] == mf_name]
    bias_row = ranking_table[ranking_table["model"] == bias_name]

    mf_hr = float(mf_row["TEST_HR@10"].iloc[0]) if len(mf_row) else float("nan")
    bias_hr = float(bias_row["TEST_HR@10"].iloc[0]) if len(bias_row) else float("nan")
    mf_minus_bias_hr = float(mf_hr - bias_hr) if np.isfinite(mf_hr) and np.isfinite(bias_hr) else float("nan")

    hr_summary = pd.DataFrame({
        "metric": ["top1_model", "top1_TEST_HR@10",
                   "top2_model", "top2_TEST_HR@10",
                   "top3_model", "top3_TEST_HR@10",
                   "MF_TEST_HR@10", "Bias_TEST_HR@10", "MF_minus_Bias"],
        "value": [
            str(top3_hr.loc[0, "model"]) if len(top3_hr) > 0 else "",
            float(top3_hr.loc[0, "TEST_HR@10"]) if len(top3_hr) > 0 else float("nan"),
            str(top3_hr.loc[1, "model"]) if len(top3_hr) > 1 else "",
            float(top3_hr.loc[1, "TEST_HR@10"]) if len(top3_hr) > 1 else float("nan"),
            str(top3_hr.loc[2, "model"]) if len(top3_hr) > 2 else "",
            float(top3_hr.loc[2, "TEST_HR@10"]) if len(top3_hr) > 2 else float("nan"),
            mf_hr,
            bias_hr,
            mf_minus_bias_hr,
        ]
    })

    hr_summary_path = os.path.join(out_dir, "topn_test_hr10_summary.csv")
    hr_summary.to_csv(hr_summary_path, index=False)

    print("\n[Top-N] TEST_HR@10 TOP-3:")
    print(top3_hr.to_string(index=False))
    print(f"[Top-N] MF vs Bias (HR@10): {mf_hr:.6f} - {bias_hr:.6f} = {mf_minus_bias_hr:.6f}")
    print(f"[Saved] {hr_summary_path}")

    print("DDD")
    



    # =========================
    # 8.2) DODATO: Top-N grafici + rank hist + neg pool + tie-rate (TEST)
    # =========================

    # Graf 1 (obavezan): TEST_NDCG@10 po modelu (centralni dokaz za 6.3)
    plots.plot_topn_bar_test_metric(
        ranking_table=ranking_table,
        out_dir=out_dir,
        metric_col="TEST_NDCG@10",
        title="Top-N evaluacija: TEST_NDCG@10 po modelu",
        filename="topn_test_ndcg10_bar.png",
    )

    # Opciono: TEST_HR@10 bar chart
    plots.plot_topn_bar_test_metric(
        ranking_table=ranking_table,
        out_dir=out_dir,
        metric_col="TEST_HR@10",
        title="Top-N evaluacija: TEST_HR@10 po modelu",
        filename="topn_test_hr10_bar.png",
    )

    # Graf 2 (preporučen): distribucija ranga pozitivnog (1..100) za Bias vs MF (TEST)
    try:
        ranks_bias, ties_bias = collect_positive_ranks_and_ties(
            df_pos=df_test_pos,
            user_interactions=user_interactions,
            all_items=all_items,
            score_fn=score_bias,
            n_neg=99,
            seed=43,
        )
        ranks_mf, ties_mf = collect_positive_ranks_and_ties(
            df_pos=df_test_pos,
            user_interactions=user_interactions,
            all_items=all_items,
            score_fn=score_mf,
            n_neg=99,
            seed=43,
        )

        plots.plot_positive_rank_histogram(
            ranks_a=ranks_bias,
            label_a="Bias(μ+bu+bi)",
            ranks_b=ranks_mf,
            label_b="MF(Biased SGD, k=32)",
            out_dir=out_dir,
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

        stats_bias = rank_stats(ranks_bias, k_hit=10)
        stats_mf = rank_stats(ranks_mf, k_hit=10)

        rank_summary = pd.DataFrame([
            {
                "model": "Bias(μ+bu+bi)",
                "median_rank": stats_bias["median_rank"],
                "p90_rank": stats_bias["p90_rank"],
                "pct_rank_le_10": stats_bias["pct_rank_le_k"],
            },
            {
                "model": "MF(Biased SGD, k=32)",
                "median_rank": stats_mf["median_rank"],
                "p90_rank": stats_mf["p90_rank"],
                "pct_rank_le_10": stats_mf["pct_rank_le_k"],
            },
        ])

        rank_summary_path = os.path.join(out_dir, "positive_rank_hist_test_bias_vs_mf_summary.csv")
        rank_summary.to_csv(rank_summary_path, index=False)

        print("\n[RankHist] Summary (TEST):")
        print(rank_summary.to_string(index=False))
        print(f"[Saved] {rank_summary_path}")
    except Exception as e:
        print(f"[WARN] Rank histogram (Bias vs MF) nije snimljen zbog greške: {e}")

    # Graf 3 (preporučen): negativni pool size (TEST)
    pool_sizes_test = compute_negative_pool_sizes(
        df_pos=df_test_pos,
        user_interactions=user_interactions,
        n_items=n_items,
    )
    plots.plot_negative_pool_histogram(
        pool_sizes=pool_sizes_test,
        out_dir=out_dir,
        filename="neg_pool_size_hist_test.png",
    )

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
    pool_stats.to_csv(os.path.join(out_dir, "neg_pool_size_stats_test.csv"), index=False)
    print(f"[Saved] {os.path.join(out_dir, 'neg_pool_size_stats_test.csv')}")

    # Graf 4 (optional, ali jak): tie-rate po modelu (TEST)
    tie_table_test = compute_tie_rate_table(
        df_pos=df_test_pos,
        user_interactions=user_interactions,
        all_items=all_items,
        models=models_for_ranking,  # isti modeli koje evaluiraš
        n_neg=99,
        seed=43,
    )
    tie_table_test.to_csv(os.path.join(out_dir, "tie_rate_test_table.csv"), index=False)
    print(f"[Saved] {os.path.join(out_dir, 'tie_rate_test_table.csv')}")

    plots.plot_tie_rate_bar(
        tie_table=tie_table_test,
        out_dir=out_dir,
        filename="tie_rate_test_bar.png",
        title="Tie-rate na TEST-u (ties > 1) po modelu",
    )

    final_table = rating_table.merge(ranking_table, on="model", how="outer")  # spoji rating+ranking.
    print("\n[FINAL] Poređenje modela (rating + ranking):")  # naslov.
    print(final_table)  # ispis.
    final_table.to_csv(os.path.join(out_dir, "final_models_comparison_fixed.csv"), index=False)  # snimi.

    print(f"\n[Saved] {os.path.join(out_dir, 'rating_metrics_table.csv')}")  # putanja.
    print(f"[Saved] {os.path.join(out_dir, 'ranking_metrics_table_fixed.csv')}")  # putanja.
    print(f"[Saved] {os.path.join(out_dir, 'final_models_comparison_fixed.csv')}")  # putanja.

    # =========================
    # 8.4) DODATO: Hybrid grid search (mf_factors × mf_reg × knn_topk × α)
    # =========================
    if run_hybrid_grid and item_knn_ready:
        print("\n" + "=" * 80)
        print("[Hybrid Grid] Pokrećem grid search za hybrid model "
              "(očekivano trajanje: 15–30 min, zavisno od dataseta)...")
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
                out_dir=out_dir,
            )
            if not df_grid.empty:
                # Plot α curve at the best (mf_factors, mf_reg, knn_topk) slice
                k_grid = 10
                sub = df_grid[
                    (df_grid["mf_factors"] == int(best_cfg["mf_factors"])) &
                    (df_grid["mf_reg"] == float(best_cfg["mf_reg"])) &
                    (df_grid["knn_topk"] == int(best_cfg["knn_topk"]))
                ].copy()
                plots.plot_hybrid_alpha_curve(
                    sub, out_dir=out_dir, k=k_grid,
                    filename="hybrid_alpha_curve_best_config.png",
                    title=(
                        f"Hybrid α sweep at best grid config "
                        f"(factors={best_cfg['mf_factors']}, reg={best_cfg['mf_reg']:.2f}, "
                        f"top_k={best_cfg['knn_topk']})"
                    ),
                )
                plots.plot_hybrid_grid_heatmap(df_grid, out_dir=out_dir, k=k_grid)
        except Exception as e:
            print(f"[WARN] Hybrid grid search nije završen: {e}")
    elif not run_hybrid_grid:
        print("\n[Hybrid Grid] Preskočen (run_hybrid_grid=False).")
    else:
        print("\n[Hybrid Grid] Preskočen (Item-kNN nije spreman).")


# =========================
# 9) Pokretanje
# =========================

if __name__ == "__main__":  # Entry point.
    try:
        raw = dataprep.load_raw_reviews()  # Učitaj raw.
        df_raw = raw  # Raw DF.
        prepare_and_run_all(df_raw, out_dir="eda_outputs")  # Start pipeline.
    except DATA_ERRORS as e:
        sys.exit(f"{type(e).__name__}: {e}")
