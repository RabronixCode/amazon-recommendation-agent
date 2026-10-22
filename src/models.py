from __future__ import annotations  # Forward type hints.

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

try:  # Sparse podrška (za item-kNN).
    from scipy.sparse import csr_matrix  # CSR matrice.
except ImportError:  # Ako scipy nije instaliran.
    csr_matrix = None  # Obeleži da nema sparse.

itemknn_valid_neigh_sum = 0
itemknn_pred_count = 0

def itemknn_reset_neighbor_stats() -> None:
    global itemknn_valid_neigh_sum, itemknn_pred_count
    itemknn_valid_neigh_sum = 0
    itemknn_pred_count = 0

def itemknn_get_avg_valid_neighbors() -> float | None:
    if itemknn_pred_count == 0:
        return None
    return float(itemknn_valid_neigh_sum) / float(itemknn_pred_count)


def fit_bias_model_by_index(
    df_train: pd.DataFrame,  # Train.
    mu: float,  # Global mean.
    n_users: int,  # #users.
    n_items: int,  # #items.
    reg: float = 10.0,  # L2 regularizacija.
    n_iters: int = 20,  # Iteracije.
) -> tuple[np.ndarray, np.ndarray]:  # Vrati bu, bi.
    bu = np.zeros(n_users, dtype=np.float32)  # User bias.
    bi = np.zeros(n_items, dtype=np.float32)  # Item bias.
    ug = df_train.groupby("user_index")  # Grupisanje po useru.
    ig = df_train.groupby("item_index")  # Grupisanje po itemu.
    for _ in range(n_iters):  # Iteracije coordinate descent.
        for u, g in ug:  # Update bu.
            items = g["item_index"].to_numpy(dtype=int)  # Item indeksi.
            resid = g["rating"].to_numpy(dtype=float) - mu - bi[items]  # Residual bez bu.
            bu[int(u)] = float(resid.sum() / (reg + len(g)))  # Zatvorena forma.
        for i, g in ig:  # Update bi.
            users = g["user_index"].to_numpy(dtype=int)  # User indeksi.
            resid = g["rating"].to_numpy(dtype=float) - mu - bu[users]  # Residual bez bi.
            bi[int(i)] = float(resid.sum() / (reg + len(g)))  # Zatvorena forma.
    return bu, bi  # Vrati parametre.

def pred_global_mean(df: pd.DataFrame, mu: float) -> np.ndarray:  # Global mean predikcija.
    return np.full(len(df), mu, dtype=float)  # Konstantno.


def pred_user_mean(df: pd.DataFrame, user_mean_arr: np.ndarray) -> np.ndarray:  # User mean predikcija.
    return user_mean_arr[df["user_index"].to_numpy(dtype=int)].astype(float)  # Po useru.


def pred_item_mean(df: pd.DataFrame, item_mean_arr: np.ndarray) -> np.ndarray:  # Item mean predikcija.
    return item_mean_arr[df["item_index"].to_numpy(dtype=int)].astype(float)  # Po itemu.


def pred_bias(df: pd.DataFrame, mu: float, bu: np.ndarray, bi: np.ndarray) -> np.ndarray:  # Bias predikcija.
    u = df["user_index"].to_numpy(dtype=int)  # u.
    i = df["item_index"].to_numpy(dtype=int)  # i.
    return np.clip(mu + bu[u] + bi[i], 1.0, 5.0).astype(float)  # μ+bu+bi i clip.


def train_biased_mf_sgd(
    df_train: pd.DataFrame,  # Train.
    df_val: pd.DataFrame,  # Val (za best).
    n_users: int,  # #users.
    n_items: int,  # #items.
    factors: int = 32,  # Latent dim.
    lr: float = 0.02,  # Learning rate.
    reg: float = 0.05,  # L2.
    n_epochs: int = 10,  # Epohe.
    seed: int = 42,  # Seed.
) -> dict[str, object]:  # Vrati best state.
    rng = np.random.default_rng(seed)  # RNG.
    mu = float(df_train["rating"].mean())  # μ.
    P = (0.1 * rng.standard_normal((n_users, factors))).astype(np.float32)  # User faktori.
    Q = (0.1 * rng.standard_normal((n_items, factors))).astype(np.float32)  # Item faktori.
    bu = np.zeros(n_users, dtype=np.float32)  # bu.
    bi = np.zeros(n_items, dtype=np.float32)  # bi.

    tr_u = df_train["user_index"].to_numpy(dtype=int)  # u indeksi.
    tr_i = df_train["item_index"].to_numpy(dtype=int)  # i indeksi.
    tr_r = df_train["rating"].to_numpy(dtype=np.float32)  # rating.
    order = np.arange(len(df_train), dtype=int)  # redosled za shuffle.

    best_val = float("inf")  # best val rmse.
    best_state = None  # best params.
    val_true = df_val["rating"].to_numpy(dtype=float)

    for epoch in range(1, n_epochs + 1):  # epohe.
        rng.shuffle(order)  # shuffle.
        for idx in order:  # SGD.
            u = tr_u[idx]  # user.
            i = tr_i[idx]  # item.
            r_ui = tr_r[idx]  # rating.
            pred = mu + bu[u] + bi[i] + float(np.dot(P[u], Q[i]))  # pred.
            err = r_ui - pred  # error.
            bu[u] += lr * (err - reg * bu[u])  # update bu.
            bi[i] += lr * (err - reg * bi[i])  # update bi.
            Pu = P[u].copy()  # cache.
            P[u] += lr * (err * Q[i] - reg * P[u])  # update P.
            Q[i] += lr * (err * Pu - reg * Q[i])  # update Q.

        val_rmse = float(root_mean_squared_error(val_true, mf_predict_ratings(df_val, mu, bu, bi, P, Q)))
        print(f"[MF] Epoch {epoch}/{n_epochs}  Val RMSE={val_rmse:.4f}")  # log.
        if val_rmse < best_val:  # best?
            best_val = val_rmse  # set.
            best_state = (mu, bu.copy(), bi.copy(), P.copy(), Q.copy())  # save.

    mu_b, bu_b, bi_b, P_b, Q_b = best_state  # unpack.
    return {"mu": mu_b, "bu": bu_b, "bi": bi_b, "P": P_b, "Q": Q_b, "best_val_rmse": best_val}  # return.

def mf_predict_ratings(df: pd.DataFrame, mu: float, bu: np.ndarray, bi: np.ndarray,
                       P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    u = df["user_index"].to_numpy(dtype=int)
    i = df["item_index"].to_numpy(dtype=int)
    return np.clip(mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1), 1.0, 5.0)

def mf_score_candidates(mu: float, bu: np.ndarray, bi: np.ndarray, P: np.ndarray, Q: np.ndarray, u_idx: int, cand_items: np.ndarray) -> np.ndarray:  # MF scoring.
    return (mu + bu[u_idx] + bi[cand_items] + (Q[cand_items] @ P[u_idx])).astype(float)  # skorovi.


def build_sparse_deviation_matrix(
    df_train: pd.DataFrame,  # Train.
    n_users: int,  # #users.
    n_items: int,  # #items.
    user_mean_arr: np.ndarray,  # user mean.
) -> "csr_matrix":  # CSR dev matrica.
    if csr_matrix is None:  # Ako nema scipy.
        raise ImportError("scipy nije dostupan (potreban za sparse item-kNN).")  # Stop.
    rows = df_train["user_index"].to_numpy(dtype=int)  # user.
    cols = df_train["item_index"].to_numpy(dtype=int)  # item.
    vals = df_train["rating"].to_numpy(dtype=float) - user_mean_arr[rows]  # dev.
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))  # CSR.


def compute_item_topk_similarities(
    dev_mat: "csr_matrix",  # Dev matrix.
    top_k: int = 50,  # TopK susedi.
    block_size: int = 256,  # Chunk.
) -> tuple[np.ndarray, np.ndarray]:  # neigh_idx, neigh_sim.
    item_user = dev_mat.T.tocsr()  # item×user.
    n_items = item_user.shape[0]  # #items.
    top_k = int(min(top_k, max(1, n_items - 1)))  # da argpartition ne pukne
    neigh_idx = np.zeros((n_items, top_k), dtype=np.int32)  # indeksi.
    neigh_sim = np.zeros((n_items, top_k), dtype=np.float32)  # sim.
    for start in range(0, n_items, block_size):  # blokovi.
        end = min(start + block_size, n_items)  # kraj.
        sims_block = cosine_similarity(item_user[start:end], item_user)  # cosine.
        for r in range(sims_block.shape[0]):  # svaki u bloku.
            item_i = start + r  # global id.
            sims_row = sims_block[r]  # sim prema svima.
            sims_row[item_i] = -np.inf  # ukloni self.
            top = np.argpartition(-sims_row, top_k)[:top_k]  # topK kandidati.
            top = top[np.argsort(-sims_row[top])]  # sortiraj.
            neigh_idx[item_i, :] = top.astype(np.int32)  # snimi indekse.
            neigh_sim[item_i, :] = sims_row[top].astype(np.float32)  # snimi sim.
    return neigh_idx, neigh_sim  # vrati.


def predict_item_knn_one(
    u_idx: int,  # user.
    i_idx: int,  # item.
    user_hist: dict[int, dict[int, float]],  # istorija.
    user_mean_arr: np.ndarray,  # user mean.
    neigh_idx: np.ndarray,  # susedi.
    neigh_sim: np.ndarray,  # sim.
    fallback: float,  # fallback.
) -> float:  # predikcija.
    global itemknn_valid_neigh_sum, itemknn_pred_count

    if u_idx not in user_hist:  # nema user.
        return float(fallback)  # fallback.
    rated = user_hist[u_idx]  # rated items.
    u_mean = float(user_mean_arr[u_idx])  # u_mean.
    num = 0.0  # brojilac.
    den = 0.0  # imenilac.
    valid_k = 0  # broj "validnih suseda" koji doprinose predikciji

    for j, s in zip(neigh_idx[i_idx], neigh_sim[i_idx]):  # susedi.
        if not np.isfinite(s):  # preskoči -inf.
            continue  # next.
        if float(s) == 0.0:
            continue
        jj = int(j)  # idx.
        if jj in rated:  # user ocenio suseda.
            dev = float(rated[jj] - u_mean)  # devijacija.
            num += float(s) * dev  # ponderiši.
            den += abs(float(s))  # zbir težina.
            valid_k += 1  # sused je stvarno korišćen

    # update statistika (po jednoj predikciji)
    itemknn_valid_neigh_sum += int(valid_k)
    itemknn_pred_count += 1

    if den == 0.0:  # nema poklapanja.
        return float(fallback)  # fallback.
    pred = u_mean + (num / den)  # u_mean + weighted dev.
    return float(np.clip(pred, 1.0, 5.0))  # clip.


def compute_user_topk_similarities(
    dev_mat: "csr_matrix",  # user×item dev matrix (r_ui - mean_u).
    top_k: int = 50,  # TopK susedi po korisniku.
    block_size: int = 256,  # Chunk size.
) -> tuple[np.ndarray, np.ndarray]:  # neigh_u_idx, neigh_u_sim
    """
    User-based kNN:
      - Vektori korisnika su mean-centered (r_ui - mean_u), tj. dev_mat.
      - Sličnost je cosine_similarity između redova (user vectors).
      - Precompute topK suseda za svakog korisnika (bez self).
    """
    n_users = dev_mat.shape[0]  # #users.
    neigh_u_idx = np.zeros((n_users, top_k), dtype=np.int32)  # indeksi suseda.
    neigh_u_sim = np.zeros((n_users, top_k), dtype=np.float32)  # sličnosti.

    for start in range(0, n_users, block_size):  # blokovi.
        end = min(start + block_size, n_users)  # kraj.
        sims_block = cosine_similarity(dev_mat[start:end], dev_mat)  # cosine user×user.
        for r in range(sims_block.shape[0]):  # svaki u bloku.
            u_i = start + r  # global id korisnika.
            sims_row = sims_block[r]  # sim prema svima.
            sims_row[u_i] = -np.inf  # ukloni self.
            top = np.argpartition(-sims_row, top_k)[:top_k]  # topK kandidati.
            top = top[np.argsort(-sims_row[top])]  # sortiraj.
            neigh_u_idx[u_i, :] = top.astype(np.int32)  # snimi indekse.
            neigh_u_sim[u_i, :] = sims_row[top].astype(np.float32)  # snimi sim.

    return neigh_u_idx, neigh_u_sim  # vrati.


def predict_user_knn_one(
    u_idx: int,  # user.
    i_idx: int,  # item.
    user_hist: dict[int, dict[int, float]],  # istorija user->(item->rating).
    user_mean_arr: np.ndarray,  # user mean.
    neigh_u_idx: np.ndarray,  # susedi po korisniku.
    neigh_u_sim: np.ndarray,  # sličnosti po korisniku.
    fallback: float,  # fallback.
) -> float:
    """
    Predikcija za user-based CF (standardni mean-centered weighted sum):
      \\hat r_{ui} = mean_u + ( Σ_{v in N_k(u) & v rated i} sim(u,v) * (r_vi - mean_v) ) / Σ |sim(u,v)|
    """
    if u_idx not in user_hist:  # nema usera.
        return float(fallback)

    u_mean = float(user_mean_arr[u_idx])  # mean_u.
    num = 0.0
    den = 0.0

    for v, s in zip(neigh_u_idx[u_idx], neigh_u_sim[u_idx]):  # susedi korisnika.
        if not np.isfinite(s):  # preskoči -inf.
            continue
        vv = int(v)
        # Da li komšija v ima rating za item i?
        if vv in user_hist and int(i_idx) in user_hist[vv]:
            r_vi = float(user_hist[vv][int(i_idx)])
            dev_v = r_vi - float(user_mean_arr[vv])  # (r_vi - mean_v)
            num += float(s) * dev_v
            den += abs(float(s))

    if den == 0.0:  # nema nijednog komšije koji je ocenio item.
        return float(fallback)

    pred = u_mean + (num / den)
    return float(np.clip(pred, 1.0, 5.0))


def user_knn_predict_df(
    df: pd.DataFrame,
    user_hist: dict[int, dict[int, float]],
    user_mean_arr: np.ndarray,
    neigh_u_idx: np.ndarray,
    neigh_u_sim: np.ndarray,
    fallback_fn,  # fn(u_idx, i_idx)->fallback
) -> np.ndarray:
    preds = np.zeros(len(df), dtype=float)
    u_arr = df["user_index"].to_numpy(int)
    i_arr = df["item_index"].to_numpy(int)
    for t in range(len(df)):
        u = int(u_arr[t])
        i = int(i_arr[t])
        fb = float(fallback_fn(u, i))
        preds[t] = predict_user_knn_one(u, i, user_hist, user_mean_arr, neigh_u_idx, neigh_u_sim, fb)
    return preds


def build_user_mean(df_train: pd.DataFrame, n_users: int) -> np.ndarray:
    """Prosek ocena po korisniku (korisnici bez train ocena dobijaju globalni prosek)."""
    arr = np.full(n_users, float(df_train["rating"].mean()), dtype=np.float32)
    um = df_train.groupby("user_index")["rating"].mean()
    arr[um.index.to_numpy(int)] = um.to_numpy(np.float32)
    return arr


def build_user_history_ratings(df_train: pd.DataFrame) -> dict[int, dict[int, float]]:
    """user_index -> {item_index: rating} iz train skupa."""
    hist: dict[int, dict[int, float]] = {}
    for u, i, r in zip(
        df_train["user_index"].to_numpy(int),
        df_train["item_index"].to_numpy(int),
        df_train["rating"].to_numpy(float),
    ):
        hist.setdefault(int(u), {})[int(i)] = float(r)
    return hist
