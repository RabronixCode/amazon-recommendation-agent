from __future__ import annotations  # Forward type hints.

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

try:  # Sparse support (needed for item-kNN).
    from scipy.sparse import csr_matrix  # CSR matrices.
except ImportError:  # scipy is not installed.
    csr_matrix = None  # Mark sparse support as unavailable.

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
    df_train: pd.DataFrame,  # Training split.
    mu: float,  # Global mean.
    n_users: int,  # #users.
    n_items: int,  # #items.
    reg: float = 10.0,  # L2 regularization.
    n_iters: int = 20,  # Number of iterations.
) -> tuple[np.ndarray, np.ndarray]:  # Returns bu, bi.
    bu = np.zeros(n_users, dtype=np.float32)  # User bias.
    bi = np.zeros(n_items, dtype=np.float32)  # Item bias.
    ug = df_train.groupby("user_index")  # Group by user.
    ig = df_train.groupby("item_index")  # Group by item.
    for _ in range(n_iters):  # Coordinate descent iterations.
        for u, g in ug:  # Update user biases.
            items = g["item_index"].to_numpy(dtype=int)  # Item indices.
            resid = g["rating"].to_numpy(dtype=float) - mu - bi[items]  # Residual without bu.
            bu[int(u)] = float(resid.sum() / (reg + len(g)))  # Closed-form update.
        for i, g in ig:  # Update item biases.
            users = g["user_index"].to_numpy(dtype=int)  # User indices.
            resid = g["rating"].to_numpy(dtype=float) - mu - bu[users]  # Residual without bi.
            bi[int(i)] = float(resid.sum() / (reg + len(g)))  # Closed-form update.
    return bu, bi  # Return the parameters.

def pred_global_mean(df: pd.DataFrame, mu: float) -> np.ndarray:  # Global mean prediction.
    return np.full(len(df), mu, dtype=float)  # Constant.


def pred_user_mean(df: pd.DataFrame, user_mean_arr: np.ndarray) -> np.ndarray:  # User mean prediction.
    return user_mean_arr[df["user_index"].to_numpy(dtype=int)].astype(float)  # Per user.


def pred_item_mean(df: pd.DataFrame, item_mean_arr: np.ndarray) -> np.ndarray:  # Item mean prediction.
    return item_mean_arr[df["item_index"].to_numpy(dtype=int)].astype(float)  # Per item.


def pred_bias(df: pd.DataFrame, mu: float, bu: np.ndarray, bi: np.ndarray) -> np.ndarray:  # Bias prediction.
    u = df["user_index"].to_numpy(dtype=int)  # u.
    i = df["item_index"].to_numpy(dtype=int)  # i.
    return np.clip(mu + bu[u] + bi[i], 1.0, 5.0).astype(float)  # mu + bu + bi, clipped to the rating range.


def train_biased_mf_sgd(
    df_train: pd.DataFrame,  # Train.
    df_val: pd.DataFrame,  # Validation split (used to pick the best epoch).
    n_users: int,  # #users.
    n_items: int,  # #items.
    factors: int = 32,  # Latent dimension.
    lr: float = 0.02,  # Learning rate.
    reg: float = 0.05,  # L2.
    n_epochs: int = 10,  # Number of epochs.
    seed: int = 42,  # Seed.
) -> dict[str, object]:  # Returns the best state.
    rng = np.random.default_rng(seed)  # RNG.
    mu = float(df_train["rating"].mean())  # μ.
    P = (0.1 * rng.standard_normal((n_users, factors))).astype(np.float32)  # User factors.
    Q = (0.1 * rng.standard_normal((n_items, factors))).astype(np.float32)  # Item factors.
    bu = np.zeros(n_users, dtype=np.float32)  # bu.
    bi = np.zeros(n_items, dtype=np.float32)  # bi.

    tr_u = df_train["user_index"].to_numpy(dtype=int)  # user indices.
    tr_i = df_train["item_index"].to_numpy(dtype=int)  # item indices.
    tr_r = df_train["rating"].to_numpy(dtype=np.float32)  # rating.
    order = np.arange(len(df_train), dtype=int)  # order used for shuffling.

    best_val = float("inf")  # best validation RMSE.
    best_state = None  # best parameters.
    val_true = df_val["rating"].to_numpy(dtype=float)

    for epoch in range(1, n_epochs + 1):  # epochs.
        rng.shuffle(order)  # shuffle.
        for idx in order:  # SGD pass.
            u = tr_u[idx]  # user.
            i = tr_i[idx]  # item.
            r_ui = tr_r[idx]  # rating.
            pred = mu + bu[u] + bi[i] + float(np.dot(P[u], Q[i]))  # pred.
            err = r_ui - pred  # error.
            bu[u] += lr * (err - reg * bu[u])  # update bu.
            bi[i] += lr * (err - reg * bi[i])  # update bi.
            Pu = P[u].copy()  # cache the old user vector.
            P[u] += lr * (err * Q[i] - reg * P[u])  # update P.
            Q[i] += lr * (err * Pu - reg * Q[i])  # update Q.

        val_rmse = float(root_mean_squared_error(val_true, mf_predict_ratings(df_val, mu, bu, bi, P, Q)))
        print(f"[MF] Epoch {epoch}/{n_epochs}  Val RMSE={val_rmse:.4f}")  # log.
        if val_rmse < best_val:  # new best?
            best_val = val_rmse  # set.
            best_state = (mu, bu.copy(), bi.copy(), P.copy(), Q.copy())  # save a copy.

    mu_b, bu_b, bi_b, P_b, Q_b = best_state  # unpack.
    return {"mu": mu_b, "bu": bu_b, "bi": bi_b, "P": P_b, "Q": Q_b, "best_val_rmse": best_val}  # return.

def mf_predict_ratings(df: pd.DataFrame, mu: float, bu: np.ndarray, bi: np.ndarray,
                       P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    u = df["user_index"].to_numpy(dtype=int)
    i = df["item_index"].to_numpy(dtype=int)
    return np.clip(mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1), 1.0, 5.0)

def mf_score_candidates(mu: float, bu: np.ndarray, bi: np.ndarray, P: np.ndarray, Q: np.ndarray, u_idx: int, cand_items: np.ndarray) -> np.ndarray:  # MF scoring.
    return (mu + bu[u_idx] + bi[cand_items] + (Q[cand_items] @ P[u_idx])).astype(float)  # scores.


def build_sparse_deviation_matrix(
    df_train: pd.DataFrame,  # Training split.
    n_users: int,  # #users.
    n_items: int,  # #items.
    user_mean_arr: np.ndarray,  # user mean.
) -> "csr_matrix":  # CSR deviation matrix.
    if csr_matrix is None:  # If scipy is missing.
        raise ImportError("scipy is not available (required for sparse item-kNN).")  # Stop.
    rows = df_train["user_index"].to_numpy(dtype=int)  # user.
    cols = df_train["item_index"].to_numpy(dtype=int)  # item.
    vals = df_train["rating"].to_numpy(dtype=float) - user_mean_arr[rows]  # dev.
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))  # CSR.


def compute_item_topk_similarities(
    dev_mat: "csr_matrix",  # Dev matrix.
    top_k: int = 50,  # Top-K neighbors.
    block_size: int = 256,  # Block size.
) -> tuple[np.ndarray, np.ndarray]:  # neigh_idx, neigh_sim.
    item_user = dev_mat.T.tocsr()  # item×user.
    n_items = item_user.shape[0]  # #items.
    top_k = int(min(top_k, max(1, n_items - 1)))  # keeps argpartition valid
    neigh_idx = np.zeros((n_items, top_k), dtype=np.int32)  # indices.
    neigh_sim = np.zeros((n_items, top_k), dtype=np.float32)  # similarities.
    for start in range(0, n_items, block_size):  # blocks.
        end = min(start + block_size, n_items)  # block end.
        sims_block = cosine_similarity(item_user[start:end], item_user)  # cosine.
        for r in range(sims_block.shape[0]):  # each row in the block.
            item_i = start + r  # global id.
            sims_row = sims_block[r]  # similarity to all items.
            sims_row[item_i] = -np.inf  # remove self.
            top = np.argpartition(-sims_row, top_k)[:top_k]  # top-K candidates.
            top = top[np.argsort(-sims_row[top])]  # sort.
            neigh_idx[item_i, :] = top.astype(np.int32)  # store indices.
            neigh_sim[item_i, :] = sims_row[top].astype(np.float32)  # store similarities.
    return neigh_idx, neigh_sim  # return.


def predict_item_knn_one(
    u_idx: int,  # user.
    i_idx: int,  # item.
    user_hist: dict[int, dict[int, float]],  # history.
    user_mean_arr: np.ndarray,  # user mean.
    neigh_idx: np.ndarray,  # neighbors.
    neigh_sim: np.ndarray,  # similarities.
    fallback: float,  # fallback.
) -> float:  # prediction.
    global itemknn_valid_neigh_sum, itemknn_pred_count

    if u_idx not in user_hist:  # unknown user.
        return float(fallback)  # fallback.
    rated = user_hist[u_idx]  # rated items.
    u_mean = float(user_mean_arr[u_idx])  # u_mean.
    num = 0.0  # numerator.
    den = 0.0  # denominator.
    valid_k = 0  # number of valid neighbors that contribute to the prediction

    for j, s in zip(neigh_idx[i_idx], neigh_sim[i_idx]):  # neighbors.
        if not np.isfinite(s):  # skip -inf.
            continue  # next.
        if float(s) == 0.0:
            continue
        jj = int(j)  # idx.
        if jj in rated:  # the user rated this neighbor.
            dev = float(rated[jj] - u_mean)  # deviation.
            num += float(s) * dev  # weight it.
            den += abs(float(s))  # sum of weights.
            valid_k += 1  # the neighbor was actually used

    # update the statistics (once per prediction)
    itemknn_valid_neigh_sum += int(valid_k)
    itemknn_pred_count += 1

    if den == 0.0:  # no overlap.
        return float(fallback)  # fallback.
    pred = u_mean + (num / den)  # u_mean + weighted dev.
    return float(np.clip(pred, 1.0, 5.0))  # clip.


def compute_user_topk_similarities(
    dev_mat: "csr_matrix",  # user×item dev matrix (r_ui - mean_u).
    top_k: int = 50,  # Top-K neighbors per user.
    block_size: int = 256,  # Chunk size.
) -> tuple[np.ndarray, np.ndarray]:  # neigh_u_idx, neigh_u_sim
    """
    User-based kNN:
      - User vectors are mean-centered (r_ui - mean_u), i.e. dev_mat.
      - Similarity is the cosine similarity between rows (user vectors).
      - Precompute the top-K neighbors of every user (excluding the user itself).
    """
    n_users = dev_mat.shape[0]  # #users.
    neigh_u_idx = np.zeros((n_users, top_k), dtype=np.int32)  # neighbor indices.
    neigh_u_sim = np.zeros((n_users, top_k), dtype=np.float32)  # similarities.

    for start in range(0, n_users, block_size):  # blocks.
        end = min(start + block_size, n_users)  # block end.
        sims_block = cosine_similarity(dev_mat[start:end], dev_mat)  # user x user cosine similarity.
        for r in range(sims_block.shape[0]):  # each row in the block.
            u_i = start + r  # global user id.
            sims_row = sims_block[r]  # similarity to all items.
            sims_row[u_i] = -np.inf  # remove self.
            top = np.argpartition(-sims_row, top_k)[:top_k]  # top-K candidates.
            top = top[np.argsort(-sims_row[top])]  # sort.
            neigh_u_idx[u_i, :] = top.astype(np.int32)  # store indices.
            neigh_u_sim[u_i, :] = sims_row[top].astype(np.float32)  # store similarities.

    return neigh_u_idx, neigh_u_sim  # return.


def predict_user_knn_one(
    u_idx: int,  # user.
    i_idx: int,  # item.
    user_hist: dict[int, dict[int, float]],  # history: user -> (item -> rating).
    user_mean_arr: np.ndarray,  # user mean.
    neigh_u_idx: np.ndarray,  # neighbors per user.
    neigh_u_sim: np.ndarray,  # similarities per user.
    fallback: float,  # fallback.
) -> float:
    """
    Prediction for user-based CF (standard mean-centered weighted sum):
      \\hat r_{ui} = mean_u + ( Σ_{v in N_k(u) & v rated i} sim(u,v) * (r_vi - mean_v) ) / Σ |sim(u,v)|
    """
    if u_idx not in user_hist:  # unknown user.
        return float(fallback)

    u_mean = float(user_mean_arr[u_idx])  # mean_u.
    num = 0.0
    den = 0.0

    for v, s in zip(neigh_u_idx[u_idx], neigh_u_sim[u_idx]):  # neighbors of the user.
        if not np.isfinite(s):  # skip -inf.
            continue
        vv = int(v)
        # Did neighbor v rate item i?
        if vv in user_hist and int(i_idx) in user_hist[vv]:
            r_vi = float(user_hist[vv][int(i_idx)])
            dev_v = r_vi - float(user_mean_arr[vv])  # (r_vi - mean_v)
            num += float(s) * dev_v
            den += abs(float(s))

    if den == 0.0:  # no neighbor rated the item.
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
    """Mean rating per user (users without training ratings get the global mean)."""
    arr = np.full(n_users, float(df_train["rating"].mean()), dtype=np.float32)
    um = df_train.groupby("user_index")["rating"].mean()
    arr[um.index.to_numpy(int)] = um.to_numpy(np.float32)
    return arr


def build_user_history_ratings(df_train: pd.DataFrame) -> dict[int, dict[int, float]]:
    """user_index -> {item_index: rating} from the training split."""
    hist: dict[int, dict[int, float]] = {}
    for u, i, r in zip(
        df_train["user_index"].to_numpy(int),
        df_train["item_index"].to_numpy(int),
        df_train["rating"].to_numpy(float),
    ):
        hist.setdefault(int(u), {})[int(i)] = float(r)
    return hist
