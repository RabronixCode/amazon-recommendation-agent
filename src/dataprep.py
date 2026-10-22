from __future__ import annotations

import ast
import gzip
import json

import pandas as pd

from config import REQUIRED_COLUMNS, resolve_raw_reviews_path

# Amazon column names -> internal names. The only place where this mapping is defined.
COLUMN_MAPPING = {
    "reviewerID": "user_id",
    "asin": "item_id",
    "overall": "rating",
    "unixReviewTime": "timestamp",
}


def load_raw_reviews() -> pd.DataFrame:
    """Load the raw review file (JSON lines, optionally gzipped), tolerating malformed lines."""
    # Line-by-line parsing: pandas and ujson fail on malformed lines (for example unescaped quotes
    # in review text). Bad lines are skipped, and ast.literal_eval is tried as a fallback for the
    # old Amazon format where each line is a Python dict literal (single quotes).
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
        print(f"[Warn] Skipped {bad} malformed lines while loading {path}.")
    print(f"[Info] Loaded {len(records)} rows from {path}.")
    return pd.DataFrame.from_records(records)


def map_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Select the required columns, rename them, fix dtypes, drop nulls and exact duplicates."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available: {list(df_raw.columns)}")

    df = df_raw[list(COLUMN_MAPPING)].rename(columns=COLUMN_MAPPING)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["rating"] = df["rating"].astype(float)
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    return df.drop_duplicates()


def iterative_filter_min_interactions(
    df: pd.DataFrame,
    min_user_interactions,  # minimum interactions per user
    min_item_interactions,  # minimum interactions per item
    max_iters: int = 20,  # safety limit so the loop always terminates
) -> pd.DataFrame:
    """Repeatedly drop users and items below the thresholds until the data is stable."""
    df_out = df.copy()
    for _ in range(max_iters):
        before = len(df_out)
        df_out = df_out.groupby("user_id").filter(lambda x: len(x) >= min_user_interactions)
        df_out = df_out.groupby("item_id").filter(lambda x: len(x) >= min_item_interactions)
        after = len(df_out)
        if after == before:
            break
    return df_out


def reindex_after_filter(df_cut: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list]]:
    """Add contiguous user_index / item_index columns and return the index -> original id maps."""
    df_cut = df_cut.copy()
    user_cats = df_cut["user_id"].astype("category")
    item_cats = df_cut["item_id"].astype("category")
    df_cut["user_index"] = user_cats.cat.codes.astype(int)
    df_cut["item_index"] = item_cats.cat.codes.astype(int)
    idx_maps = {
        "user_index_to_id": list(user_cats.cat.categories),
        "item_index_to_id": list(item_cats.cat.categories),
    }
    return df_cut, idx_maps


def build_splits_leave_one_out(df_cut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal leave-one-out: last interaction is test, the one before is validation."""
    df_cut = df_cut.sort_values(["user_id", "timestamp"], ascending=[True, True]).copy()
    n = df_cut.groupby("user_id")["rating"].transform("size")  # interactions per user
    r = df_cut.groupby("user_id").cumcount()  # position within the user's history
    df_cut["split"] = "train"
    df_cut.loc[r == (n - 1), "split"] = "test"
    df_cut.loc[r == (n - 2), "split"] = "val"
    df_train = df_cut[df_cut["split"] == "train"].reset_index(drop=True)
    df_val = df_cut[df_cut["split"] == "val"].reset_index(drop=True)
    df_test = df_cut[df_cut["split"] == "test"].reset_index(drop=True)
    return df_train, df_val, df_test


def compute_dataset_profile(df: pd.DataFrame, label: str) -> dict[str, object]:
    """Snapshot of dataset size and density."""
    n_interactions = int(len(df))
    n_users = int(df["user_id"].nunique())
    n_items = int(df["item_id"].nunique())
    density = float(n_interactions) / (float(n_users) * float(n_items)) if n_users and n_items else float("nan")
    return {"label": label, "interactions": n_interactions, "users": n_users, "items": n_items, "density": density}


def make_filtering_report_table(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """Table comparing the dataset before and after filtering."""
    return pd.DataFrame([compute_dataset_profile(df_before, "before_filter"), compute_dataset_profile(df_after, "after_filter")])
