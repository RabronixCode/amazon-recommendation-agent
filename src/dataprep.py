from __future__ import annotations  # Forward type hints.

import pandas as pd
import gzip
import ast
import json

from config import REQUIRED_COLUMNS, resolve_raw_reviews_path  # Putanja i obavezne kolone.

# Amazon nazivi kolona -> interna imena. Jedino mesto gde se ovo mapiranje definise.
COLUMN_MAPPING = {
    "reviewerID": "user_id",
    "asin": "item_id",
    "overall": "rating",
    "unixReviewTime": "timestamp",
}


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


def iterative_filter_min_interactions(
    df: pd.DataFrame,  # Interakcije.
    min_user_interactions,  # Minimum po korisniku.
    min_item_interactions, # Minimum po artiklu.
    max_iters: int = 20,  # Maks iteracija da ne zapne.
) -> pd.DataFrame:  # Vraća filtriran DF.
    df_out = df.copy()  # Kopija.
    for _ in range(max_iters):  # Iteracije dok se ne stabilizuje.
        before = len(df_out)  # Broj redova pre.
        df_out = df_out.groupby("user_id").filter(lambda x: len(x) >= min_user_interactions)  # Skini user-e ispod praga.
        df_out = df_out.groupby("item_id").filter(lambda x: len(x) >= min_item_interactions)  # Skini item-e ispod praga.
        after = len(df_out)  # Broj redova posle.
        if after == before:  # Ako nema promene -> stabilno.
            break  # Stop.
    return df_out  # Vrati stabilno filtriran DF.


def build_splits_leave_one_out(df_cut: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # LOO split po useru.
    df_cut = df_cut.sort_values(["user_id", "timestamp"], ascending=[True, True]).copy()  # Sort po user+vreme.
    n = df_cut.groupby("user_id")["rating"].transform("size")  # Broj interakcija po useru.
    r = df_cut.groupby("user_id").cumcount()  # Redni broj interakcije unutar usera.
    df_cut["split"] = "train"  # Default train.
    df_cut.loc[r == (n - 1), "split"] = "test"  # Poslednja interakcija -> test.
    df_cut.loc[r == (n - 2), "split"] = "val"  # Pretposlednja -> val.
    df_train = df_cut[df_cut["split"] == "train"].reset_index(drop=True)  # Train DF.
    df_val = df_cut[df_cut["split"] == "val"].reset_index(drop=True)  # Val DF.
    df_test = df_cut[df_cut["split"] == "test"].reset_index(drop=True)  # Test DF.
    return df_train, df_val, df_test  # Vrati.


def compute_dataset_profile(df: pd.DataFrame, label: str) -> dict[str, object]:  # Snapshot metrika.
    n_interactions = int(len(df))  # # interakcija.
    n_users = int(df["user_id"].nunique())  # # korisnika.
    n_items = int(df["item_id"].nunique())  # # itema.
    density = float(n_interactions) / (float(n_users) * float(n_items)) if n_users and n_items else float("nan")  # Gustina.
    return {"label": label, "interactions": n_interactions, "users": n_users, "items": n_items, "density": density}  # Vrati.


def make_filtering_report_table(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([compute_dataset_profile(df_before, "before_filter"), compute_dataset_profile(df_after, "after_filter")])  # Tabela.


def map_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Uzima obavezne kolone, preimenuje ih, pretvara tipove, izbacuje null i duplikate."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Nedostaju kolone {missing}. Dostupno: {list(df_raw.columns)}")

    df = df_raw[list(COLUMN_MAPPING)].rename(columns=COLUMN_MAPPING)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["rating"] = df["rating"].astype(float)
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    return df.drop_duplicates()


def reindex_after_filter(df_cut: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list]]:
    """Dodaje user_index / item_index (0..n-1) i vraca mape index -> originalni id."""
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
