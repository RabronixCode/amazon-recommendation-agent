"""demo.py

Pronalazi reprezentativne korisnike za demonstraciju agenta, po jedan za svaki
scenario koji agent razlikuje u kodu (agent_cli.recommend_top_n_for_user).

Kategorije:
  A) Poznati korisnici (postoje u treniranom modelu, personalizovan tok)
     A1  heavy        korisnik sa najdužom TRAIN istorijom
     A2  typical      korisnik sa medijanskom TRAIN istorijom
     A3  light        korisnik sa najkraćom TRAIN istorijom
     A4  knn_rich     Item-kNN: najviše preporuka iz Top-N ima stvarne susede u istoriji
     A5  knn_fallback Item-kNN: najviše preporuka iz Top-N ide na fallback μ+bu+bi
     A6  hit          Item-kNN: korisnikov stvarni TEST artikal se nalazi u Top-N
  B) Nepoznati korisnici (cold-start tok, μ + bi)
     B1  new          ID koji nikad nije postojao u podacima
     B2  filtered_low      postojao u očišćenim podacima, ali ispod min_user praga
     B3  filtered_cascade  imao >= min_user interakcija, ali ispao iterativnim
                           filtriranjem jer su mu artikli bili previše retki
  (B2 i B3 zahtevaju --with-raw, jer se sirovi podaci ponovo učitavaju.)

Pokretanje (iz istog foldera kao agent_cli.py, posle `agent_cli.py --retrain`):
  python demo.py
  python demo.py --with-raw
  python demo.py --sample 300 --topn 10 --with-raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agent_cli import ART_DIR, load_artifacts, recommend_top_n_for_user
from dataprep import build_splits_leave_one_out, load_raw_reviews, map_and_clean

OUT_FILE = ART_DIR / "demo_users.txt"


def knn_support(u_idx: int, i_idx: int, payload: dict[str, object]) -> int:
    """Broj suseda artikla i_idx koje je korisnik ocenio u TRAIN-u.

    Ako je 0, predict_item_knn pada na fallback μ + bu + bi (den == 0).
    Logika je ista kao u predict_item_knn: preskaču se ne-konačne sličnosti.
    """
    rated = payload["user_hist"].get(u_idx, {})
    cnt = 0
    for j, s in zip(payload["neigh_idx"][i_idx], payload["neigh_sim"][i_idx]):
        if np.isfinite(s) and float(s) != 0.0 and int(j) in rated:
            cnt += 1
    return cnt


def pick_by_history(payload: dict[str, object]) -> dict[str, tuple[str, str]]:
    """A1-A3: izbor po dužini TRAIN istorije (brzo, bez skoringa)."""
    user_ids = payload["idx_maps"]["user_index_to_id"]
    hist = payload["user_hist"]
    counts = pd.Series({u: len(items) for u, items in hist.items()}).sort_index()

    heavy = int(counts.idxmax())
    light = int(counts.idxmin())
    med_val = counts.median()
    typical = int((counts - med_val).abs().idxmin())

    return {
        "A1_heavy": (user_ids[heavy], f"train={counts[heavy]} interakcija (maksimum)"),
        "A2_typical": (user_ids[typical], f"train={counts[typical]} interakcija (medijana={med_val:g})"),
        "A3_light": (user_ids[light], f"train={counts[light]} interakcija (minimum)"),
    }


def pick_by_knn_behavior(
    payload: dict[str, object], sample: int, top_n: int, seed: int
) -> dict[str, tuple[str, str]]:
    """A4-A6: pokreće stvarni Item-kNN Top-N za uzorak korisnika i meri ponašanje."""
    user_ids = payload["idx_maps"]["user_index_to_id"]
    df_cut = payload["df_cut"]
    _, _, df_test = build_splits_leave_one_out(df_cut)
    test_item = dict(zip(df_test["user_index"].astype(int), df_test["item_index"].astype(int)))

    rng = np.random.default_rng(seed)
    n_users = len(user_ids)
    sample_idx = rng.choice(n_users, size=min(sample, n_users), replace=False)

    rows = []
    for k, u_idx in enumerate(sample_idx, start=1):
        u_idx = int(u_idx)
        resp = recommend_top_n_for_user(user_ids[u_idx], payload, model_name="Item-kNN", top_n=top_n)
        rec_items = [int(r["item_index"]) for r in resp["results"]]
        supported = sum(1 for i in rec_items if knn_support(u_idx, i, payload) > 0)
        t_item = test_item.get(u_idx)
        hit_rank = rec_items.index(t_item) + 1 if t_item in rec_items else None
        rows.append({
            "u_idx": u_idx,
            "train_len": len(payload["user_hist"].get(u_idx, {})),
            "supported": supported,
            "hit_rank": hit_rank,
        })
        if k % 50 == 0:
            print(f"[Scan] {k}/{len(sample_idx)} korisnika obrađeno...")

    df = pd.DataFrame(rows)
    out: dict[str, tuple[str, str]] = {}

    rich = df.sort_values(["supported", "train_len"], ascending=[False, False]).iloc[0]
    out["A4_knn_rich"] = (
        user_ids[int(rich["u_idx"])],
        f"{int(rich['supported'])}/{top_n} preporuka ima susede u istoriji, train={int(rich['train_len'])}",
    )

    fb = df.sort_values(["supported", "train_len"], ascending=[True, True]).iloc[0]
    out["A5_knn_fallback"] = (
        user_ids[int(fb["u_idx"])],
        f"samo {int(fb['supported'])}/{top_n} preporuka ima susede (ostalo fallback μ+bu+bi), train={int(fb['train_len'])}",
    )

    hits = df[df["hit_rank"].notna()].sort_values("hit_rank")
    if len(hits) > 0:
        h = hits.iloc[0]
        out["A6_hit"] = (
            user_ids[int(h["u_idx"])],
            f"TEST artikal je na poziciji {int(h['hit_rank'])} u Top-{top_n} (Item-kNN)",
        )
    else:
        out["A6_hit"] = ("", f"nijedan hit u uzorku od {len(df)}; povećaj --sample")

    print(f"[Scan] Udeo korisnika sa hitom u uzorku: {df['hit_rank'].notna().mean():.3f}")
    return out


def pick_filtered_out(payload: dict[str, object], min_user: int) -> dict[str, tuple[str, str]]:
    """B2-B3: korisnici koji postoje u očišćenim podacima, ali nisu u modelu."""
    df = map_and_clean(load_raw_reviews())
    counts = df.groupby("user_id").size()
    in_model = set(payload["idx_maps"]["user_index_to_id"])
    not_in_model = counts[~counts.index.isin(in_model)]

    out: dict[str, tuple[str, str]] = {}

    low = not_in_model[not_in_model == min_user - 1]
    if len(low) > 0:
        uid = str(sorted(low.index)[0])
        out["B2_filtered_low"] = (uid, f"{min_user - 1} interakcije u očišćenim podacima (< min_user={min_user})")
    else:
        out["B2_filtered_low"] = ("", "nije pronađen")

    cascade = not_in_model[not_in_model >= min_user].sort_values(ascending=False)
    if len(cascade) > 0:
        uid = str(cascade.index[0])
        out["B3_filtered_cascade"] = (
            uid,
            f"{int(cascade.iloc[0])} interakcija (>= min_user), ali ispao zbog retkih artikala; "
            f"ukupno takvih korisnika: {len(cascade)}",
        )
    else:
        out["B3_filtered_cascade"] = ("", "nema takvih korisnika")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Izbor reprezentativnih korisnika za demo agenta.")
    parser.add_argument("--sample", type=int, default=200, help="Broj korisnika za Item-kNN skeniranje (A4-A6)")
    parser.add_argument("--topn", type=int, default=10, help="Top-N kao u agentu")
    parser.add_argument("--seed", type=int, default=42, help="Seed za uzorak korisnika")
    parser.add_argument("--with-raw", action="store_true", help="Učitaj sirove podatke i nađi B2/B3 (sporije)")
    args = parser.parse_args()

    payload = load_artifacts()
    if payload is None:
        raise SystemExit("Nema artefakata. Prvo pokreni: python agent_cli.py --retrain")

    min_user = int(payload.get("min_user", 4))

    picks: dict[str, tuple[str, str]] = {}
    picks.update(pick_by_history(payload))
    picks.update(pick_by_knn_behavior(payload, sample=args.sample, top_n=args.topn, seed=args.seed))
    picks["B1_new"] = ("NEW_USER_123", "ID ne postoji ni u sirovim podacima")
    if args.with_raw:
        picks.update(pick_filtered_out(payload, min_user=min_user))

    lines = []
    print("\n" + "=" * 80)
    print("Reprezentativni korisnici za demo")
    print("=" * 80)
    for key in sorted(picks):
        uid, note = picks[key]
        print(f"{key:<20} {uid:<24} {note}")
        if uid:
            lines.append(f"{key}\t{uid}\t{note}\tpython agent_cli.py --user {uid}")
    print("=" * 80)

    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Saved] {OUT_FILE}")


if __name__ == "__main__":
    main()