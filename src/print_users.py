"""print_users.py

Snima u tekstualni fajl SVE reviewerID-eve koje stvarno koristi trenirani agent
(tj. korisnike koji su preživeli iterativno filtriranje min_user/min_item i imaju
indeks u finalnom modelu).

Strategija (jedan izvor istine = agent_cli):
  1) Ako postoji `agent_artifacts/agent_itemknn.pkl` (rezultat treniranja
     `agent_cli.py`), koristi `idx_maps["user_index_to_id"]` direktno odatle —
     to su tačno korisnici koje agent može da prepozna.
  2) Ako pickle ne postoji (ili je zadat --rebuild), pozovi pipeline iz
     `agent_cli` (load_raw_reviews → map_and_clean → iterative_filter_min_interactions
     → reindex_after_filter) sa istim parametrima.

Ovako je `filtered_users.txt` UVEK 1:1 sa onim što agent zna; više nema 30k
duhova iz starog `pd.read_json` pipeline-a.
"""

from __future__ import annotations

import argparse
import os

from agent_cli import ART_FILE, load_artifacts
from config import MIN_ITEM_INTERACTIONS, MIN_USER_INTERACTIONS
from dataprep import iterative_filter_min_interactions, load_raw_reviews, map_and_clean, reindex_after_filter


def get_filtered_users(min_user: int, min_item: int, rebuild: bool) -> list[str]:
    """Vrati listu reviewerID-eva koje agent zaista koristi.

    - Default: čita iz keša (`agent_artifacts/agent_itemknn.pkl`) ako postoji,
      jer to je tačno snapshot treniranog modela.
    - Sa `rebuild=True` ili ako keš ne postoji: pokreće isti filter pipeline
      koji koristi `prepare_agent_model()` u `agent_cli.py`.
    """
    if not rebuild:
        payload = load_artifacts()
        if payload is not None:
            user_ids = list(payload["idx_maps"]["user_index_to_id"])
            print(f"[Source] Učitano iz artefakata: {ART_FILE}")
            return [str(u) for u in user_ids]

    print("[Source] Artefakti nisu nađeni (ili --rebuild). Pokrećem filter pipeline iz agent_cli.")
    df_raw = load_raw_reviews()
    df = map_and_clean(df_raw)
    df = df.sort_values("timestamp", ascending=True)
    df_cut = iterative_filter_min_interactions(df, min_user_interactions=min_user, min_item_interactions=min_item)
    _, idx_maps = reindex_after_filter(df_cut)
    return [str(u) for u in idx_maps["user_index_to_id"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save reviewerID-eve koje agent stvarno koristi (1:1 sa treniranim modelom)."
    )
    parser.add_argument("--min_user", type=int, default=MIN_USER_INTERACTIONS, help="Min interakcija po useru (samo kod --rebuild)")
    parser.add_argument("--min_item", type=int, default=MIN_ITEM_INTERACTIONS, help="Min interakcija po itemu (samo kod --rebuild)")
    parser.add_argument("--out", type=str, default="filtered_users.txt", help="Izlazni fajl sa reviewerID-evima")
    parser.add_argument("--check", type=str, default=None, help="Provera: da li je dati user_id u modelu")
    parser.add_argument("--head", type=int, default=0, help="Ako >0, ispiši i prvih N user-a u konzolu")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ne koristi keširane artefakte; pokreni filter pipeline iz početka (sporo)",
    )
    args = parser.parse_args()

    users = get_filtered_users(min_user=args.min_user, min_item=args.min_item, rebuild=args.rebuild)
    users_sorted = sorted(users)

    print(f"[Filtered] users={len(users_sorted)}  (min_user={args.min_user}, min_item={args.min_item})")

    if args.check is not None:
        u = str(args.check).strip()
        ok = u in set(users_sorted)
        print(f"[Check] user_id='{u}' -> {'FOUND' if ok else 'NOT FOUND'}")

    out_path = os.path.abspath(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        for u in users_sorted:
            f.write(f"{u}\n")
    print(f"[Saved] {len(users_sorted)} reviewerID-eva u: {out_path}")

    if args.head and args.head > 0:
        print(f"\nPrvih {min(args.head, len(users_sorted))} reviewerID-eva:")
        for u in users_sorted[: args.head]:
            print(u)


if __name__ == "__main__":
    main()
