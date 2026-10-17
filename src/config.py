# Putanje i osnovna podesavanja
from __future__ import annotations

import os
from pathlib import Path

# Root projekta = jedan nivo iznad src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATTERNS = ("*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz")

REQUIRED_COLUMNS = ("reviewerID", "asin", "overall", 'unixReviewTime')

class DatasetNotFoundError(FileNotFoundError):
    """Podaci ne postoje na očekivanom mestu."""

class TooManyFilesInDataFolderError(ValueError):
    """Broj fajlova u data folderu iz kojih je moguće čitati je veći od 1!"""

class FolderNotFoundError(FileNotFoundError):
    """Data folder ne postoji!"""

DATA_ERRORS = (DatasetNotFoundError, TooManyFilesInDataFolderError, FolderNotFoundError)

def resolve_raw_reviews_path() -> Path:
    """Resolves dataset path with env override and safe fallbacks."""
    env_path = os.environ.get("RAW_REVIEWS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    if DATA_DIR.is_dir():
        found = sorted(p for pattern in DATA_PATTERNS for p in DATA_DIR.glob(pattern))
        if len(found) == 1:
            return found[0]
        elif len(found) > 1:
            raise TooManyFilesInDataFolderError("Broj fajlova iz kojih je moguće čitati je veći od 1!" \
            " Postarajte se da postoji samo jedan fajl tipa .json, .jsonl, .json.gz ili .jsonl.gz" \
            " gde jedna linija predstavlja jednu recenziju.")
        else:
            raise DatasetNotFoundError("Trenutno ne postoji fajl iz kog se mogu obraditi podaci, potrebno je ubaciti "
        "fajl u data/ takav da sadrži 4 kolone: reviewerID, asin, overall i "
        "unixReviewTime, gde su tipovi podataka respektivno string, string, float i "
        "ceo broj (Unix vreme u sekundama). Srećno sa daljim radom!\n")
    else:
        raise FolderNotFoundError("Morate napraviti data/ folder u root-u projekta" \
        " i u njega ubaciti fajl za obradu.")