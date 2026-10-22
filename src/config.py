# Paths and basic settings.
from __future__ import annotations

import os
from pathlib import Path

# Project root is one level above src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATTERNS = ("*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz")

# Experiment outputs: figures and tables are written here.
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
TABLES_DIR = DOCS_DIR / "tables"

REQUIRED_COLUMNS = ("reviewerID", "asin", "overall", "unixReviewTime")

# Default thresholds of the iterative filtering (single source for the agent and the experiments).
MIN_USER_INTERACTIONS = 4
MIN_ITEM_INTERACTIONS = 2


class DatasetNotFoundError(FileNotFoundError):
    """The data file does not exist at the expected location."""


class TooManyFilesInDataFolderError(ValueError):
    """The data folder contains more than one readable file."""


class FolderNotFoundError(FileNotFoundError):
    """The data folder does not exist."""


DATA_ERRORS = (DatasetNotFoundError, TooManyFilesInDataFolderError, FolderNotFoundError)


def resolve_raw_reviews_path() -> Path:
    """Resolve the dataset path: RAW_REVIEWS_PATH env override, otherwise the single file in data/."""
    env_path = os.environ.get("RAW_REVIEWS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    if DATA_DIR.is_dir():
        found = sorted(p for pattern in DATA_PATTERNS for p in DATA_DIR.glob(pattern))
        if len(found) == 1:
            return found[0]
        elif len(found) > 1:
            raise TooManyFilesInDataFolderError(
                "More than one readable file was found in data/. Make sure there is exactly one file "
                "of type .json, .jsonl, .json.gz or .jsonl.gz where each line is one review."
            )
        else:
            raise DatasetNotFoundError(
                "There is no file to process. Put a file into data/ that contains the 4 fields "
                "reviewerID, asin, overall and unixReviewTime (string, string, float and integer "
                "Unix time in seconds, respectively).\n"
            )
    else:
        raise FolderNotFoundError(
            "Create a data/ folder in the project root and put the file to process into it."
        )
