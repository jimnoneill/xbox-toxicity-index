"""
Data collection and loading — senator handles, tweet datasets, local archives.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

from .config import (
    CONGRESS_LEGISLATORS_CURRENT_URL,
    CONGRESS_LEGISLATORS_URL,
    DATA_DIR,
    SENATOR_TWEETS_DATASET,
    XBOX_DATA,
)

log = logging.getLogger(__name__)


# ── Senator handle collection ──────────────────────────────────────

def fetch_senator_handles(cache: bool = True) -> pd.DataFrame:
    """
    Fetch current US senator Twitter/X handles from the canonical
    unitedstates/congress-legislators repo.

    Returns DataFrame with columns:
        bioguide_id, first_name, last_name, party, state, twitter_handle, twitter_id
    """
    cache_path = DATA_DIR / "senator_handles.parquet"
    if cache and cache_path.exists():
        log.info("Loading cached senator handles from %s", cache_path)
        return pd.read_parquet(cache_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Fetching legislator social media data...")
    social_resp = requests.get(CONGRESS_LEGISLATORS_URL, timeout=30)
    social_resp.raise_for_status()
    social_data = yaml.safe_load(social_resp.text)

    log.info("Fetching current legislator data...")
    current_resp = requests.get(CONGRESS_LEGISLATORS_CURRENT_URL, timeout=30)
    current_resp.raise_for_status()
    current_data = yaml.safe_load(current_resp.text)

    # Build lookup: bioguide_id -> legislator info
    legislator_info = {}
    for leg in current_data:
        bio_id = leg["id"]["bioguide"]
        name = leg["name"]
        # Get most recent term
        terms = leg.get("terms", [])
        if not terms:
            continue
        latest_term = terms[-1]
        if latest_term.get("type") != "sen":
            continue
        legislator_info[bio_id] = {
            "bioguide_id": bio_id,
            "first_name": name.get("first", ""),
            "last_name": name.get("last", ""),
            "party": latest_term.get("party", ""),
            "state": latest_term.get("state", ""),
        }

    # Merge with social media handles
    records = []
    for entry in social_data:
        bio_id = entry["id"]["bioguide"]
        if bio_id not in legislator_info:
            continue
        social = entry.get("social", {})
        twitter = social.get("twitter") or social.get("twitter_id")
        if not twitter:
            continue
        rec = legislator_info[bio_id].copy()
        rec["twitter_handle"] = social.get("twitter", "")
        rec["twitter_id"] = social.get("twitter_id", "")
        records.append(rec)

    df = pd.DataFrame(records)
    # Ensure twitter_id is string (mixed int/str causes parquet errors)
    if "twitter_id" in df.columns:
        df["twitter_id"] = df["twitter_id"].astype(str)
    log.info("Found %d senators with Twitter handles", len(df))

    if cache:
        df.to_parquet(cache_path, index=False)
        log.info("Cached to %s", cache_path)

    return df


# ── HuggingFace dataset loading ────────────────────────────────────

def load_hf_senator_tweets(split: str = "train") -> pd.DataFrame:
    """
    Load the m-newhauser/senator-tweets dataset from HuggingFace.
    ~99,693 tweets from US Senators (2021).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets`: pip install datasets")

    log.info("Loading HuggingFace dataset: %s (split=%s)", SENATOR_TWEETS_DATASET, split)
    ds = load_dataset(SENATOR_TWEETS_DATASET, split=split)
    df = ds.to_pandas()
    log.info("Loaded %d tweets from HuggingFace", len(df))
    return df


# ── Local archive loading ──────────────────────────────────────────

def load_local_archive(
    path: Optional[str] = None,
    senator_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a local tweet archive (xlsx, csv, or json).
    Default: BasedMikeLee_full_archive.xlsx from the x_box directory.
    """
    if path is None:
        path = str(XBOX_DATA / "BasedMikeLee_full_archive.xlsx")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Archive not found: {path}")

    log.info("Loading local archive: %s", path)

    if p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    elif p.suffix == ".csv":
        df = pd.read_csv(path)
    elif p.suffix == ".json":
        df = pd.read_json(path)
    elif p.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif p.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

    log.info("Loaded %d rows from %s", len(df), p.name)

    # Normalize column names
    df = _normalize_columns(df)

    if senator_name:
        df["senator_name"] = senator_name

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column name variants to canonical names."""
    col_map = {
        "id": "tweet_id",
        "tweet_id": "tweet_id",
        "created_at": "created_at",
        "full_text": "text",
        "text": "text",
        "content": "text",
        "like_count": "like_count",
        "favorite_count": "like_count",
        "retweet_count": "retweet_count",
        "reply_count": "reply_count",
        "quote_count": "quote_count",
        "in_reply_to_user_id": "in_reply_to_user_id",
        "type": "tweet_type",
        "username": "username",
        "author_id": "author_id",
        "referenced_tweet_ids": "referenced_tweet_ids",
    }

    rename = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in col_map:
            rename[col] = col_map[lower]

    if rename:
        df = df.rename(columns=rename)

    # Ensure created_at is datetime
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    # Ensure text is string
    if "text" in df.columns:
        df["text"] = df["text"].astype(str).fillna("")

    return df


# ── Combined loader ────────────────────────────────────────────────

def load_all_data(
    include_hf: bool = True,
    local_paths: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load and combine all available tweet data sources.
    """
    frames = []

    if include_hf:
        try:
            hf_df = load_hf_senator_tweets()
            hf_df["source"] = "huggingface"
            frames.append(hf_df)
        except Exception as e:
            log.warning("Could not load HuggingFace dataset: %s", e)

    if local_paths:
        for lp in local_paths:
            try:
                local_df = load_local_archive(lp)
                local_df["source"] = "local"
                frames.append(local_df)
            except Exception as e:
                log.warning("Could not load %s: %s", lp, e)

    # Always try the default Mike Lee archive
    try:
        ml_df = load_local_archive(senator_name="Mike Lee")
        ml_df["source"] = "local"
        frames.append(ml_df)
    except Exception as e:
        log.debug("Mike Lee archive not found: %s", e)

    if not frames:
        raise RuntimeError("No data sources loaded successfully")

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined dataset: %d total rows", len(combined))
    return combined
