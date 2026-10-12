"""Shared loaders for the FULL-corpus T1/T2 reruns (new file). Reuses t1_common (describe, build_locations, load_amap).
Differences from t1_common: raw = full daily archive (5.07M posts, de-duplicated on id); outputs go to
output/v7/t1full and output/v7/t2full; the outage mask is read from t1full."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_common import *  # noqa: E402,F401,F403
import t1_common as _c  # noqa: E402

SEED = 20261010
T1 = ROOT / "output/v7/t1full"
T2 = ROOT / "output/v7/t2full"
T1OLD = ROOT / "output/v7/t1"
T2OLD = ROOT / "output/v7/t2"


def load_raw():
    """All posts of the full archive, de-duplicated on id. Columns: id, screen_name, t (UTC epoch s), text, source."""
    tw = pd.read_parquet(ROOT / "data/congresstweets_full/congresstweets_full.parquet",
                         columns=["id", "screen_name", "created_at", "text", "source"])
    tw = tw.drop_duplicates("id")
    tw["t"] = (tw["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).values.astype("datetime64[s]")
               .astype("int64"))
    return tw.drop(columns=["created_at"])


def load_mask():
    m = pd.read_parquet(T1 / "outage_mask.parquet")
    hour0 = int(m.hour.iloc[0])
    obs = m.observed.values
    cum = np.concatenate([[0], np.cumsum(obs)]).astype("int64")
    return m, hour0, obs, cum
