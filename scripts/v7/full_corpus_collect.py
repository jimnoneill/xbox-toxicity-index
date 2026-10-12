"""Collect every daily file of alexlitel/congresstweets (resumable), build congresstweets_full.parquet.
Raw gz JSON in data/congresstweets_full/raw/. Expected days come from the repo git tree listing."""
import gzip, json, os, sys, time, random, glob
import requests
import pandas as pd, numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.expanduser("~/x_box/pipeline")
D = f"{ROOT}/data/congresstweets_full"; RAW = f"{D}/raw"
os.makedirs(RAW, exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "congresstweets-research-fetch"

def expected_days():
    p = f"{D}/expected_days.json"
    if os.path.exists(p):
        return json.load(open(p))
    r = S.get("https://api.github.com/repos/alexlitel/congresstweets/git/trees/master?recursive=1", timeout=60)
    r.raise_for_status()
    j = r.json(); print("tree truncated:", j.get("truncated"), flush=True)
    days = sorted(x["path"][5:-5] for x in j["tree"] if x["path"].startswith("data/") and x["path"].endswith(".json")
                  and len(x["path"]) == 20)
    json.dump(days, open(p, "w")); return days

def fetch(day):
    out = f"{RAW}/{day}.json.gz"
    if os.path.exists(out):
        return day, "have"
    url = f"https://raw.githubusercontent.com/alexlitel/congresstweets/master/data/{day}.json"
    for a in range(7):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                json.loads(r.content)  # validate
                with gzip.open(out + ".tmp", "wb") as f: f.write(r.content)
                os.replace(out + ".tmp", out); time.sleep(0.15); return day, "ok"
            if r.status_code == 404: return day, "404"
        except Exception as e:
            pass
        time.sleep(min(60, 2 ** a + random.random()))
    return day, "fail"

if __name__ == "__main__":
    days = expected_days()
    print("expected days", len(days), days[0], days[-1], flush=True)
    res = {}
    with ThreadPoolExecutor(4) as ex:
        for i, (d, s) in enumerate(ex.map(fetch, days)):
            res[d] = s
            if i % 100 == 0: print(i, d, s, flush=True)
    bad = {d: s for d, s in res.items() if s not in ("ok", "have")}
    print("not fetched:", bad, flush=True)
    # build parquet in chunks by year
    cols = ["id", "screen_name", "time", "link", "text", "source", "user_id", "_date", "created_at", "year"]
    writer = None; tot = 0
    out = f"{D}/congresstweets_full.parquet"
    files = sorted(glob.glob(f"{RAW}/*.json.gz"))
    buf = []
    def flush():
        global writer, tot, buf
        if not buf: return
        df = pd.DataFrame(buf); buf = []
        for c in ["id", "screen_name", "time", "link", "text", "source", "user_id", "_date"]:
            if c not in df: df[c] = None
            df[c] = df[c].where(df[c].isna(), df[c].astype(str))
        df["created_at"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["created_at"])
        df["year"] = df["created_at"].dt.year.astype("int32")
        t = pa.Table.from_pandas(df[cols], preserve_index=False)
        if writer is None: writer = pq.ParquetWriter(out + ".tmp", t.schema)
        else: t = t.cast(writer.schema)
        writer.write_table(t); tot += len(df)
    for f in files:
        day = os.path.basename(f)[:10]
        for t in json.loads(gzip.open(f).read()):
            t["_date"] = day; buf.append({k: t.get(k) for k in cols if k not in ("created_at", "year")})
        if len(buf) > 300000: flush()
    flush(); writer.close(); os.replace(out + ".tmp", out)
    print("WROTE", tot, flush=True)
