"""T3 step 0: fetch FEC election-dates (official FEC API, public DEMO_KEY) for 2018-2026.
Retries with backoff on rate limit. Output: output/v7/t3/src/fec_dates_raw.json (all rows)."""
import json, time, urllib.request, urllib.error, os, sys
OUT = os.path.expanduser("~/x_box/pipeline/output/v7/t3/src")
rows = {}
for y in (2018, 2020, 2022, 2024, 2026):
    p = 1
    while True:
        cf = OUT + ("/fec_%d.json" % y if p == 1 else "/fec_%d_p%d.json" % (y, p))
        if os.path.exists(cf):
            j = json.load(open(cf))
            if "results" in j:
                rows[(y, p)] = j["results"]
                if p >= j["pagination"]["pages"]: break
                p += 1; continue
        url = ("https://api.open.fec.gov/v1/election-dates/?api_key=DEMO_KEY&election_year=%d"
               "&per_page=100&page=%d&sort=election_date" % (y, p))
        try:
            j = json.load(urllib.request.urlopen(url, timeout=60))
        except urllib.error.HTTPError as e:
            print("http", e.code, y, p, "sleep", flush=True); time.sleep(600); continue
        except Exception as e:
            print("err", e, flush=True); time.sleep(60); continue
        rows[(y, p)] = j["results"]
        json.dump(j, open(cf, "w"))
        print(y, p, j["pagination"]["pages"], flush=True)
        if p >= j["pagination"]["pages"]: break
        p += 1; time.sleep(2)
json.dump([r for v in rows.values() for r in v], open(OUT + "/fec_dates_raw.json", "w"))
print("done")
