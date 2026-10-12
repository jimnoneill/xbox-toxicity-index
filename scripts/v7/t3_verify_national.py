"""T3: check the hard-coded dates in t3_national.py against Wikipedia page text. Output national_checks.csv."""
import os, re, json, time, urllib.request, urllib.parse, datetime as dt
import pandas as pd
MON = ["January","February","March","April","May","June","July","August","September","October","November","December"]
C = [
("sotu","2018-01-30","2018_State_of_the_Union_Address"),("sotu","2019-02-05","2019_State_of_the_Union_Address"),
("sotu","2020-02-04","2020_State_of_the_Union_Address"),("sotu","2021-04-28","2021_Joe_Biden_speech_to_a_joint_session_of_Congress"),
("sotu","2022-03-01","2022_State_of_the_Union_Address"),("sotu","2023-02-07","2023_State_of_the_Union_Address"),
("sotu","2024-03-07","2024_State_of_the_Union_Address"),("sotu","2025-03-04","2025_Donald_Trump_speech_to_a_joint_session_of_Congress"),
("sotu","2026-02-24","2026_in_the_United_States"),
("shutdown_start","2018-01-20","January_2018_United_States_federal_government_shutdown"),
("shutdown_end","2018-01-22","January_2018_United_States_federal_government_shutdown"),
("shutdown_start","2018-02-09","2018_United_States_federal_government_shutdowns"),
("shutdown_start","2018-12-22","2018–2019_United_States_federal_government_shutdown"),
("shutdown_end","2019-01-25","2018–2019_United_States_federal_government_shutdown"),
("shutdown_start","2025-10-01","2025_United_States_federal_government_shutdown"),
("shutdown_end","2025-11-12","2025_United_States_federal_government_shutdown"),
("shutdown_start","2026-01-31","2026_in_the_United_States"),("shutdown_end","2026-02-03","2026_in_the_United_States"),
("shutdown_start","2026-02-14","2026_in_the_United_States"),("shutdown_end","2026-04-30","2026_in_the_United_States"),
("inauguration","2021-01-20","Inauguration_of_Joe_Biden"),("inauguration","2025-01-20","Second_inauguration_of_Donald_Trump"),
("senate_majority_change","2021-01-20","117th_United_States_Congress"),("house_majority_change","2019-01-03","116th_United_States_Congress"),
("house_majority_change","2023-01-03","118th_United_States_Congress"),("senate_majority_change","2025-01-03","119th_United_States_Congress"),
]
def page(t):
    u = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=" + urllib.parse.quote(t.replace("_", " "))
    for k in range(6):
        try:
            time.sleep(3)
            return list(json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "xbox-t3/1.0"}), timeout=60))["query"]["pages"].values())[0].get("extract", "")
        except Exception:
            time.sleep(15 * (k + 1))
    return ""
rows = []; cache = {}
for k, d, t in C:
    if t not in cache: cache[t] = page(t)
    x = dt.date.fromisoformat(d); txt = cache[t]
    full = f"{MON[x.month-1]} {x.day}, {x.year}" in txt or f"{x.day} {MON[x.month-1]} {x.year}" in txt
    md = re.search(f"{MON[x.month-1]} {x.day}\\b", txt) is not None
    rows.append(dict(item=k, date=d, source_url="https://en.wikipedia.org/wiki/" + urllib.parse.quote(t, safe="/_(),:'"),
                     date_match="full_date" if full else ("month_day_only" if md else "not_found"), page_len=len(txt)))
df = pd.DataFrame(rows)
df.to_csv(os.path.expanduser("~/x_box/pipeline/output/v7/t3/national_checks.csv"), index=False)
print(df[df.date_match != "full_date"].to_string())
