"""T3 table 6: events.csv. Curated list; dates then machine-checked against the cited Wikipedia page text
(full date string, or "Month D" on year-in-review pages). date_on_page=1 means the date string was found on the
cited page; 0 means not found (needs manual check). Run on a host with internet."""
import os, re, json, time, urllib.request, urllib.parse, datetime as dt
import pandas as pd
W = "https://en.wikipedia.org/wiki/"
E = [
("2017-06-14","Congressional baseball shooting (Scalise wounded)","violence","2017_Congressional_baseball_shooting"),
("2017-06-22","Senate GOP health bill text released (BCRA)","legislation","Better_Care_Reconciliation_Act_of_2017"),
("2017-07-28","Senate skinny repeal vote fails","legislation","Health_Care_Freedom_Act"),
("2017-11-16","House passes Tax Cuts and Jobs Act","legislation","Tax_Cuts_and_Jobs_Act"),
("2018-07-16","Helsinki Trump-Putin summit","foreign policy","2018_Russia–United_States_summit"),
("2018-11-30","George H. W. Bush dies","crisis","George_H._W._Bush"),
("2021-11-19","House passes Build Back Better Act","legislation","Build_Back_Better_Act"),
("2017-08-12","Charlottesville Unite the Right rally","violence","Unite_the_Right_rally"),
("2017-08-25","Hurricane Harvey landfall in Texas","disaster","Hurricane_Harvey"),
("2017-09-20","Hurricane Maria hits Puerto Rico","disaster","Hurricane_Maria"),
("2017-10-01","Las Vegas shooting","violence","2017_Las_Vegas_shooting"),
("2017-10-30","Manafort and Gates indicted by Mueller","scandal/impeachment","Paul_Manafort"),
("2017-12-22","Tax Cuts and Jobs Act signed","legislation","Tax_Cuts_and_Jobs_Act"),
("2018-01-30","State of the Union address","legislation","2018_State_of_the_Union_Address"),
("2018-02-14","Parkland school shooting","violence","Stoneman_Douglas_High_School_shooting"),
("2018-05-08","US withdraws from Iran nuclear deal","foreign policy","Joint_Comprehensive_Plan_of_Action"),
("2018-06-12","Trump-Kim Singapore summit","foreign policy","2018_North_Korea–United_States_summit"),
("2018-06-27","Justice Kennedy announces retirement","court ruling","Brett_Kavanaugh_Supreme_Court_nomination"),
("2018-09-27","Kavanaugh and Ford Judiciary hearing","scandal/impeachment","Brett_Kavanaugh_Supreme_Court_nomination"),
("2018-10-06","Kavanaugh confirmed to Supreme Court","court ruling","Brett_Kavanaugh_Supreme_Court_nomination"),
("2018-10-27","Pittsburgh synagogue shooting","violence","Pittsburgh_synagogue_shooting"),
("2018-11-06","2018 midterm elections","election","2018_United_States_elections"),
("2018-12-22","Partial government shutdown begins","crisis","2018–2019_United_States_federal_government_shutdown"),
("2019-01-25","Shutdown ends after 35 days","crisis","2018–2019_United_States_federal_government_shutdown"),
("2019-02-15","Border national emergency declared","crisis","National_Emergency_Concerning_the_Southern_Border_of_the_United_States"),
("2019-04-18","Mueller report released","scandal/impeachment","Mueller_report"),
("2019-08-03","El Paso shooting","violence","2019_El_Paso_shooting"),
("2019-09-24","Pelosi announces impeachment inquiry","scandal/impeachment","First_impeachment_of_Donald_Trump"),
("2019-12-18","House impeaches Trump (first)","scandal/impeachment","First_impeachment_of_Donald_Trump"),
("2020-01-03","US kills Soleimani","foreign policy","Assassination_of_Qasem_Soleimani"),
("2020-02-05","Senate acquits Trump (first impeachment)","scandal/impeachment","First_impeachment_of_Donald_Trump"),
("2020-03-13","COVID-19 national emergency declared","crisis","COVID-19_pandemic_in_the_United_States"),
("2020-03-27","CARES Act signed","legislation","CARES_Act"),
("2020-05-25","George Floyd killed","violence","Murder_of_George_Floyd"),
("2020-09-18","Ginsburg dies","court ruling","Ruth_Bader_Ginsburg"),
("2020-10-02","Trump COVID-19 diagnosis","crisis","Health_of_Donald_Trump"),
("2020-10-26","Barrett confirmed to Supreme Court","court ruling","Amy_Coney_Barrett_Supreme_Court_nomination"),
("2020-11-03","2020 general election","election","2020_United_States_presidential_election"),
("2020-11-07","Networks call race for Biden","election","2020_United_States_presidential_election"),
("2020-12-27","COVID relief and omnibus signed","legislation","Consolidated_Appropriations_Act,_2021"),
("2021-01-05","Georgia Senate runoffs","election","2020–21_United_States_Senate_special_election_in_Georgia"),
("2021-01-06","January 6 Capitol attack","violence","January_6_United_States_Capitol_attack"),
("2021-01-13","House impeaches Trump (second)","scandal/impeachment","Second_impeachment_of_Donald_Trump"),
("2021-01-20","Biden inauguration","election","Inauguration_of_Joe_Biden"),
("2021-02-13","Senate acquits Trump (second impeachment)","scandal/impeachment","Second_impeachment_of_Donald_Trump"),
("2021-03-11","American Rescue Plan signed","legislation","American_Rescue_Plan_Act_of_2021"),
("2021-08-15","Kabul falls to Taliban","foreign policy","Fall_of_Kabul_(2021)"),
("2021-08-26","Kabul airport bombing","violence","2021_Kabul_airport_attack"),
("2021-08-30","Afghanistan withdrawal completed","foreign policy","Withdrawal_of_United_States_troops_from_Afghanistan_(2020–2021)"),
("2021-11-15","Infrastructure law signed","legislation","Infrastructure_Investment_and_Jobs_Act"),
("2022-02-24","Russia invades Ukraine","foreign policy","Russian_invasion_of_Ukraine"),
("2022-03-01","State of the Union address","legislation","2022_State_of_the_Union_Address"),
("2022-05-02","Dobbs draft opinion leaked","court ruling","Dobbs_v._Jackson_Women's_Health_Organization"),
("2022-05-24","Uvalde school shooting","violence","Robb_Elementary_School_shooting"),
("2022-06-09","January 6 Committee public hearings begin","scandal/impeachment","United_States_House_Select_Committee_on_the_January_6_Attack"),
("2022-06-24","Dobbs decision overturns Roe","court ruling","Dobbs_v._Jackson_Women's_Health_Organization"),
("2022-06-25","Bipartisan Safer Communities Act signed","legislation","Bipartisan_Safer_Communities_Act"),
("2022-08-08","FBI searches Mar-a-Lago","scandal/impeachment","Federal_Bureau_of_Investigation_search_of_Mar-a-Lago"),
("2022-08-16","Inflation Reduction Act signed","legislation","Inflation_Reduction_Act"),
("2022-11-08","2022 midterm elections","election","2022_United_States_elections"),
("2023-01-07","McCarthy elected Speaker on 15th ballot","legislation","Kevin_McCarthy"),
("2023-03-30","Trump indicted in New York","scandal/impeachment","Prosecution_of_Donald_Trump_in_New_York"),
("2023-06-03","Fiscal Responsibility Act signed","legislation","Fiscal_Responsibility_Act_of_2023"),
("2023-08-01","Trump indicted over 2020 election","scandal/impeachment","Federal_prosecution_of_Donald_Trump_(election_obstruction_case)"),
("2023-09-12","House Biden impeachment inquiry announced","scandal/impeachment","Impeachment_inquiry_into_Joe_Biden"),
("2023-10-03","McCarthy removed as Speaker","legislation","Removal_of_Kevin_McCarthy_as_Speaker_of_the_United_States_House_of_Representatives"),
("2023-10-07","Hamas attack on Israel","foreign policy","October_7_attacks"),
("2024-04-24","Ukraine Israel Taiwan aid package signed","legislation","2024_United_States_federal_budget"),
("2024-05-30","Trump convicted in New York","court ruling","Trial_of_Donald_Trump_in_New_York"),
("2024-06-27","Biden-Trump debate","election","2024_United_States_presidential_debates"),
("2024-07-01","Trump v. United States immunity ruling","court ruling","Trump_v._United_States"),
("2024-07-13","Trump assassination attempt in Butler","violence","Attempted_assassination_of_Donald_Trump_in_Pennsylvania"),
("2024-07-21","Biden withdraws from race","election","2024_United_States_presidential_election"),
("2024-09-10","Harris-Trump debate","election","2024_United_States_presidential_debates"),
("2024-11-05","2024 general election","election","2024_United_States_elections"),
("2025-01-20","Trump second inauguration, DOGE created","election","Second_inauguration_of_Donald_Trump"),
("2025-02-28","Zelensky Oval Office confrontation","foreign policy","2025_Trump–Zelenskyy_Oval_Office_meeting"),
("2025-03-24","Signal chat leak reported","scandal/impeachment","Signalgate"),
("2025-04-02","Liberation Day tariffs announced","foreign policy","Liberation_Day_tariffs"),
("2025-05-22","House passes One Big Beautiful Bill","legislation","One_Big_Beautiful_Bill_Act"),
("2025-06-14","Minnesota lawmakers shot","violence","Shootings_of_Minnesota_legislators"),
("2025-06-22","US strikes Iranian nuclear sites","foreign policy","United_States_strikes_on_Iranian_nuclear_sites"),
("2025-07-04","One Big Beautiful Bill signed","legislation","One_Big_Beautiful_Bill_Act"),
("2025-07-04","Central Texas floods","disaster","2025_Central_Texas_floods"),
("2025-09-10","Charlie Kirk assassinated","violence","Assassination_of_Charlie_Kirk"),
("2025-10-01","Government shutdown begins","crisis","2025_United_States_federal_government_shutdown"),
("2025-11-04","2025 off-year elections","election","2025_United_States_elections"),
("2025-11-12","Shutdown ends after 43 days","crisis","2025_United_States_federal_government_shutdown"),
("2025-11-19","Epstein Files Transparency Act signed","legislation","Epstein_Files_Transparency_Act"),
("2026-01-03","US captures Maduro","foreign policy","2026_in_the_United_States"),
("2026-01-07","Renee Good shot by ICE officer in Minneapolis","violence","2026_in_the_United_States"),
("2026-01-24","Alex Pretti shot in Minneapolis","violence","2026_in_the_United_States"),
("2026-01-31","Brief government shutdown begins","crisis","2026_in_the_United_States"),
("2026-02-14","DHS-only shutdown begins","crisis","2026_in_the_United_States"),
("2026-02-20","Supreme Court strikes down IEEPA tariffs","court ruling","2026_in_the_United_States"),
("2026-02-24","State of the Union address","legislation","2026_in_the_United_States"),
("2026-02-28","US and Israel strike Iran, war begins","foreign policy","2026_in_the_United_States"),
("2026-04-29","Louisiana v. Callais redistricting ruling","court ruling","2026_in_the_United_States"),
("2026-04-30","DHS shutdown ends after 76 days","crisis","2026_in_the_United_States"),
("2026-06-30","Supreme Court rules on birthright citizenship","court ruling","2026_in_the_United_States"),
]
MON = ["January","February","March","April","May","June","July","August","September","October","November","December"]
def page(title):
    t = urllib.parse.quote(title.replace("_", " "))
    u = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles=" + t
    for k in range(6):
        try:
            time.sleep(3)
            j = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "xbox-t3/1.0 (research; contact via repo)"}), timeout=60))
            return list(j["query"]["pages"].values())[0].get("extract", "")
        except Exception:
            time.sleep(10 * (k + 1))
    return ""
cache = {}; rows = []
for d, lab, cat, title in E:
    x = dt.date.fromisoformat(d)
    if title not in cache: cache[title] = page(title)
    txt = cache[title]
    a = f"{MON[x.month-1]} {x.day}, {x.year}"; b = f"{x.day} {MON[x.month-1]} {x.year}"; c = f"{MON[x.month-1]} {x.day}"
    yearpage = re.fullmatch(r"\d{4}_in_the_United_States", title) is not None
    full = (a in txt) or (b in txt)
    md = bool(re.search(re.escape(c) + r"(?!\d)", txt))
    ok = full or md
    lvl = "full_date" if full else ("month_day_only" if md else "not_found")
    rows.append(dict(date=d, label=lab, category=cat, source_url=W + urllib.parse.quote(title, safe="/_(),:'"),
                     date_on_page=int(ok), date_match=lvl, page_found=int(len(txt) > 0)))
df = pd.DataFrame(rows).sort_values("date")
# manual web verification where no Wikipedia page carries the date (source read directly: Axios 2024-04-24 'Biden signs $95 billion foreign aid package')
m = df.label.eq("Ukraine Israel Taiwan aid package signed")
df.loc[m, "source_url"] = "https://www.axios.com/2024/04/24/biden-signs-foreign-aid-package-ukraine-israel"
df.loc[m, ["date_on_page", "date_match"]] = [1, "manual_web_verified"]
out = os.path.expanduser("~/x_box/pipeline/output/v7/t3/events.csv")
df.to_csv(out, index=False)
print(len(df), df.date_on_page.sum()); print(df[df.date_on_page == 0][["date", "label", "page_found"]].to_string())
