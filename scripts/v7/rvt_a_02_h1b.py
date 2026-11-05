"""RVT-A H1b hysteresis. Follows output/v7/rvt_a/DESIGN.md."""
import sys, json
sys.path.insert(0, "/home/joneill/x_box/pipeline/scripts/v7")
from rvt_a_common import *

panel = load_panel()
v, first_start = load_service(panel)
nomtab = {i: g.dropna(subset=["nominate_dim1"]).set_index("congress").nominate_dim1.abs() for i, g in v.groupby("icpsr")}


def nom(i, T):
    s = nomtab.get(i)
    if s is None or len(s) == 0:
        return np.nan
    c = (T - 1 - 1789) // 2 + 1
    return float(s.iloc[(np.abs(s.index.values - c)).argmin()])


def win(years):
    p = panel[panel.Year.isin(years)]
    g = p.groupby("MemberICPSR").agg(share=("share", "mean"), logn=("n", lambda x: np.log(x).mean()), party=("party", "first"),
                                     senate=("chamber", lambda x: float((x == "Senate").iloc[0])))
    return g


def build(yA, yB, vet_max, newA, newB):
    A, Bw = win(yA), win(yB)
    fs = first_start
    vets = [i for i in A.index if i in Bw.index and fs.get(i, 9999) <= vet_max]
    nA = [i for i in A.index if fs.get(i, 0) in newA]
    nB = [i for i in Bw.index if fs.get(i, 0) in newB]
    rows = []
    tA, tB = max(yA), max(yB)
    for i in vets:
        for w, W, T in ((0, A, tA), (1, Bw, tB)):
            r = W.loc[i]
            rows.append(dict(id=i, vet=1.0, B=float(w), y=r.share, logn=r.logn, senate=r.senate, party=r.party, nom=nom(i, T)))
    for i in nA:
        if i in vets:
            continue
        r = A.loc[i]; rows.append(dict(id=i, vet=0.0, B=0.0, y=r.share, logn=r.logn, senate=r.senate, party=r.party, nom=nom(i, tA)))
    for i in nB:
        r = Bw.loc[i]; rows.append(dict(id=i, vet=0.0, B=1.0, y=r.share, logn=r.logn, senate=r.senate, party=r.party, nom=nom(i, tB)))
    d = pd.DataFrame(rows).dropna(subset=["nom"])
    return d


def est(d, pooled, wild=True):
    d = d.copy()
    d["vB"] = d.vet * d.B
    cols = [np.ones(len(d)), d.vet, d.B, d.vB, d.nom, d.senate, d.logn]
    j = 3
    if pooled:
        R = (d.party == "R").astype(float)
        cols += [R, d.vet * R, d.B * R]
    X = np.column_stack(cols)
    y = d.y.values; cl = codes(d.id)
    if wild:
        r = wild_boot_t(X, y, cl, j)
    else:
        b, se, e, V = ols_cluster(X, y, cl); r = dict(b=float(b[j]), se=float(se[j]), ci_cl=[float(b[j] - 1.96 * se[j]), float(b[j] + 1.96 * se[j])])
    r["n_rows"] = len(d)
    r["n_vet"] = int(d[(d.vet == 1) & (d.B == 0)].id.nunique())
    r["n_newA"] = int(((d.vet == 0) & (d.B == 0)).sum()); r["n_newB"] = int(((d.vet == 0) & (d.B == 1)).sum())
    return r


def desc(d):
    o = {}
    vv = d[d.vet == 1].pivot(index="id", columns="B", values="y")
    o["vet_change_mean"] = float((vv[1.0] - vv[0.0]).mean()); o["vet_change_se"] = float((vv[1.0] - vv[0.0]).std() / np.sqrt(len(vv)))
    for w in (0, 1):
        g = d[d.B == w]
        o[f"gap_vet_minus_new_win{w}"] = float(g[g.vet == 1].y.mean() - g[g.vet == 0].y.mean())
        o[f"mean_vet_win{w}"] = float(g[g.vet == 1].y.mean()); o[f"mean_new_win{w}"] = float(g[g.vet == 0].y.mean())
    return o


out = {}
main = build([2015, 2016], [2021, 2022], 2011, {2013, 2015}, {2019, 2021})
out["primary"] = {"pooled": est(main, True), "D": est(main[main.party == "D"], False), "R": est(main[main.party == "R"], False)}
out["desc"] = {p: desc(main[main.party == p]) for p in ("D", "R")}
# level gap in B adjusted (within window B only)
gap = {}
for p in ("D", "R", "both"):
    d = main[main.B == 1] if p == "both" else main[(main.B == 1) & (main.party == p)]
    cols = [np.ones(len(d)), d.vet, d.nom, d.senate, d.logn] + ([(d.party == "R").astype(float)] if p == "both" else [])
    b, se, e, V = ols_cluster(np.column_stack(cols), d.y.values, codes(d.id)); gap[p] = dict(b=float(b[1]), se=float(se[1]), n=len(d))
out["adj_gap_B"] = gap
# variant: newcomers B = 2021 starters only
v2 = build([2015, 2016], [2021, 2022], 2011, {2013, 2015}, {2021})
out["new2021only"] = {"pooled": est(v2, True, wild=False), "D": est(v2[v2.party == "D"], False, wild=False), "R": est(v2[v2.party == "R"], False, wild=False)}
v3 = build([2015, 2016], [2021, 2022], 2015, {2013, 2015}, {2019, 2021})
out["vets_start_le_2015"] = {"pooled": est(v3, True, wild=False)}
# leave-one-year-out
loyo = {}
for yA in (2015, 2016):
    for yB in (2021, 2022):
        d = build([yA], [yB], 2011, {2013, 2015}, {2019, 2021})
        loyo[f"{yA}|{yB}"] = {"pooled": est(d, True, wild=False), "D": est(d[d.party == "D"], False, wild=False), "R": est(d[d.party == "R"], False, wild=False)}
out["leave_one_year_out"] = loyo
# placebos
pl = {}
for name, (wA, wB, vm, nA, nB) in {"P1_2011-12_vs_2013-14": ([2011, 2012], [2013, 2014], 2009, {2011}, {2013}),
                                   "P2_2017-18_vs_2019-20": ([2017, 2018], [2019, 2020], 2015, {2017}, {2019})}.items():
    d = build(wA, wB, vm, nA, nB)
    pl[name] = {"pooled": est(d, True, wild=False), "D": est(d[d.party == "D"], False, wild=False), "R": est(d[d.party == "R"], False, wild=False)}
out["placebos"] = pl
json.dump(out, open(f"{OUT}/h1b_results.json", "w"), indent=1, default=float)
main.to_csv(f"{OUT}/h1b_data.csv", index=False)


def f(x): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}
for k in ("primary",):
    for g, x in out[k].items(): print(k, g, f(x))
print(out["desc"]); print(out["adj_gap_B"])
print("new2021", {g: f(x) for g, x in out["new2021only"].items()}); print("vets<=2015", f(out["vets_start_le_2015"]["pooled"]))
for k, x in loyo.items(): print(k, {g: (round(y["b"], 4), round(y["se"], 4)) for g, y in x.items()})
for k, x in pl.items(): print(k, {g: (round(y["b"], 4), round(y["se"], 4), y["n_rows"]) for g, y in x.items()})
