"""RVT-A H2 heterogeneous transition effects. Follows output/v7/rvt_a/DESIGN.md."""
import sys, json
sys.path.insert(0, "/home/joneill/x_box/pipeline/scripts/v7")
from rvt_a_common import *

panel = load_panel()
share = panel.set_index(["MemberICPSR", "Year"]).share
EV = {2011: ("D", "House"), 2015: ("D", "Senate"), 2017: ("D", None), 2019: ("R", "House"), 2021: ("R", None)}


def d_table(T, kind="d1"):
    cur = panel[panel.Year == T][["MemberICPSR", "party", "chamber"]]
    rows = []
    for r in cur.itertuples():
        i = r.MemberICPSR
        if kind == "d1":
            if (i, T - 1) not in share.index: continue
            y = share[(i, T)] - share[(i, T - 1)]
        else:
            pre = [share[(i, t)] for t in (T - 2, T - 1) if (i, t) in share.index]
            post = [share[(i, t)] for t in (T, T + 1) if (i, t) in share.index]
            if not pre or not post: continue
            y = np.mean(post) - np.mean(pre)
        rows.append(dict(id=i, party=r.party, chamber=r.chamber, y=y))
    return pd.DataFrame(rows)


def contrast(d, loser, wild=False):
    """beta = loser minus gainer mean change, chamber control if both chambers present."""
    L = (d.party == loser).astype(float).values
    cols = [np.ones(len(d)), L]
    if d.chamber.nunique() > 1:
        cols.append((d.chamber == "Senate").astype(float).values)
    X = np.column_stack(cols)
    cl = codes(d.id)
    if wild:
        r = wild_boot_t(X, d.y.values, cl, 1)
    else:
        b, se, e, V = ols_cluster(X, d.y.values, cl); r = dict(b=float(b[1]), se=float(se[1]))
    r["n"] = len(d); r["nL"] = int(L.sum()); r["nG"] = int(len(d) - L.sum())
    return r


out = {"event": {}, "event_D2": {}, "E1": {}, "E3": {}, "placebo": {}}
for T, (loser, ch) in EV.items():
    d = d_table(T)
    da = d if ch is None else d[d.chamber == ch]
    out["event"][T] = contrast(da, loser, wild=True)
    d2 = d_table(T, "d2"); d2a = d2 if ch is None else d2[d2.chamber == ch]
    out["event_D2"][T] = contrast(d2a, loser)
    ll = da[da.party == loser].y
    out["E1"][T] = dict(mean=float(ll.mean()), se=float(ll.std() / np.sqrt(len(ll))), n=len(ll))
    if ch is not None:
        dd = d[d.party == loser]
        X = np.column_stack([np.ones(len(dd)), (dd.chamber == ch).astype(float)])
        b, se, e, V = ols_cluster(X, dd.y.values, codes(dd.id)); out["E3"][T] = dict(b=float(b[1]), se=float(se[1]), n=len(dd))
    print(T, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["event"][T].items() if k in ("b", "se", "p_wild", "n", "nL", "nG")})

# placebo years
PL = [2012, 2013, 2014, 2016, 2018, 2020, 2022]
pl = {"both": {}, "House": {}, "Senate": {}}
for T in PL:
    d = d_table(T)
    pl["both"][T] = contrast(d, "D")
    pl["House"][T] = contrast(d[d.chamber == "House"], "D")
    pl["Senate"][T] = contrast(d[d.chamber == "Senate"], "D")
out["placebo"] = pl

bT = np.array([out["event"][T]["b"] for T in EV]); sT = np.array([out["event"][T]["se"] for T in EV]); Ts = np.array(list(EV))
# sign: event beta is loser minus gainer already
w = 1 / sT ** 2
meta = dl_meta(bT, sT)
out["homogeneity"] = {"betas": dict(zip(map(int, Ts), bT.tolist())), "ses": dict(zip(map(int, Ts), sT.tolist())), **meta}
num = float(((bT - bT.mean()) ** 2).sum() / 4)
for nm in ("both", "House", "Senate"):
    pb = np.array([x["b"] for x in pl[nm].values()])
    var = float(pb.var(ddof=1))
    F = num / var
    out["homogeneity"][f"F_{nm}"] = dict(F=F, placebo_var=var, placebo_mean=float(pb.mean()), p=float(1 - stats.f.cdf(F, 4, 6)))
# shock-variance chi2 alternative
pb = np.array([x["b"] for x in pl["both"].values()]); pse = np.array([x["se"] for x in pl["both"].values()])
tau2 = max(0.0, pb.var(ddof=1) - (pse ** 2).mean())
wc = 1 / (sT ** 2 + tau2); mc = (wc * bT).sum() / wc.sum(); Qc = float((wc * (bT - mc) ** 2).sum())
out["homogeneity"]["chi2_with_party_year_shock"] = dict(tau2_shock=float(tau2), Q=Qc, p=float(1 - stats.chi2.cdf(Qc, 4)))
# within party
wp = {}
for p, lst in (("D", [2011, 2015, 2017]), ("R", [2019, 2021])):
    b = np.array([out["event"][T]["b"] for T in lst]); s = np.array([out["event"][T]["se"] for T in lst]); t = np.array(lst, float)
    m = dl_meta(b, s)
    w_ = 1 / s ** 2; tb = (w_ * t).sum() / w_.sum(); sl = (w_ * (t - tb) * b).sum() / (w_ * (t - tb) ** 2).sum(); sse = 1 / np.sqrt((w_ * (t - tb) ** 2).sum())
    wp[p] = dict(Ts=lst, betas=b.tolist(), ses=s.tolist(), Q=m["Q"], df=m["df"], p_Q=m["p_Q"], I2=m["I2"], slope_per_year=float(sl), slope_se=float(sse))
out["within_party"] = wp
bD2 = np.array([out["event_D2"][T]["b"] for T in EV]); sD2 = np.array([out["event_D2"][T]["se"] for T in EV])
out["homogeneity_D2"] = {"betas": bD2.tolist(), "ses": sD2.tolist(), **dl_meta(bD2, sD2)}

# panel model
p = panel.copy()
p["R"] = (p.party == "R").astype(float)
p["outP"] = (p.party != p.Year.map(PRES)).astype(float)
own = np.where(p.chamber == "House", p.Year.map(HOUSE), p.Year.map(SENATE))
oth = np.where(p.chamber == "House", p.Year.map(SENATE), p.Year.map(HOUSE))
p["outC"] = (p.party.values != own).astype(float); p["outO"] = (p.party.values != oth).astype(float)
p["late"] = (p.Year >= 2017).astype(float)
p["tD"] = (1 - p.R) * (p.Year - 2010); p["tR"] = p.R * (p.Year - 2010)
p["outPxL"] = p.outP * p.late; p["outCxL"] = p.outC * p.late
cols = ["outP", "outC", "outO", "tD", "tR", "outPxL", "outCxL"]


def panel_fit(d, wild=False, js=(5, 6)):
    d = d.copy()
    g = d.groupby("MemberICPSR")
    X = np.column_stack([d[c] - g[c].transform("mean") for c in cols]); y = (d.share - g.share.transform("mean")).values
    cl = codes(d.MemberICPSR)
    b, se, e, V = ols_cluster(X, y, cl)
    r = {"coef": dict(zip(cols, b.tolist())), "se": dict(zip(cols, se.tolist())), "rank": int(np.linalg.matrix_rank(X)), "n": len(d), "members": int(d.MemberICPSR.nunique())}
    if wild:
        for j in js:
            r["wild_" + cols[j]] = wild_boot_t(X, y, cl, j)
    return r


out["panel"] = panel_fit(p, wild=True)
out["panel_loyo"] = {int(y): panel_fit(p[p.Year != y]) for y in sorted(p.Year.unique())}
# panel with only outP outC outO + trends (no interactions) for reference
json.dump(out, open(f"{OUT}/h2_results.json", "w"), indent=1, default=float)
print(json.dumps({k: out["homogeneity"][k] for k in out["homogeneity"] if k not in ("betas", "ses")}, indent=1, default=float))
print(json.dumps(out["within_party"], indent=1))
print("D2", out["homogeneity_D2"])
print({T: (round(x["b"], 4), round(x["se"], 4)) for T, x in out["event_D2"].items()})
print("E1", out["E1"]); print("E3", out["E3"])
for nm in pl: print(nm, {T: (round(x["b"], 4), round(x["se"], 4)) for T, x in pl[nm].items()})
print("panel", {c: (round(out["panel"]["coef"][c], 4), round(out["panel"]["se"][c], 4)) for c in cols}, out["panel"]["rank"])
for k in ("wild_outPxL", "wild_outCxL"):
    r = out["panel"][k]; print(k, round(r["b"], 4), round(r["se"], 4), r["p_wild"], r["ci_wild"])
for y, r in out["panel_loyo"].items(): print(y, round(r["coef"]["outPxL"], 3), round(r["se"]["outPxL"], 3), round(r["coef"]["outCxL"], 3), round(r["se"]["outCxL"], 3), r["rank"])
