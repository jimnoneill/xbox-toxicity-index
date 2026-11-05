"""Fig 1 (TI7 by party and account type), Fig 2 (specification curve), Fig 3 (party trends with power regimes)."""
from figs_common import *
style()

# ---------------------------------------------------------------- Fig 1
m = pd.read_csv(V7 + "/ti_full/member_ti7_primary.csv")
m = m[(m.n_primary >= 50) & m.party.isin(["R", "D"])]
print("fig1 members", m.party.value_counts().to_dict())
pc = pd.read_csv(V7 + "/stats_full/party_contrasts_ti7_nonrt.csv").set_index("measure").loc["ti7"]
acc = pd.read_csv(V7 + "/ti_full/account_ti7.csv")
acc = acc[acc.party.isin(["R", "D"]) & acc.account_type.isin(["office", "campaign"])]
g = acc.groupby(["bioguide", "account_type", "party"]).apply(
    lambda d: pd.Series({"ti7": np.average(d.ti7, weights=d.n), "n": d.n.sum()}), include_groups=False).reset_index()
g = g[g.n >= 50]
print("fig1b member x type counts\n", g.groupby(["account_type", "party"]).size())
for c in ["office", "campaign"]:
    a = g[(g.account_type == c) & (g.party == "R")].ti7; b = g[(g.account_type == c) & (g.party == "D")].ti7
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    print("fig1b recomputed d", c, (a.mean() - b.mean()) / sp)
spec = pd.read_csv(V7 + "/stats_full/spec_curve_full.csv")
spec = spec[spec.measure == "ti7"].set_index("step")

fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
ax = axs[0]
grid = np.linspace(0.0, 0.55, 300)
for i, (p, col) in enumerate([("D", D_COL), ("R", R_COL)]):
    x = m[m.party == p].ti7.values
    gr = grid[(grid >= x.min()) & (grid <= x.max())]
    d = kde(x, gr); d = d / d.max() * 0.38
    ax.fill_betweenx(gr, i - d, i + d, color=col, alpha=0.25, lw=0)
    ax.plot(i - d, gr, color=col, lw=0.8); ax.plot(i + d, gr, color=col, lw=0.8)
    rng = np.random.default_rng(1)
    ax.scatter(i + rng.uniform(-0.06, 0.06, len(x)), x, s=3, color=col, alpha=0.35, lw=0)
    mu, lo, hi = mean_ci(x)
    ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]], color="black", capsize=3, lw=1.2, marker="o", ms=4, zorder=5)
    ax.text(i, 0.02, f"n = {len(x)}", ha="center", va="bottom", fontsize=8, color=col, bbox=dict(fc="white", ec="none", pad=1))
ax.set_xticks([0, 1]); ax.set_xticklabels(["Democrats", "Republicans"])
ax.set_ylabel("Member TI7 (mean over posts)")
ax.set_ylim(0, 0.6); ax.set_xlim(-0.7, 1.7)
ax.text(0.5, 0.99, f"g = {pc.hedges_g:.2f} [{pc.g_lo:.2f}, {pc.g_hi:.2f}]", transform=ax.transAxes, ha="center", va="top", fontsize=8)
plab(ax, "a", dx=-0.18)

ax = axs[1]
cats = [("office", "Office accounts"), ("campaign", "Campaign accounts")]
gs = {"office": "M5", "campaign": "M6"}
for ci, (c, lab) in enumerate(cats):
    for p, col in [("D", D_COL), ("R", R_COL)]:
        x = g[(g.account_type == c) & (g.party == p)].ti7.values
        xpos = ci + (-0.17 if p == "D" else 0.17)
        rng = np.random.default_rng(2)
        ax.scatter(xpos + rng.uniform(-0.1, 0.1, len(x)), x, s=3, color=col, alpha=0.25, lw=0)
        mu, lo, hi = mean_ci(x)
        ax.errorbar(xpos, mu, yerr=[[mu - lo], [hi - mu]], color="black", capsize=3, marker="o", ms=4.5, mfc=col, mec="black", lw=1.2, zorder=5)
    s = spec.loc[gs[c]]
    ax.text(ci, 0.595, f"g = {s.g:.2f}\n[{s.ci_lo:.2f}, {s.ci_hi:.2f}]", ha="center", va="top", fontsize=8)
ax.set_xticks([0, 1]); ax.set_xticklabels([c[1] for c in cats])
ax.set_ylim(0, 0.6); ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("Member TI7 within account type")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=D_COL, label="Democrats"), Line2D([], [], marker="o", ls="", color=R_COL, label="Republicans")],
          loc="lower right", ncol=2, handletextpad=0.2, columnspacing=0.8)
plab(ax, "b", dx=-0.14)
save(fig, "fig1_ti7_party_account_type")

# ---------------------------------------------------------------- Fig 2
s = pd.read_csv(V7 + "/stats_full/spec_curve_full.csv")
s = s[s.measure == "ti7"].copy()
meta = {  # step -> (unit, posts, minposts, accounts)
    "A-all1": ("account", "all", 50, "119th handles"), "A-all3": ("account", "all", 50, "all mapped"),
    "A-all4": ("account", "all", 50, "office"), "A-all5": ("account", "all", 50, "campaign"),
    "A-nonrt1": ("account", "nonrt", 50, "119th handles"), "A-nonrt3": ("account", "nonrt", 50, "all mapped"),
    "A-nonrt4": ("account", "nonrt", 50, "office"), "A-nonrt5": ("account", "nonrt", 50, "campaign"),
    "A-orig1": ("account", "orig", 50, "119th handles"), "A-orig3": ("account", "orig", 50, "all mapped"),
    "M1": ("member", "all", 50, "all mapped"), "M2": ("member", "nonrt", 50, "all mapped"),
    "M3": ("member", "nonrt", 200, "all mapped"), "M3a": ("member", "all", 200, "all mapped"),
    "M4b": ("member", "nonrt", 50, "switchers: original party"), "M5": ("member", "nonrt", 50, "office"),
    "M6": ("member", "nonrt", 50, "campaign"), "M7": ("member", "orig", 50, "all mapped"),
}
missing = set(s.step) - set(meta)
assert not missing, missing
s = s.sort_values("g").reset_index(drop=True)
print("fig2 n specs", len(s), "median g", s.g.median(), "range", s.g.min(), s.g.max())
rows = [("Unit", [("member", "member"), ("account", "account")]),
        ("Posts", [("all", "all posts"), ("nonrt", "non-retweet"), ("orig", "original only")]),
        ("Min. posts", [(50, "50"), (200, "200")]),
        ("Accounts", [("all mapped", "all mapped"), ("119th handles", "119th Congress handles"), ("office", "office only"),
                      ("campaign", "campaign only"), ("switchers: original party", "switchers, orig. party")])]
fig = plt.figure(figsize=(W2, 5.4))
gsp = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.25], hspace=0.06)
ax = fig.add_subplot(gsp[0]); ax2 = fig.add_subplot(gsp[1], sharex=ax)
x = np.arange(len(s))
prim = s.is_primary.values.astype(bool)
ax.vlines(x, s.ci_lo, s.ci_hi, color=GREY, lw=1.2)
ax.scatter(x[~prim], s.g[~prim], s=22, color=BLACK, zorder=3)
ax.scatter(x[prim], s.g[prim], s=50, color=ORANGE, zorder=4, edgecolor="black", lw=0.6)
ax.axhline(0, color="black", lw=0.7)
med = s.g.median()
ax.axhline(med, color=BLUE, lw=0.9, ls="--")
ax.text(len(s) - 1.2, med - 0.012, f"median {med:.2f}", color=BLUE, ha="right", va="top", fontsize=8)
ax.set_ylabel("Hedges g (R minus D)")
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_xlim(-0.8, len(s) - 0.2)
ax.tick_params(axis="x", length=0)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, mec="black", label="Primary specification"),
                   Line2D([], [], marker="o", ls="", color=BLACK, label="Other specifications (95% CI)")], loc="upper left")
ypos = []
y = 0
sepl = []
for rname, opts in rows:
    for key, lab in opts:
        for xi, st in enumerate(s.step):
            v = meta[st][["Unit", "Posts", "Min. posts", "Accounts"].index(rname)]
            if v == key:
                ax2.scatter(xi, y, s=16, color=ORANGE if prim[xi] else BLACK, zorder=3)
        ypos.append((y, lab)); y -= 1
    sepl.append(y + 0.5); y -= 0.6
ax2.set_yticks([p[0] for p in ypos]); ax2.set_yticklabels([p[1] for p in ypos])
for yy in sepl[:-1]:
    ax2.axhline(yy - 0.3, color="#cccccc", lw=0.6)
ax2.set_ylim(min(p[0] for p in ypos) - 0.8, 0.8)
ax2.set_xticks([])
for sp in ["left", "bottom"]:
    ax2.spines[sp].set_visible(False)
ax2.tick_params(axis="y", length=0)
k = 0
for rname, opts in rows:
    mid = [p[0] for p in ypos[k:k + len(opts)]]
    ax2.text(len(s) - 0.1, np.mean(mid), rname, ha="left", va="center", fontsize=8, color=GREY, style="italic")
    k += len(opts)
ax2.set_xlabel("Specifications ordered by effect size")
save(fig, "fig2_spec_curve")

# ---------------------------------------------------------------- Fig 3
my = pd.read_csv(V7 + "/ti_full/member_year_ti7.csv")
ym = pd.read_csv(V7 + "/ti_full/yearly_means_ti7.csv")
chosen = None
for thr in [1, 10, 20, 30, 50]:
    t = my[(my.n >= thr) & my.party.isin(["R", "D"])].groupby(["year", "party"]).ti7.mean().unstack()
    err = max(abs(t.loc[ym.year, "R"].values - ym.member_year_mean_R.values).max(), abs(t.loc[ym.year, "D"].values - ym.member_year_mean_D.values).max())
    nn = my[(my.n >= thr) & my.party.isin(["R", "D"])].groupby("year").size().loc[ym.year].values
    print("fig3 min n", thr, "max abs diff vs yearly_means", err, "members/yr", nn.tolist())
    if chosen is None and err < 1e-4:
        chosen = thr
print("fig3 chosen threshold", chosen)
chosen = chosen or 1
my = my[(my.n >= chosen) & my.party.isin(["R", "D"])]
ym = ym.set_index("year")


def fy(d):
    return d.year + (d.dayofyear - 1) / 365.25


xs = {}
for yr, r in ym.iterrows():
    a, b = pd.Timestamp(r.first_post), pd.Timestamp(r.last_post)
    xs[yr] = fy(a + (b - a) / 2)
nat = pd.read_csv(V7 + "/t3/national.csv", parse_dates=["date"])


def spans(col):
    ch = nat.loc[nat[col] != nat[col].shift(), ["date", col]].values.tolist()
    out = []
    for i, (d, v) in enumerate(ch):
        e = ch[i + 1][0] if i + 1 < len(ch) else nat.date.max() + pd.Timedelta(days=1)
        out.append((d, e, v))
    return out


x0, x1 = fy(pd.Timestamp("2017-06-21")), fy(pd.Timestamp("2023-07-11"))
tr = pd.read_csv(V7 + "/stats_full/trend_ti7.csv")
tr = tr[(tr.panel == "full_2017_2023") & (tr.model == "mixed_RI")].set_index("party")
fig = plt.figure(figsize=(W2, 3.9))
gsp = fig.add_gridspec(4, 1, height_ratios=[0.16, 0.16, 0.16, 2.6], hspace=0.12)
strips = [fig.add_subplot(gsp[i]) for i in range(3)]
ax = fig.add_subplot(gsp[3])
for sa, (lab, col) in zip(strips, [("President", "president_party"), ("House", "house_majority"), ("Senate", "senate_majority")]):
    for d0, d1, v in spans(col):
        a, b = max(fy(d0), x0), min(fy(d1), x1)
        if b > a:
            sa.axvspan(a, b, color=R_COL if v == "R" else D_COL, alpha=0.75, lw=0)
            if col == "president_party":
                ax.axvspan(a, b, color=R_COL if v == "R" else D_COL, alpha=0.07, lw=0)
    sa.set_xlim(x0, x1); sa.set_yticks([]); sa.set_xticks([])
    sa.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    for sp in sa.spines.values():
        sp.set_visible(False)
for p, col in [("D", D_COL), ("R", R_COL)]:
    rows_ = []
    for yr in sorted(ym.index):
        v = my[(my.year == yr) & (my.party == p)].ti7.values
        mu, lo, hi = mean_ci(v)
        rows_.append((xs[yr], mu, lo, hi, yr))
    a = np.array(rows_)
    ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=col, alpha=0.2, lw=0)
    ax.plot(a[:, 0], a[:, 1], color=col, lw=1.5)
    full = a[:, 4] < 2023
    ax.plot(a[full, 0], a[full, 1], "o", color=col, ms=4)
    ax.plot(a[~full, 0], a[~full, 1], "o", mfc="white", mec=col, ms=4.5)
    s_ = tr.loc[p]
    ax.text(0.015 if p == "D" else 0.985, 0.04, f"{'Democrats' if p == 'D' else 'Republicans'}: {s_.slope:+.3f} per year [{s_.lo:+.3f}, {s_.hi:+.3f}]",
            transform=ax.transAxes, color=col, ha="left" if p == "D" else "right", fontsize=8)
ax.set_xlim(x0, x1)
ax.set_xticks([xs[y] for y in sorted(ym.index)]); ax.set_xticklabels([str(y) for y in sorted(ym.index)])
ax.set_ylim(0.165, 0.32)
ax.set_ylabel("Member-year mean TI7 (95% CI)")
ax.set_xlabel("Calendar year (2017 from 21 Jun; 2023 to 11 Jul)")
save(fig, "fig3_party_trends")
