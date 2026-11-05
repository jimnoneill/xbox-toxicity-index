"""POST-HOC exploratory (not in DESIGN.md): H2 heterogeneity stratified by body type; flagged as deviation."""
import sys, json
sys.path.insert(0, "/home/joneill/x_box/pipeline/scripts/v7")
from rvt_a_common import *
r = json.load(open(f"{OUT}/h2_results.json"))
b = {int(k): v for k, v in r["homogeneity"]["betas"].items()}; s = {int(k): v for k, v in r["homogeneity"]["ses"].items()}
def pair(a, c):
    d = b[c] - b[a]; se = np.hypot(s[a], s[c]); z = d / se
    return dict(diff=d, se=se, z=z, p=float(2 * (1 - stats.norm.cdf(abs(z)))))
o = {"presidency_2017_vs_2021": pair(2017, 2021), "house_2011_vs_2019": pair(2011, 2019), "senate2015_vs_house2011": pair(2011, 2015)}
m = dl_meta([b[2011], b[2015], b[2019]], [s[2011], s[2015], s[2019]]); o["chamber_type_Q"] = {k: m[k] for k in ("Q", "df", "p_Q", "I2")}
pb = np.array([x["b"] for x in r["placebo"]["both"].values()])
o["placebo_sd_D_minus_R"] = float(pb.std(ddof=1)); o["placebo_range"] = [float(pb.min()), float(pb.max())]
# does the 2017-vs-2021 diff exceed placebo noise: z using placebo sd
o["pres_diff_vs_placebo_sd"] = float((b[2021] - b[2017]) / (pb.std(ddof=1) * np.sqrt(2)))
o["house_diff_vs_placebo_sd"] = float((b[2019] - b[2011]) / (pb.std(ddof=1) * np.sqrt(2)))
json.dump(o, open(f"{OUT}/h2_exploratory_posthoc.json", "w"), indent=1, default=float); print(json.dumps(o, indent=1, default=float))
