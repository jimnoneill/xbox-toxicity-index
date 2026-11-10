"""Relabelled re-render of supplementary Figure S2 (notebook reproduce_manuscript_final*.ipynb cell 19,
'RQ1_gbt_matched_repair'). Same values, same whiskers, read from ../rq1_gbt_v2 (read-only), with the
main design (strict input timing) first. Writes only figS2_timing_sensitivity.png/.pdf here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
G = HERE.parent / "rq1_gbt_v2"
sys.path.insert(0, str(HERE.parents[2] / "scripts/v7"))
import figs_common as common  # noqa: E402

gbt_eval = json.loads((G / "aligned_evaluation.json").read_text())
gbt_pred = pd.concat([pd.read_parquet(p) for p in sorted((G / "predictions_aligned").glob("origin_*.parquet"))],
                     ignore_index=True)
assert len(gbt_pred) == gbt_eval["n"] == 422529 and gbt_pred.thread_id.is_unique
rows = []
for prefix, key in [("aligned", "repaired"), ("orig", "legacy_matched")]:  # main design first
    p_mse = (gbt_pred.ti7 - gbt_pred[f"{prefix}_p"]).pow(2).mean()
    r_mse = (gbt_pred.ti7 - gbt_pred[f"{prefix}_r"]).pow(2).mean()
    ref = gbt_eval["models"][key]
    assert abs((1 - r_mse / p_mse) - ref["skill"]) < 1e-6
    rows.append({"skill": ref["skill"], "recomputed_float32": 1 - r_mse / p_mse, "ci_lo": ref["ci95_skill"][0], "ci_hi": ref["ci95_skill"][1]})
vals = pd.DataFrame(rows)
print(vals.assign(**{c: 100 * vals[c] for c in vals}).to_string())

common.style()
common.plt.rcParams["font.family"] = "Arial"
fig, ax = common.plt.subplots(figsize=(7.2, 3.5))
heights = 100 * vals.skill.to_numpy()
ax.bar([0, 1], heights, width=0.48, color=[common.BLACK, common.GREY])
ax.errorbar([0, 1], heights, yerr=[100 * (vals.skill - vals.ci_lo).to_numpy(), 100 * (vals.ci_hi - vals.skill).to_numpy()],
            fmt="none", ecolor="black", capsize=4)
ax.set_xticks([0, 1], ["Strict input timing (main analysis)", "Less strict input timing (sensitivity)"])
ax.set_ylabel("Rhythm MSE skill over content history (%)")
ax.set_title("Rhythm gain under two input-timing rules, same 422,529 events")
diff = abs(100 * gbt_eval["paired_skill_change_repaired_minus_legacy"])
ax.text(0.5, 0.98, "Paired difference: %.3f pp" % diff, ha="center", va="top", transform=ax.transAxes, fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "figS2_timing_sensitivity.pdf", bbox_inches="tight")
fig.savefig(HERE / "figS2_timing_sensitivity.png", dpi=600, bbox_inches="tight")
print("saved figS2_timing_sensitivity", "paired difference pp", diff)
