"""Run on hpcf inside manuscript/epjds_submission_work/fig_relabel_v30. Writes fig5_v3.py and fig7_v3.py;
the originals are copied, never edited."""
import shutil
from pathlib import Path

H = Path(__file__).resolve().parent
shutil.copy(H.parent / "rq1_strict_v2_fig5" / "fig5_v2.py", H / "fig5_v3.py")
p = H / "fig5_v3.py"
s = p.read_text()
reps = [
    ('R = json.loads((HERE / "results.json").read_text())',
     'R = json.loads((HERE.parent / "rq1_strict_v2_fig5" / "results.json").read_text())'),
    ('("Rhythm, GBT", "m_R_vs_m_P")', '("Rhythm added", "m_R_vs_m_P")'),
    ('"Rhythm, GBT, both\\nwith previous-event\\ntype controls"', '"Rhythm added, both\\nmodels also given\\nprevious-event type"'),
    ('"Rhythm, GBT, both\\nwith previous and\\nown post type"', '"Rhythm added, both\\nalso given first-post\\nreply status"'),
    ('("Rhythm, mixed model"', '("Rhythm added,\\nmixed model"'),
    ('"Scheduler share"', '"Prior management-\\nclient use"'),
    ('"Lagged Congress"', '"Lagged Congress-\\nwide activity"'),
    ('"Regime flags"', '"Calendar indicators"'),
    ('"Previous post type"', '"Previous event type"'),
    ('"Own post type\\n(diagnostic)"', '"First post is a\\nreply (diagnostic)"'),
    ('"Rhythm, no Congress"', '"Rhythm without\\nCongress-wide activity"'),
    ('HERE / "fig5_strict_v2.pdf"', 'HERE / "fig5_v3.pdf"'),
    ('HERE / "fig5_strict_v2.png"', 'HERE / "fig5_v3.png"'),
]
for a, b in reps:
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)
p.write_text(s)

# Figure 7: take the Fig 7 block of figs_03_forecast.py, relabel, save to this folder.
src = (H.parents[2] / "scripts" / "v7" / "figs_03_forecast.py").read_text()
start = src.index("# ------------------------------------------------------------------ Fig 7")
block = src[start:]
reps7 = [
    ('"Control features, GBT"', '"Profile model"'),
    ('"Control plus power features"', '"Power-status model"'),
    ('"Party-year oracle"', '"Group-year oracle"'),
    ('("Power-feature model", wm(', '("Power-status model", wm('),
    ('label="Power-feature model"', 'label="Power-status model"'),
    ('("Party-mean baseline", wm(', '("Training party-mean baseline", wm('),
    ('label="Party-mean baseline"', 'label="Training party-mean baseline"'),
    ('"Next-year RMSE skill vs persistence"', '"RMSE skill over\\npredicting no change"'),
    ('"Change in party-mean TI7, 2021 to 2022"', '"Change in party-mean TI7,\\npolitical year 2021 to 2022"'),
    ('save(fig, "fig7_trajectory_skill")', 'save(fig, "fig7_v3")'),
]
for a, b in reps7:
    assert block.count(a) == 1, (a, block.count(a))
    block = block.replace(a, b)
head = ("import sys, json\nsys.path.insert(0, %r)\nimport figs_common\nfigs_common.OUT = %r\n"
        "from figs_common import *\nstyle()\n" % (str(H.parents[2] / "scripts" / "v7"), str(H)))
(H / "fig7_v3.py").write_text(head + block)
print("written fig5_v3.py fig7_v3.py")
