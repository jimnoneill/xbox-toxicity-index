"""Run only the Fig 8 block of the label-patched copy, saving into this directory (originals untouched)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.expanduser("~/x_box/pipeline/scripts/v7"))
import figs_common
figs_common.OUT = OUTDIR
src = open(os.path.join(HERE, "figs_04_lee_temporal_nonrepost.py")).read()
fig8 = src.split("# ------------------------------------------------------------------ Fig 9")[0]
exec(compile(fig8, "figs_04_lee_temporal_nonrepost.py[fig8]", "exec"), {"__name__": "__main__"})
for ext in ("png", "pdf"):
    os.replace(os.path.join(OUTDIR, "fig8_lee_case_study." + ext), os.path.join(OUTDIR, "fig8_lee." + ext))
print("wrote", OUTDIR + "/fig8_lee.png/.pdf")
