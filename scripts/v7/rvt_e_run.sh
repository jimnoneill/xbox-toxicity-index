#!/bin/bash
# Runs the RVT-E scripts named on the command line (01 to 06), one after another, from scripts/v7.
cd ~/x_box/pipeline/scripts/v7 || exit 1
export PYTHONPATH=$HOME/x_box/pyextra:$HOME/x_box/pylib_v7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=$HOME/x_box/pipeline/output/v7/rvt_e
mkdir -p "$OUT"
[ -f "$OUT/DESIGN.md" ] || { cp rvt_e_DESIGN.md "$OUT/DESIGN.md"; chmod a-w "$OUT/DESIGN.md"; }
for k in "$@"; do
  f=$(ls rvt_e_${k}_*.py)
  nice -n 10 timeout 7200 "$HOME/x_box/venv/bin/python" "$f" > "$OUT/run_${k}.log" 2>&1
  echo "EXIT $?" >> "$OUT/run_${k}.log"
done
