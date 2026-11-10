#!/bin/bash
# Sequential pipeline; each stage logs separately.
cd $(dirname $0)
export PYTHONPATH=$HOME/x_box/pyextra OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
PY=$HOME/x_box/venv/bin/python
for s in monthly final; do $PY -u fig5_v2.py --stage $s > log_$s.log 2>&1; done
$PY -u fig5_v2.py --stage lee --mode archived > log_lee_archived.log 2>&1
$PY -u fig5_v2.py --stage lee --mode strict > log_lee_strict.log 2>&1
$PY -u fig5_v2.py --stage lmm > log_lmm.log 2>&1
echo ALLDONE > log_done.txt
