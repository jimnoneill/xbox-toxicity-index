#!/bin/bash
# P1 full pipeline: bash p1_all.sh subset|full
S=${1:-full}
cd ~/x_box/pipeline
export PYTHONPATH=~/x_box/pyextra:~/x_box/pylib_v7 OMP_NUM_THREADS=16
PY=~/x_box/venv/bin/python
D=scripts/v7
set -e
$PY $D/p1_build.py --source $S
$PY $D/p1_run.py --source $S --stage tune
$PY $D/p1_run.py --source $S --stage dev
$PY $D/p1_report.py --source $S --stage dev
$PY $D/p1_run.py --source $S --stage test
$PY $D/p1_report.py --source $S --stage test
$PY $D/p1_final.py --source $S
$PY $D/p1_lee.py --source $S
$PY $D/p1_numbers.py --source $S
echo ALLDONE
