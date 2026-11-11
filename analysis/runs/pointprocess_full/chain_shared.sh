#!/bin/bash
# Shared-activity models: simulation, per-member fits and diagnostics, then the bootstrap for every member.
cd ~/x_box/pipeline/manuscript/epjds_submission_work/pointprocess_full
export PYTHONPATH=~/x_box/pyextra:~/x_box/pylib_v7
PY=~/x_box/venv/bin/python
$PY run_shared.py sim --workers 16 > sh_sim.log 2>&1
$PY run_shared.py fit --workers 16 > sh_fit.log 2>&1
touch SH_FIT_DONE
$PY run_shared.py boot --workers 16 > sh_boot.log 2>&1 || exit 1
touch SH_ALL_DONE
