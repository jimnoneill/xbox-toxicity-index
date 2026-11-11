#!/bin/bash
cd ~/x_box/pipeline/manuscript/epjds_submission_work/pointprocess_full
export PYTHONPATH=~/x_box/pyextra:~/x_box/pylib_v7
PY=~/x_box/venv/bin/python
$PY run_events.py cache > ev_cache.log 2>&1 || exit 1
$PY run_events.py fit --workers 16 > ev_fit.log 2>&1 || exit 1
touch EV_FIT_DONE
$PY run_events.py validate --workers 16 > ev_validate.log 2>&1
$PY run_events.py validate --dir temporal_full --workers 16 > full_validate.log 2>&1
touch EV_VAL_DONE
$PY run_events.py boot --workers 16 > ev_boot.log 2>&1 || exit 1
touch EV_ALL_DONE
