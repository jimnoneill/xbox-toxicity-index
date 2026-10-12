#!/bin/bash
# Build -> render on hpcf -> check page gaps -> adjust figure delays, until the layout settles.
cd /c/x_box/pipeline/manuscript/v7
H=joneill@192.168.1.223
for it in 1 2 3 4 5 6 7 8; do
  echo "=== iteration $it ==="
  wsl bash -c 'cd /mnt/c/x_box/pipeline/manuscript/v7 && ~/x_box_venv/bin/python typeset_docx.py' | tail -2
  timeout 60 ssh $H 'rm -rf /tmp/tsr; mkdir -p /tmp/tsr'
  timeout 120 scp -q manuscript_v7_epjds.docx supplement_v7.docx $H:/tmp/tsr/
  timeout 300 ssh $H 'cd /tmp/tsr && timeout 240 soffice --headless --convert-to pdf *.docx >/dev/null 2>&1; ls *.pdf'
  timeout 120 scp -q $H:/tmp/tsr/manuscript_v7_epjds.pdf $H:/tmp/tsr/supplement_v7.pdf .
  wsl bash -c 'cd /mnt/c/x_box/pipeline/manuscript/v7 && ~/x_box_venv/bin/python layout_check.py manuscript_v7_epjds.pdf figure_delays.json'; a=$?
  wsl bash -c 'cd /mnt/c/x_box/pipeline/manuscript/v7 && ~/x_box_venv/bin/python layout_check.py supplement_v7.pdf figure_delays_supp.json'; b=$?
  if [ $a -eq 0 ] && [ $b -eq 0 ]; then echo "layout settled after $it iteration(s)"; break; fi
done
