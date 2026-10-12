#!/bin/bash
# render docx -> pdf -> png via hpcf (WSL soffice unavailable)
cd /c/x_box/pipeline/manuscript/v7
H=joneill@192.168.1.223
timeout 60 ssh $H 'rm -rf /tmp/tsr; mkdir -p /tmp/tsr/png'
timeout 120 scp manuscript_v7_epjds.docx supplement_v7.docx $H:/tmp/tsr/
timeout 300 ssh $H 'cd /tmp/tsr && timeout 240 soffice --headless --convert-to pdf *.docx >/dev/null 2>&1; for f in manuscript_v7_epjds supplement_v7; do pdftoppm -r 70 -png $f.pdf png/$f; done; ls png | wc -l; pdffonts manuscript_v7_epjds.pdf | awk "NR>2{print \$1}" | sort -u; pdffonts supplement_v7.pdf | awk "NR>2{print \$1}" | sort -u'
timeout 120 scp $H:/tmp/tsr/*.pdf .
rm -rf prev; mkdir prev
timeout 120 scp $H:/tmp/tsr/png/*.png prev/
ls prev | head -80
