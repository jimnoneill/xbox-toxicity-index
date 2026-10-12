"""Find pages that end early because the next page opens with a figure, and delay that figure.

Usage: layout_check.py <pdf> <delays.json>
Exit status 0 when the layout needs no change, 1 when delays.json was updated.
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

TOP, BOTTOM = 60.0, 722.0          # body area in points (header above, footer below)
MAX_GAP = 0.18 * (BOTTOM - TOP)    # white space tolerated at the foot of a page
MAX_DELAY = 5

pdf, dpath = sys.argv[1], Path(sys.argv[2])
delays = json.loads(dpath.read_text()) if dpath.exists() else {}
doc = pymupdf.open(pdf)


def content(page):
    items = []
    for b in page.get_text("blocks"):
        if b[1] > TOP - 5 and b[3] < BOTTOM + 8 and b[4].strip():
            items.append((b[1], b[3], "text", b[4]))
    for im in page.get_image_info():
        x0, y0, x1, y1 = im["bbox"]
        if y0 > TOP - 5:
            items.append((y0, y1, "img", ""))
    return sorted(items)


changed = []
for i in range(len(doc) - 1):
    cur, nxt = content(doc[i]), content(doc[i + 1])
    if not cur or not nxt:
        continue
    gap = BOTTOM - max(y1 for _, y1, _, _ in cur)
    if gap <= MAX_GAP or nxt[0][2] != "img":
        continue
    caps = [re.match(r"Figure (S?\d+)\.", t.strip()) for _, _, k, t in nxt if k == "text"]
    caps = [m.group(1) for m in caps if m]
    if not caps:
        continue
    lab = caps[0]
    if delays.get(lab, 0) < MAX_DELAY:
        delays[lab] = delays.get(lab, 0) + 1
        changed.append(f"page {i + 1}: gap {gap:.0f} pt before Figure {lab} -> delay {delays[lab]}")

order = []
for page in doc:
    for b in sorted(page.get_text("blocks"), key=lambda b: b[1]):
        m = re.match(r"Figure (S?)(\d+)\.", b[4].strip())
        if m and m.group(0) not in [o[0] for o in order]:
            order.append((m.group(0), int(m.group(2))))
nums = [n for _, n in order]
print("figure order:", nums, "OK" if nums == sorted(nums) else "OUT OF ORDER")

for c in changed:
    print(c)
dpath.write_text(json.dumps(delays, indent=1))
print("pages", len(doc), "| changes", len(changed), "| delays", delays)
sys.exit(1 if changed else 0)
