#!/usr/bin/env python3
"""Typeset manuscript_v7_epjds.md and supplement_v7.md as journal-style .docx.

Usage (WSL):  ~/x_box_venv/bin/python typeset_docx.py
Reads the markdown sources and figures_main.md / figures_supp.md (captions),
figs/*.png (images). Writes manuscript_v7_epjds.docx and supplement_v7.docx
next to this script. Never modifies the markdown.
"""
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.image.image import Image as DocxImage
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Emu

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figs"
FONT = "Arial"
BLACK = RGBColor(0, 0, 0)
TEXT_W_DXA = 9360  # 6.5 in

SHORT_TITLE = "Forecasting provocation in the US Congress"

AUTHORS = [("Jamey O’Neill", "a,b"), ("Jose Cabanillas", "c"), ("Wayne Kepner", "d"),
           ("Stephen N. Goggin", "e"), ("Joanna Brooks", "c")]
AFFILS = [
    ("a", "Department of Computer Science, San Diego State University, San Diego, CA, USA"),
    ("b", "Department of Mechanical Engineering, San Diego State University, San Diego, CA, USA"),
    ("c", "Department of English and Comparative Literature, San Diego State University, San Diego, CA, USA"),
    ("d", "Department of Psychiatry and Behavioral Sciences, Stanford University School of Medicine, Stanford, CA, USA"),
    ("e", "Department of Political Science, San Diego State University, San Diego, CA, USA"),
]
CORRESPONDING = "Corresponding author: Jamey O’Neill (joneilliii@sdsu.edu)"

SUPP_TABLE_CAPTIONS = {  # index (0-based) of tables lacking a caption paragraph
    1: "Table S2. Rank correlation of member Toxicity Index scores under alternative index constructions.",
    2: "Table S3. Classifier validity against the two-coder consensus (area under the curve).",
    3: "Table S4. Held-out forecast skill on the 2009 to 2022 replication panel (targets 2019 to 2022).",
}


# --------------------------------------------------------------------------- text
def smart_quotes(s):
    s = re.sub(r"(?<=\w)'(?=\w)", "’", s)
    s = re.sub(r"(^|[\s(\[—-])\"", lambda m: m.group(1) + "“", s)
    s = s.replace('"', "”")
    s = re.sub(r"(^|[\s(\[])'", lambda m: m.group(1) + "‘", s)
    s = s.replace("'", "’")
    return s


def segments(text):
    """Return [(text, dict(italic, sub))] after quotes/backticks/italics/subscripts."""
    text = smart_quotes(text.replace("`", ""))
    out = []
    for i, part in enumerate(re.split(r"\*([^*]+)\*", text)):
        ital = i % 2 == 1
        for tok in re.split(r"(A_n|\blog2\b)", part):
            if not tok:
                continue
            if tok == "A_n":
                out.append(("A", dict(italic=ital)))
                out.append(("n", dict(italic=ital, sub=True)))
            elif tok == "RMSE_persistence":
                out.append(("RMSE", dict(italic=ital)))
                out.append(("persistence", dict(italic=ital, sub=True)))
            elif tok == "log2":
                out.append(("log", dict(italic=ital)))
                out.append(("2", dict(italic=ital, sub=True)))
            else:
                out.append((tok, dict(italic=ital)))
    return out


# --------------------------------------------------------------------------- xml helpers
def set_rfonts(rPr):
    for el in rPr.findall(qn("w:rFonts")):
        rPr.remove(el)
    rf = OxmlElement("w:rFonts")
    for a in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + a), FONT)
    rPr.insert(0, rf)


def strip_theme_color(rPr):
    for el in rPr.findall(qn("w:color")):
        rPr.remove(el)
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "000000")
    rPr.append(c)


def style_rpr(style):
    el = style.element
    rPr = el.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        el.append(rPr)
    return rPr


def fix_style(style):
    rPr = style_rpr(style)
    set_rfonts(rPr)
    strip_theme_color(rPr)
    # keep schema order loosely valid: rFonts first, color anywhere before sz is tolerated by Word/LO


def fix_all_fonts(doc):
    styles = doc.styles
    dd = styles.element.find(qn("w:docDefaults"))
    if dd is not None:
        rPr = dd.find(qn("w:rPrDefault") + "/" + qn("w:rPr"))
        if rPr is not None:
            set_rfonts(rPr)
            strip_theme_color(rPr)
    for st in styles:
        if st.type in (1, 2, 3):  # paragraph, character, table
            fix_style(st)
    # drop unused built-in table styles (they carry theme colours/fonts)
    for st in list(styles.element.findall(qn("w:style"))):
        if st.get(qn("w:type")) == "table" and st.get(qn("w:default")) != "1":
            styles.element.remove(st)
    # paragraph borders in styles (e.g. the built-in Title rule) -> black
    for bd in styles.element.iter(qn("w:pBdr")):
        for side in list(bd):
            side.set(qn("w:color"), "000000")
            for a in ("themeColor", "themeShade", "themeTint"):
                if side.get(qn("w:" + a)) is not None:
                    del side.attrib[qn("w:" + a)]
    # drop the legacy stylesWithEffects part (duplicate styles with theme colours, unused by Word 2010+ and LibreOffice)
    for rid, rel in list(doc.part.rels.items()):
        if rel.reltype.endswith("/stylesWithEffects"):
            del doc.part.rels[rid]
    # theme fonts -> Arial
    for rel in doc.part.rels.values():
        if rel.reltype.endswith("/theme"):
            part = rel.target_part
            xml = part.blob.decode("utf8")
            xml = re.sub(r'(<a:(?:major|minor)Font>\s*<a:latin typeface=")[^"]*(")', r"\1Arial\2", xml)
            xml = re.sub(r'(<a:(?:ea|cs) typeface=")[^"]*(")', r"\1Arial\2", xml)
            part._blob = xml.encode("utf8")


def setup_style(doc, name, size, bold=False, italic=False, before=0, after=0, align=None,
                line=None, keep_next=False, base="Normal"):
    try:
        st = doc.styles[name]
    except KeyError:
        from docx.enum.style import WD_STYLE_TYPE
        st = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    pPr = st.element.find(qn("w:pPr"))
    if pPr is not None:
        for tag in ("w:pBdr", "w:contextualSpacing", "w:ind", "w:jc"):
            for el in pPr.findall(qn(tag)):
                pPr.remove(el)
    rPr = style_rpr(st)
    for tag in ("w:sz", "w:szCs", "w:b", "w:bCs", "w:i", "w:iCs", "w:spacing", "w:kern", "w:u", "w:caps"):
        for el in rPr.findall(qn(tag)):
            rPr.remove(el)
    f = st.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = BLACK
    set_rfonts(rPr)
    strip_theme_color(rPr)
    pf = st.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = keep_next
    pf.keep_together = keep_next
    pf.widow_control = True
    if align is not None:
        pf.alignment = align
    if line is not None:
        pf.line_spacing = line
    return st


def para_border(p, top=False, bottom=False, sz=4, space=4):
    pPr = p._p.get_or_add_pPr()
    bd = OxmlElement("w:pBdr")
    for side, on in (("top", top), ("bottom", bottom)):
        if on:
            e = OxmlElement("w:" + side)
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(sz))
            e.set(qn("w:space"), str(space))
            e.set(qn("w:color"), "000000")
            bd.append(e)
    # pBdr must precede spacing/ind/jc in pPr; insert after keepNext/keepLines/pageBreakBefore etc.
    anchor = None
    for tag in ("w:shd", "w:tabs", "w:suppressAutoHyphens", "w:spacing", "w:ind", "w:contextualSpacing", "w:jc"):
        anchor = pPr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(bd)
    else:
        pPr.append(bd)


def add_run(p, text, size, bold=False, italic=False, sub=False, sup=False):
    r = p.add_run(text)
    rPr = r._r.get_or_add_rPr()
    set_rfonts(rPr)
    r.font.size = Pt(size)
    r.font.bold = bold or None
    r.font.italic = italic or None
    r.font.color.rgb = BLACK
    if sub:
        r.font.subscript = True
    if sup:
        r.font.superscript = True
    # normalise rPr child order: rFonts first
    return r


def add_text(p, text, size, bold=False, italic=False):
    for t, pr in segments(text):
        add_run(p, t, size, bold=bold, italic=italic or pr.get("italic", False), sub=pr.get("sub", False))


def new_para(doc, style="Normal", size=11, align=None, before=None, after=None, first_indent=None,
             left=None, hanging=None, keep_next=None, line=None):
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    if align is not None:
        pf.alignment = align
    if before is not None:
        pf.space_before = Pt(before)
    if after is not None:
        pf.space_after = Pt(after)
    if first_indent is not None:
        pf.first_line_indent = Inches(first_indent)
    if left is not None:
        pf.left_indent = Inches(left)
    if hanging is not None:
        pf.left_indent = Inches(hanging)
        pf.first_line_indent = Inches(-hanging)
    if keep_next is not None:
        pf.keep_with_next = keep_next
    if line is not None:
        pf.line_spacing = line
    pf.widow_control = True
    return p


# --------------------------------------------------------------------------- document setup
def field_run(p, instr, size):
    def mk(kind):
        r = add_run(p, "", size)
        fc = OxmlElement("w:fldChar")
        fc.set(qn("w:fldCharType"), kind)
        r._r.append(fc)
        return r
    mk("begin")
    r = add_run(p, "", size)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = instr
    r._r.append(it)
    mk("separate")
    add_run(p, "1", size)
    mk("end")


def build_base(header_text):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(1)
    sec.header_distance = Inches(0.5)
    sec.footer_distance = Inches(0.5)

    fix_all_fonts(doc)
    st = setup_style(doc, "Normal", 11, line=1.15, after=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    setup_style(doc, "Title", 16, bold=True, after=12, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.0, keep_next=True)
    setup_style(doc, "Heading 1", 12, bold=True, before=14, after=6, align=WD_ALIGN_PARAGRAPH.LEFT,
                line=1.15, keep_next=True)
    setup_style(doc, "Heading 2", 11, bold=True, italic=True, before=10, after=4,
                align=WD_ALIGN_PARAGRAPH.LEFT, line=1.15, keep_next=True)
    setup_style(doc, "Heading 3", 11, bold=True, italic=True, before=10, after=4,
                align=WD_ALIGN_PARAGRAPH.LEFT, line=1.15, keep_next=True)
    setup_style(doc, "Caption", 9, before=4, after=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.0)
    setup_style(doc, "Header", 8, align=WD_ALIGN_PARAGRAPH.RIGHT, line=1.0)
    setup_style(doc, "Footer", 9, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.0)
    for nm in ("List Paragraph", "List", "List Bullet", "List Number"):
        try:
            setup_style(doc, nm, 11, line=1.15, after=6)
        except KeyError:
            pass
    try:
        doc.styles["Hyperlink"]
    except KeyError:
        from docx.enum.style import WD_STYLE_TYPE
        doc.styles.add_style("Hyperlink", WD_STYLE_TYPE.CHARACTER)
    hl = doc.styles["Hyperlink"]
    hl.font.color.rgb = BLACK
    hl.font.underline = False
    set_rfonts(style_rpr(hl))
    strip_theme_color(style_rpr(hl))

    # settings: compatibility mode 15
    settings = doc.settings.element
    compat = settings.find(qn("w:compat"))
    if compat is None:
        compat = OxmlElement("w:compat")
        settings.append(compat)
    cs = OxmlElement("w:compatSetting")
    cs.set(qn("w:name"), "compatibilityMode")
    cs.set(qn("w:uri"), "http://schemas.microsoft.com/office/word")
    cs.set(qn("w:val"), "15")
    compat.append(cs)

    # header/footer
    sec.different_first_page_header_footer = True
    hp = sec.header.paragraphs[0]
    hp.style = doc.styles["Header"]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_run(hp, header_text, 8)
    fp0 = sec.first_page_header.paragraphs[0]
    fp0.style = doc.styles["Header"]
    for ft in (sec.footer, sec.first_page_footer):
        p = ft.paragraphs[0]
        p.style = doc.styles["Footer"]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field_run(p, "PAGE", 9)

    cp = doc.core_properties
    cp.author = "Jamey O’Neill"
    cp.last_modified_by = "Jamey O’Neill"
    cp.comments = ""
    return doc


def title_block(doc, line1, line1_size, line2=None):
    p = new_para(doc, "Title", after=12 if not line2 else 4)
    add_run(p, line1, line1_size, bold=True)
    if line2:
        p2 = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.LEFT, after=12, line=1.0)
        add_text(p2, line2, 12)
    pa = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.LEFT, after=6, line=1.15, keep_next=True)
    for i, (name, sup) in enumerate(AUTHORS):
        if i:
            add_run(pa, ", ", 11)
        add_run(pa, name, 11)
        add_run(pa, sup, 11, sup=True)
    for k, (letter, txt) in enumerate(AFFILS):
        p = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.LEFT, after=0, line=1.0, keep_next=True)
        add_run(p, letter, 9, sup=True)
        add_run(p, " " + txt, 9)
    p = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=10, line=1.0)
    add_run(p, CORRESPONDING, 9)


# --------------------------------------------------------------------------- parsing
def strip_comments(s):
    return re.sub(r"<!--.*?-->", "", s, flags=re.S)


def parse_blocks(md):
    blocks, buf, tab = [], [], []

    def flush_p():
        nonlocal buf
        if buf:
            blocks.append(("p", " ".join(x.strip() for x in buf)))
            buf = []

    def flush_t():
        nonlocal tab
        if tab:
            rows = []
            for ln in tab:
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    continue
                rows.append(cells)
            blocks.append(("table", rows))
            tab = []

    for ln in md.splitlines():
        s = ln.strip()
        if not s:
            flush_p(); flush_t(); continue
        if s.startswith("|"):
            flush_p(); tab.append(s); continue
        flush_t()
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            flush_p()
            blocks.append(("h", len(m.group(1)), m.group(2).strip()))
            continue
        m = re.match(r"^!\[(.*)\]\((.*)\)$", s)
        if m:
            flush_p()
            blocks.append(("img", m.group(1), m.group(2)))
            continue
        buf.append(s)
    flush_p(); flush_t()
    return blocks


def load_figs(path):
    figs = {}
    if not path.exists():
        return figs
    for alt, fn in re.findall(r"!\[(.*?)\]\((.*?)\)", strip_comments(path.read_text(encoding="utf8"))):
        m = re.match(r"Figure (S?\d+)\.\s*(.*)$", alt, flags=re.S)
        figs[m.group(1)] = (m.group(2).strip(), FIGDIR / Path(fn).name)
    return figs


# --------------------------------------------------------------------------- blocks -> docx
NUM_RE = re.compile(r"^[\s\d.,+\-()%−]*$")


def is_numeric_cell(c):
    c = re.sub(r"\bto\b", " ", c.strip())
    return bool(c) and bool(re.search(r"\d", c)) and bool(NUM_RE.match(c))


def cell_borders(cell, **sides):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders")
        tcPr.append(b)
    for side, sz in sides.items():
        e = OxmlElement("w:" + side)
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
        b.append(e)


def add_table(doc, rows):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    body = rows[1:]
    numeric = []
    for j in range(ncol):
        vals = [r[j] for r in body if r[j].strip()]
        numeric.append(bool(vals) and sum(is_numeric_cell(v) for v in vals) > len(vals) / 2)
    weights = [min(max(max(len(r[j]) for r in rows), 7), 42) for j in range(ncol)]
    tot = sum(weights)
    widths = [max(int(TEXT_W_DXA * w / tot), 800) for w in weights]
    widths[0] += TEXT_W_DXA - sum(widths)

    t = doc.add_table(rows=len(rows), cols=ncol)
    t.autofit = False
    tblPr = t._tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout", "w:tblBorders", "w:tblStyle"):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    w = OxmlElement("w:tblW"); w.set(qn("w:w"), str(TEXT_W_DXA)); w.set(qn("w:type"), "dxa"); tblPr.append(w)
    bd = OxmlElement("w:tblBorders")
    for side, sz in (("top", 12), ("left", 0), ("bottom", 12), ("right", 0), ("insideH", 0), ("insideV", 0)):
        e = OxmlElement("w:" + side)
        if sz:
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        else:
            e.set(qn("w:val"), "nil")
        bd.append(e)
    tblPr.append(bd)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    # tblGrid widths
    grid = t._tbl.tblGrid
    for gc, wd in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(wd))

    for i, r in enumerate(rows):
        row = t.rows[i]
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if i == 0:
            th = OxmlElement("w:tblHeader"); trPr.append(th)
        for j, txt in enumerate(r):
            cell = row.cells[j]
            cell.width = Emu(int(widths[j] * 635))
            p = cell.paragraphs[0]
            p.style = doc.styles["Normal"]
            pf = p.paragraph_format
            pf.space_before = Pt(2.5); pf.space_after = Pt(2.5); pf.line_spacing = 1.0
            pf.first_line_indent = Inches(0)
            pf.alignment = WD_ALIGN_PARAGRAPH.RIGHT if numeric[j] else WD_ALIGN_PARAGRAPH.LEFT
            pf.keep_with_next = i < len(rows) - 1
            add_text(p, txt, 9, bold=(i == 0))
            if i == 0:
                cell_borders(cell, bottom=6)
    return t


def add_figure(doc, label, caption, path):
    img = DocxImage.from_file(str(path))
    w_in = 6.5
    h_in = w_in * img.px_height / img.px_width
    if h_in > 7.5:
        h_in = 7.5
        w_in = h_in * img.px_width / img.px_height
    p = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=0, first_indent=0,
                 keep_next=True, line=1.0)
    p.add_run().add_picture(str(path), width=Inches(w_in), height=Inches(h_in))
    cap = new_para(doc, "Caption", align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=4, after=12, first_indent=0, line=1.0)
    add_run(cap, f"Figure {label}.", 9, bold=True)
    add_text(cap, " " + caption, 9)


def add_caption(doc, text, kind="Table"):
    m = re.match(r"^(Table S?\d+\.)\s*(.*)$", text, flags=re.S)
    cap = new_para(doc, "Caption", align=WD_ALIGN_PARAGRAPH.LEFT, before=10, after=4, first_indent=0,
                   keep_next=True, line=1.0)
    add_run(cap, m.group(1), 9, bold=True)
    add_text(cap, " " + m.group(2), 9)


def typeset(md_path, figs_path, out_path, supplement):
    md = strip_comments(md_path.read_text(encoding="utf8"))
    blocks = parse_blocks(md)
    figs = load_figs(figs_path)
    header = ("Supplementary Information: " + SHORT_TITLE) if supplement else SHORT_TITLE
    doc = build_base(header)

    # title
    assert blocks[0][0] == "h" and blocks[0][1] == 1
    title = blocks[0][2]
    if supplement:
        title = re.sub(r"^Supplementary material:\s*", "", title)
        title_block(doc, "Supplementary Information", 16, line2=title)
    else:
        title_block(doc, smart_quotes(title), 16)
    blocks = blocks[1:]

    placed = set()
    delays = {}
    dpath = HERE / ("figure_delays_supp.json" if supplement else "figure_delays.json")
    if dpath.exists():
        delays = json.loads(dpath.read_text())
    pending = []          # [label, paragraphs_remaining]

    def place_due(force=False):
        nonlocal flush, last_para
        def num(lab):
            return int(re.sub(r"\D", "", lab))
        for item in sorted(pending, key=lambda it: num(it[0])):
            if not (force or item[1] <= 0):
                break  # later figures wait for earlier ones, so numbering stays in order
            cap, path = figs[item[0]]
            add_figure(doc, item[0], cap, path)
            placed.add(item[0])
            pending.remove(item)
            flush = True
            last_para = ("fig",)

    flush = True          # next body paragraph is flush left
    in_refs = False
    in_abstract = False
    table_idx = 0
    pending_caption_done = False
    i = 0
    last_para = None
    n = len(blocks)
    while i < n:
        b = blocks[i]
        kind = b[0]
        if kind == "h":
            level, text = b[1], b[2]
            text = re.sub(r"^(\d+|S\d+)\.\s+", r"\1 ", text)
            in_abstract = text == "Abstract"
            in_refs = text == "References"
            if pending and text in ("Ethics statement", "Data and code availability", "References"):
                place_due(force=True)
            style = "Heading 1" if level == 2 else "Heading 2"
            p = new_para(doc, style, keep_next=True)
            size = 12 if level == 2 else 11
            if in_abstract:
                size = 11
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(4)
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                para_border(p, top=True, sz=4, space=6)
            add_run(p, smart_quotes(text), size, bold=True, italic=(level == 3))
            flush = True
            i += 1
            continue

        if kind == "table":
            # caption handling: previous paragraph is a "Table Sn." caption?
            rows = b[1]
            if last_para is not None and last_para[0] == "cap":
                pass  # caption already written
            else:
                cap = SUPP_TABLE_CAPTIONS.get(table_idx)
                if cap:
                    add_caption(doc, cap)
            add_table(doc, rows)
            table_idx += 1
            last_para = ("table",)
            flush = True
            # trailing space handled via next paragraph's space_before
            i += 1
            # Notes paragraph?
            if i < n and blocks[i][0] == "p" and blocks[i][1].startswith("Notes."):
                p = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.LEFT, before=4, after=12, first_indent=0, line=1.0)
                add_text(p, blocks[i][1], 9)
                last_para = ("note",)
                i += 1
            else:
                last_para = ("table_nonote",)
            continue

        if kind == "img":
            alt, fn = b[1], b[2]
            m = re.match(r"Figure (S?\d+)\.\s*(.*)$", alt, flags=re.S)
            add_figure(doc, m.group(1), m.group(2), FIGDIR / Path(fn).name)
            placed.add(m.group(1))
            flush = True
            last_para = ("fig",)
            i += 1
            continue

        # paragraph
        text = b[1]
        if re.match(r"^Table S?\d+\.", text) and i + 1 < n and blocks[i + 1][0] == "table":
            add_caption(doc, text)
            last_para = ("cap",)
            flush = True
            i += 1
            continue

        if in_abstract:
            nxt_is_end = (i + 1 >= n) or blocks[i + 1][0] != "p"
            p = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0,
                         after=6 if not nxt_is_end else 8, first_indent=0)
            if text.startswith("Keywords:"):
                add_run(p, "Keywords:", 10, bold=True)
                add_text(p, text[len("Keywords:"):], 10)
            else:
                add_text(p, text, 10)
            if nxt_is_end:
                para_border(p, bottom=True, sz=4, space=6)
            flush = True
            last_para = ("abs",)
            i += 1
            continue

        if in_refs:
            p = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=3, hanging=0.3, line=1.0)
            add_text(p, text, 9)
            i += 1
            continue

        p = new_para(doc, "Normal", align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0 if flush else 0.25)
        if last_para and last_para[0] == "table_nonote":
            p.paragraph_format.space_before = Pt(10)
        add_text(p, text, 11)
        flush = False
        last_para = ("p",)
        # figure placement after first callout, optionally delayed by N body paragraphs
        for item in pending:
            item[1] -= 1
        for lab in re.findall(r"(?<![A-Za-z])Figure (S?\d+)\b", text):
            if lab in figs and lab not in placed and all(it[0] != lab for it in pending):
                pending.append([lab, int(delays.get(lab, 0))])
        place_due()
        i += 1

    place_due(force=True)
    missing = set(figs) - placed
    if missing:
        print("WARNING: figures never called out:", missing, file=sys.stderr)
    doc.save(str(out_path))
    print("wrote", out_path)


if __name__ == "__main__":
    typeset(HERE / "manuscript_v7_epjds.md", HERE / "figures_main.md", HERE / "manuscript_v7_epjds.docx", False)
    typeset(HERE / "supplement_v7.md", HERE / "figures_supp.md", HERE / "supplement_v7.docx", True)
