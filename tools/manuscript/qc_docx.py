import sys,re,zipfile
from docx import Document
from docx.oxml.ns import qn
def style_font(st):
    while st is not None:
        rf=st.element.find(qn('w:rPr')+'/'+qn('w:rFonts')) if st.element.find(qn('w:rPr')) is not None else None
        if rf is not None: return rf.get(qn('w:ascii'))
        st=st.base_style
    return None
def iter_pars(doc):
    for p in doc.paragraphs: yield p
    for t in doc.tables:
        for r in t.rows:
            for c in r.cells:
                for p in c.paragraphs: yield p
    for s in doc.sections:
        for part in (s.header,s.footer,s.first_page_header,s.first_page_footer):
            for p in part.paragraphs: yield p
for f in ("manuscript_v7_epjds.docx","supplement_v7.docx"):
    d=Document(f); n=bad=0; txt=""
    for p in iter_pars(d):
        txt+=p.text+"\n"
        for r in p.runs:
            n+=1
            fn=r.font.name or style_font(p.style)
            rf=r._r.rPr.find(qn('w:rFonts')) if r._r.rPr is not None else None
            slots=[rf.get(qn('w:'+a)) for a in ('ascii','hAnsi','eastAsia','cs')] if rf is not None else []
            col=r.font.color.rgb if r.font.color and r.font.color.type else None
            if fn!="Arial" or (slots and set(slots)!={"Arial"}) or (col is not None and str(col)!="000000"):
                bad+=1; print("BAD",f,repr(r.text[:30]),fn,slots,col)
    stbad=[s.name for s in d.styles if s.type in(1,2) and style_font(s)!="Arial"]
    z=zipfile.ZipFile(f); allx="".join(z.read(n).decode('utf8','ignore') for n in z.namelist() if n.endswith('.xml'))
    fonts=set(re.findall(r'w:ascii="([^"]+)"',allx))|set(re.findall(r'typeface="([^"]+)"',allx))
    print(f,"runs",n,"bad runs",bad,"styles not Arial",stbad,"themeColor attrs",len(re.findall('themeColor',allx)),"fonts in xml",sorted(fonts)-{''} if False else sorted(x for x in fonts if x))
    print(" straight quotes:",txt.count('"'),txt.count("'"),"dashes:",len(re.findall('[–—]',txt)))
