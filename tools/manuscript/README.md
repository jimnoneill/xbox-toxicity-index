# Manuscript typesetting

`typeset_docx.py` builds the journal-style Word manuscript and supplement (Arial, black text, three-line tables, figures placed after their first callout) from the Markdown sources. `layout_loop.sh` renders the documents to PDF with LibreOffice, and `layout_check.py` delays any figure that would leave a page ending early, repeating until the layout settles and checking that figures stay in numerical order. `qc_docx.py` verifies that every run is Arial and black.
