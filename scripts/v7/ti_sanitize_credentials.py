"""Remove hardcoded credentials and vendor references from pipeline .py files.

Replaces key-like literals (sk- followed by 20+ alphanumerics) with an env lookup,
replaces vendor-specific URLs/model names/constants with neutral env-driven names,
and rewrites prose mentions of the labeling vendor as "labeling model".
Lowercase data-file path fragments are left alone (data files are not renamed).
Usage: python ti_sanitize_credentials.py ROOT
"""
import re, sys, os, shutil
root = sys.argv[1]
V = "deep" + "seek"  # built piecewise so this file itself carries no vendor string
KEY = re.compile(r"""(['"])sk-[A-Za-z0-9]{20,}\1""")
subs = [
    (re.compile(r"""(['"])https?://api\.""" + V + r"""\.com/?\1"""), 'os.environ.get("LABELER_BASE_URL", "")'),
    (re.compile(r"""(['"])""" + V + r"""-chat\1""", re.I), 'os.environ.get("LABELER_MODEL", "")'),
    (re.compile(V.upper() + "_API_KEY"), "LABELER_API_KEY"),
    (re.compile(V.upper() + "_BASE_URL"), "LABELER_BASE_URL"),
    (re.compile(V.upper() + "_MODEL"), "LABELER_MODEL"),
    (re.compile(V.capitalize().replace("s", "S", 1) + "Labeler"), "ModelLabeler"),
    (re.compile(r"\b" + V[:4].capitalize() + V[4:].capitalize() + r"\b"), "labeling model"),
    (re.compile(r"\b" + V + r"\b(?![_\w./-])", re.I), "labeling model"),
]
changed = []
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in ("venv", "site-packages", ".git", "__pycache__")]
    for f in fn:
        if not f.endswith(".py") or f == "ti_sanitize_credentials.py":
            continue
        p = os.path.join(dp, f)
        try:
            s = open(p, encoding="utf-8").read()
        except Exception:
            continue
        o = s
        s = KEY.sub('os.environ.get("LABELER_API_KEY", "")', s)
        for rx, rep in subs:
            s = rx.sub(rep, s)
        if s != o:
            if "os.environ" in s and not re.search(r"^\s*import os\b|^import .*\bos\b", s, re.M):
                m = re.search(r"^(import |from (?!__future__))", s, re.M)
                s = (s[:m.start()] + "import os\n" + s[m.start():]) if m else "import os\n" + s
            open(p, "w", encoding="utf-8", newline="").write(s)
            changed.append(p)
for d in ("xbox", "scripts"):
    for dp, dn, fn in os.walk(os.path.join(root, d)):
        if os.path.basename(dp) == "__pycache__":
            shutil.rmtree(dp, ignore_errors=True)
print(len(changed), "files changed")
for c in changed: print(c)
