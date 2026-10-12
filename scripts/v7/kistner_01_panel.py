import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "xbox"))
import numpy as np, pandas as pd
import kistner_ext as K
OUT = Path.home() / "x_box/pipeline/output/v7/kistner"; OUT.mkdir(parents=True, exist_ok=True)
res = {}
for name, modes in [("tw", ("Twitter",)), ("twfb", ("Twitter", "Facebook"))]:
    pan = K.build_panel(modes)
    fb = K.build_panel(("Facebook",)) if name == "tw" else K.build_panel(("Facebook",))
    t = K.build_transitions(pan, fb)
    t.to_csv(f"{K.DATA}/transitions_{name}.csv", index=False)
    pan.to_csv(f"{K.DATA}/panel_{name}.csv", index=False)
    res[name] = dict(n_panel_rows=int(len(pan[pan.n >= K.MIN_N])), n_members_panel=int(pan[pan.n >= K.MIN_N].MemberICPSR.nunique()),
                     n_transitions=int(len(t)), n_members_trans=int(t.MemberICPSR.nunique()),
                     chamber_missing_panel=int(pan.chamber.isna().sum()), senators_trans=int((t.chamber == "Senate").sum()),
                     per_year=t.groupby("Y").size().to_dict(), per_year_party=t.groupby(["Y", "party"]).size().unstack().to_dict())
    print(name, json.dumps(res[name], default=str)[:1500])
json.dump(res, open(OUT / "panel_counts.json", "w"), indent=1, default=str)
