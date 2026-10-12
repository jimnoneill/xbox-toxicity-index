import pandas as pd, os
R=os.path.expanduser("~/x_box/pipeline")
n=pd.read_parquet(f"{R}/data/congresstweets_full/congresstweets_full.parquet")
o=pd.read_parquet(f"{R}/data/congresstweets_2017_2023.parquet")
print(n.dtypes.to_dict()==o.dtypes.to_dict(), list(n.columns)==list(o.columns))
print("rows",len(n),"days",n._date.nunique(),"accounts",n.screen_name.nunique(),"dup ids",n.id.duplicated().sum())
print(n.groupby("year").agg(posts=("id","size"),accts=("screen_name","nunique")).to_string())
print("old days",o._date.nunique(),"old rows",len(o))
od=set(o._date); sub=n[n._date.isin(od)]
print("new rows on old days",len(sub))
a=set(o.id); b=set(sub.id)
print("old ids missing from new",len(a-b),"new-only ids on old days",len(b-a))
print("old accounts not in new",len(set(o.screen_name)-set(n.screen_name)))
# row-level equality on shared ids
m=o.drop_duplicates("id").merge(n.drop_duplicates("id"),on="id",suffixes=("_o","_n"))
for c in ["screen_name","time","text","user_id","_date","created_at"]:
    print(c,"mismatch",(m[c+"_o"]!=m[c+"_n"]).sum())
print("ids in old but anywhere in new", len(a-set(n.id)))
