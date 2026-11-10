"""Run on hpcf from ~/x_box/pipeline. Writes thread_audit_REVIEW.txt with the 200 sampled thread links, full texts,
post links, and the existing labels prefilled for a person to confirm or change."""
import pandas as pd

p = pd.read_parquet("output/v7/t1/posts_threads.parquet", columns=["id", "screen_name", "t"])
v = pd.read_csv("output/v7/t1/thread_validation_sample.csv")
ct = pd.read_parquet("data/congresstweets_2017_2023.parquet", columns=["id", "text", "time"])
ct["id"] = ct["id"].astype(str)
txt = ct.drop_duplicates("id").set_index("id")
out = []
out.append("""THREAD-LINK AUDIT: REVIEW SHEET (200 links)

What this is
The analysis joins consecutive posts from one account into a single event (a thread) when they are less than
60 seconds apart or are linked by thread numbering or a thread keyword. To check that rule, 200 joined pairs were
drawn at random from the every-third-day sample of the congresstweets archive. Each block below shows one pair:
the earlier post (PREVIOUS) and the post that the rule joined to it (POST).

What to do
For each block, read both posts and set the LABEL line to one of:
  continuation   the POST continues the same message as PREVIOUS (same thread, same statement, part 2 of 3, etc.)
  separate       the POST is a different message, or a duplicate/near-duplicate of PREVIOUS
  unsure         you cannot tell
The LABEL line is prefilled with the label currently on file (186 continuation, 14 separate). Change any you
disagree with. Leave a label unchanged if you agree. Do not edit anything else.
Save the file with your initials at the end of the name (for example thread_audit_REVIEW_JO.txt).
Links open the post on X if it still exists; the full text is printed here, so the links are only a convenience.

Reviewer:
Date:
""")
missing = 0
for k, r in enumerate(v.itertuples(), 1):
    cur, prev = p.iloc[r.i], p.iloc[r.i - 1]
    a, b = str(prev.id), str(cur.id)
    ta = txt.text.get(a); tb = txt.text.get(b)
    if ta is None or tb is None:
        missing += 1
    lab = "continuation" if r.manual_label == "continuation_same_message" else "separate"
    out.append(f"""==================== T{k:03d} ====================
account: @{r.screen_name}    gap: {r.gap_s} s    joined by: {r.rule}    thread size: {r.thread_size}
PREVIOUS ({txt.time.get(a, '')}) https://x.com/{r.screen_name}/status/{a}
{(ta if ta is not None else r.prev_text)}

POST ({txt.time.get(b, '')}) https://x.com/{r.screen_name}/status/{b}
{(tb if tb is not None else r.text)}

LABEL: {lab}
""")
open("output/v7/t1/thread_audit_REVIEW.txt", "w", encoding="utf-8").write("\n".join(out))
print("blocks", len(v), "full text missing", missing)
