"""emotion4_score.py: all four class probabilities of the general-purpose emotion model, in the class order of the
benchmark it was trained on.

Class order. cardiffnlp/twitter-roberta-base-emotion was trained on TweetEval emotion, whose mapping file gives
0 anger, 1 joy, 2 optimism, 3 sadness; the model card tells users to read the order from that file. The id2label
table in the model's config.json lists a different order and must not be used. The order is fixed here in
EMOTION_CLASSES and is tested on sentences of known emotion before any scoring starts.

Same checkpoint, preprocessing (@user, http), max_length=128, fp32 and length-sorted batching as
ti_classify_probs.py and full_corpus_score.py. Resumable. Keeps VRAM under about 8 GB.

Usage: python emotion4_score.py --corpus full|lee|lee_official|labels1k [--gpu 0]
Outputs (output/v7/emotion4/): <corpus>.parquet with id (or row), p_emotion_anger, p_emotion_joy, p_emotion_optimism,
p_emotion_sadness, emotion_class; <corpus>.json with the run record and, where an earlier score exists, its comparison
with the third output (which the earlier runs stored under the name anger).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.expanduser("~/x_box/pipeline")
OUT = f"{ROOT}/output/v7/emotion4"
MID = "cardiffnlp/twitter-roberta-base-emotion"
EMOTION_CLASSES = ["anger", "joy", "optimism", "sadness"]
KNOWN = [("anger", "I am furious. This is an outrage and I am so angry at these people."),
         ("joy", "What a wonderful day! I am so happy and delighted, congratulations to everyone!"),
         ("optimism", "We can do this. Together we will build a better future and I am hopeful."),
         ("sadness", "I am heartbroken. So sad to hear of his passing, we mourn this terrible loss.")]

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True, choices=["full", "lee", "lee_official", "labels1k"])
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--bs", type=int, default=128)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
DEVICE = f"cuda:{args.gpu}"
torch.cuda.set_per_process_memory_fraction(min(0.33, 8.0 * 1024 ** 3 / torch.cuda.get_device_properties(args.gpu).total_memory), args.gpu)


def preprocess(text):
    out = []
    for t in text.split():
        out.append("@user" if t.startswith("@") and len(t) > 1 else "http" if t.startswith("http") else t)
    return " ".join(out)


old = None  # earlier stored score (third output under the name anger) and its argmax flag, where available
if args.corpus == "full":
    base = pd.read_parquet(f"{ROOT}/output/v7/full/post_probs_full.parquet", columns=["id", "p_emotion_anger", "argmax_emotion_anger"])
    tx = pd.read_parquet(f"{ROOT}/data/congresstweets_full/congresstweets_full.parquet", columns=["id", "text"]).drop_duplicates("id")
    df = base[["id"]].merge(tx, on="id", how="left")
    assert len(df) == len(base) and df.text.notna().all(), "text missing for some scored posts"
    old = base
elif args.corpus == "lee":
    df = pd.read_parquet(f"{ROOT}/data/BasedMikeLee_archive_v2.parquet", columns=["id", "text"])
    o = pd.read_parquet(f"{ROOT}/output/v7/ti/post_probs_lee.parquet", columns=["id", "p_emotion_anger", "argmax_emotion_anger"])
    old = df[["id"]].astype(str).merge(o.astype({"id": str}), on="id", how="left")
elif args.corpus == "lee_official":
    df = pd.read_parquet(f"{ROOT}/data/SenMikeLee_2022_2026.parquet", columns=["id", "text"])
else:
    df = pd.read_parquet(glob.glob(f"{ROOT}/data/*_advocacy_anger_1k.parquet")[0]).reset_index(drop=True)
    df["id"] = np.arange(len(df)).astype(str)
texts = df.text.astype(str).tolist()
n = len(texts)

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForSequenceClassification.from_pretrained(MID).to(DEVICE).eval()


def probs(batch):
    enc = tok([preprocess(t) for t in batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()


# guard: the fixed class order must hold on sentences of known emotion
g = probs([t for _, t in KNOWN])
got = [EMOTION_CLASSES[i] for i in g.argmax(1)]
assert got == [e for e, _ in KNOWN] and g.max(1).min() > 0.8, ("class order check failed", got, g.round(3).tolist())
print("class order check passed", got, flush=True)

mm_path, prog_path = f"{OUT}/{args.corpus}_probs.npy", f"{OUT}/{args.corpus}_progress.json"
order = np.argsort([len(t) for t in texts], kind="stable")
if os.path.exists(mm_path) and os.path.exists(prog_path):
    P = np.lib.format.open_memmap(mm_path, mode="r+")
    start = json.load(open(prog_path))["done"]
    assert P.shape == (n, 4)
else:
    P = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32, shape=(n, 4))
    start = 0
print(f"rows {n:,} start {start:,} device {DEVICE} ({torch.cuda.get_device_name(args.gpu)})", flush=True)
t0 = time.time()
for k in range(start, n, args.bs):
    idx = order[k:k + args.bs]
    P[idx] = probs([texts[i] for i in idx])
    if (k // args.bs) % 500 == 0 and k > start:
        P.flush()
        json.dump({"done": k + args.bs}, open(prog_path, "w"))
        print(f"  {k:,}/{n:,} {(k - start) / (time.time() - t0):.0f}/s", flush=True)
P.flush()
json.dump({"done": n}, open(prog_path, "w"))
P = np.asarray(P)

res = pd.DataFrame({"id": df.id.astype(str).values})
for j, e in enumerate(EMOTION_CLASSES):
    res["p_emotion_" + e] = P[:, j]
res["emotion_class"] = pd.Categorical.from_codes(P.argmax(1), EMOTION_CLASSES)
res.to_parquet(f"{OUT}/{args.corpus}.parquet", index=False)
rec = dict(corpus=args.corpus, n=n, model_id=MID, commit=getattr(model.config, "_commit_hash", None), class_order=EMOTION_CLASSES,
           config_id2label_not_used={str(k): v for k, v in model.config.id2label.items()}, max_length=128, batch=args.bs,
           seconds=round(time.time() - t0), torch=torch.__version__,
           mean_prob={e: float(P[:, j].mean()) for j, e in enumerate(EMOTION_CLASSES)},
           class_share={e: float((P.argmax(1) == j).mean()) for j, e in enumerate(EMOTION_CLASSES)})
if old is not None:
    ok = old.p_emotion_anger.notna().values
    d = np.abs(P[ok, 2] - old.p_emotion_anger.values[ok].astype(np.float64))
    rec["earlier_score_named_anger_vs_third_output"] = dict(
        n=int(ok.sum()), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), share_within_0p001=float((d < 1e-3).mean()),
        argmax_flag_agrees_with_optimism_class=float(((P[ok].argmax(1) == 2).astype(int) == old.argmax_emotion_anger.values[ok].astype(int)).mean()))
json.dump(rec, open(f"{OUT}/{args.corpus}.json", "w"), indent=1)
print(json.dumps(rec, indent=1), flush=True)
print("DONE", flush=True)
