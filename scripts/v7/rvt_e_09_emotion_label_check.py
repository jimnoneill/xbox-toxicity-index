"""RVT-E follow-up: which output position of the four-class emotion model is which emotion.

The scoring scripts took the "anger" probability from the position that the model's config.json names "anger".
The model card instead tells users to take the class order from the TweetEval mapping file
(datasets/emotion/mapping.txt: 0 anger, 1 joy, 2 optimism, 3 sadness). The two orders differ. This script runs the model
on sentences whose emotion is not in doubt, records where the probability lands, and rewrites the label tables of
e8_anger_scores.json under the order the test supports. CPU only; uses the locally cached model."""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.expanduser("~/x_box/pipeline/output/v7")
O = ROOT + "/rvt_e"
MID = "cardiffnlp/twitter-roberta-base-emotion"
TWEETEVAL = ["anger", "joy", "optimism", "sadness"]          # mapping.txt of the benchmark the model was trained on
TESTS = [("anger", "I am furious. This is an outrage and I am so angry at these people."),
         ("anger", "I hate this. They make me so mad I could scream."),
         ("anger", "How dare they. I am livid and disgusted by this betrayal."),
         ("joy", "What a wonderful day! I am so happy and delighted, congratulations to everyone!"),
         ("joy", "Happy birthday to my amazing daughter, so much fun today!"),
         ("joy", "We won the championship! Best night ever, I can't stop smiling."),
         ("optimism", "We can do this. Together we will build a better future and I am hopeful."),
         ("optimism", "I believe brighter days are ahead. Keep going, never give up."),
         ("optimism", "Don't lose hope. Things will get better and we will get through this."),
         ("sadness", "I am heartbroken. So sad to hear of his passing, we mourn this terrible loss."),
         ("sadness", "Feeling lonely and depressed tonight, I miss her so much."),
         ("sadness", "I can't stop crying. Everything feels empty since she left.")]

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForSequenceClassification.from_pretrained(MID).eval()
config_names = [model.config.id2label[i] for i in range(4)]
with torch.no_grad():
    p = torch.softmax(model(**tok([t for _, t in TESTS], padding=True, truncation=True, max_length=128, return_tensors="pt")).logits, -1).numpy()
rows = [dict(intended=e, argmax_position=int(r.argmax()), probs=[float(x) for x in r], text=t) for (e, t), r in zip(TESTS, p)]
by_emotion = {e: sorted({r["argmax_position"] for r in rows if r["intended"] == e}) for e in TWEETEVAL}
tested = [next((e for e in TWEETEVAL if by_emotion[e] == [i]), None) for i in range(4)]
RES = dict(model_id=MID, commit=getattr(model.config, "_commit_hash", None), config_order=config_names, tweeteval_mapping_order=TWEETEVAL,
           order_supported_by_test=tested, test_agrees_with_tweeteval=bool(tested == TWEETEVAL), test_agrees_with_config=bool(tested == config_names),
           sentences=rows,
           position_used_as_anger_in_scoring=config_names.index("anger"),
           what_that_position_is=tested[config_names.index("anger")],
           stored_label_to_true_emotion={config_names[i]: tested[i] for i in range(4)})

# the label tables of e8_anger_scores.json, renamed
ren = RES["stored_label_to_true_emotion"]
if all(ren.values()) and os.path.exists(O + "/e8_anger_scores.json"):
    C = json.load(open(O + "/e8_anger_scores.json"))["C_labels"]
    RES["sample_tables_under_true_names"] = dict(
        n_posts=C["n_posts"], share={ren[k]: v for k, v in C["share"].items()},
        by_true_emotion={ren[k]: v for k, v in C["by_label"].items()},
        true_emotion_of_genuine_anger_posts={ren[k]: v for k, v in C["labels_of_genuine_anger_posts"].items()},
        true_emotion_of_top_decile_ti7_posts={ren[k]: v for k, v in C["labels_of_top_decile_ti7_posts"].items()},
        share_change_by_group={tr: {(ren[k] if k in ren else k): v for k, v in d.items()} for tr, d in C["share_change_by_group"].items()},
        lost_minus_gained={tr: {ren[k]: v for k, v in d.items()} for tr, d in C["lost_minus_gained"].items()})
json.dump(RES, open(O + "/e9_emotion_labels.json", "w"), indent=1)
print(json.dumps({k: v for k, v in RES.items() if k != "sentences"}, indent=1))
for r in rows:
    print(r["intended"], r["argmax_position"], [round(x, 3) for x in r["probs"]])
