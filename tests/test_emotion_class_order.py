"""The emotion model's classes must be named in the order of the benchmark mapping file, not of its config.json."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xbox.classifiers import _not_hate, label_map_for  # noqa: E402
from xbox.config import CLASS_ORDER_OVERRIDES, CLASSIFIER_MODELS  # noqa: E402

CONFIG_JSON_ORDER = {0: "joy", 1: "optimism", 2: "anger", 3: "sadness"}  # what the model's config.json says


def test_emotion_order_ignores_config_json():
    cfg = SimpleNamespace(num_labels=4, id2label=CONFIG_JSON_ORDER)
    assert label_map_for("emotion", cfg) == {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}


def test_other_models_use_their_config():
    cfg = SimpleNamespace(num_labels=2, id2label={0: "non_irony", 1: "irony"})
    assert label_map_for("irony", cfg) == {0: "non_irony", 1: "irony"}


def test_override_checks_number_of_outputs():
    with pytest.raises(ValueError):
        label_map_for("emotion", SimpleNamespace(num_labels=3, id2label={}))


def test_not_hate_spellings():
    assert _not_hate({"not_hate": 0.9, "racism": 0.1}) == 0.9
    assert _not_hate({"not-hate": 0.8, "hate": 0.2}) == 0.8


def test_emotion_order_on_known_sentences():
    """Runs the model when it is available locally; skipped otherwise."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(CLASSIFIER_MODELS["emotion"])
        model = transformers.AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODELS["emotion"]).eval()
    except Exception:
        pytest.skip("emotion model not cached locally")
    known = [("anger", "I am furious. This is an outrage and I am so angry at these people."),
             ("joy", "What a wonderful day! I am so happy and delighted, congratulations to everyone!"),
             ("optimism", "We can do this. Together we will build a better future and I am hopeful."),
             ("sadness", "I am heartbroken. So sad to hear of his passing, we mourn this terrible loss.")]
    with torch.no_grad():
        p = torch.softmax(model(**tok([t for _, t in known], padding=True, return_tensors="pt")).logits, -1)
    names = label_map_for("emotion", model.config)
    assert [names[int(i)] for i in p.argmax(1)] == [e for e, _ in known]
    assert tuple(CLASS_ORDER_OVERRIDES["emotion"]) == ("anger", "joy", "optimism", "sadness")
