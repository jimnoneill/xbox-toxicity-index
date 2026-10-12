"""
Multi-head text classification — sentiment, emotion, hate, offensive, irony, toxicity.

Uses CardiffNLP Twitter-RoBERTa suite + RoBERTa toxicity classifier.
Each model produces calibrated probabilities per class.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import CLASSIFICATION_BATCH_SIZE, CLASSIFIER_MODELS, TOXICITY_MODEL

log = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Per-tweet classification output across all heads."""
    tweet_id: str
    text: str
    # Sentiment: negative, neutral, positive
    sentiment_label: str = ""
    sentiment_scores: dict = field(default_factory=dict)
    # Emotion: anger, joy, optimism, sadness
    emotion_label: str = ""
    emotion_scores: dict = field(default_factory=dict)
    # Offensive: not-offensive, offensive
    offensive_label: str = ""
    offensive_score: float = 0.0
    # Irony: non_irony, irony
    irony_label: str = ""
    irony_score: float = 0.0
    # Hate: not-hate, or type (sexism, racism, etc.)
    hate_label: str = ""
    hate_scores: dict = field(default_factory=dict)
    # Toxicity: neutral, toxic
    toxicity_label: str = ""
    toxicity_score: float = 0.0


class MultiHeadClassifier:
    """
    Loads all classification heads and runs inference on tweet batches.
    All models are ~125M params (RoBERTa-base), except toxicity (~355M).
    Total: ~980M params across all heads — well under budget on CPU.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict = {}
        self._tokenizers: dict = {}
        self._label_maps: dict = {}

    def load_all(self):
        """Load all classification models."""
        log.info("Loading classification models on device=%s", self.device)

        for name, model_id in CLASSIFIER_MODELS.items():
            log.info("  Loading %s: %s", name, model_id)
            self._tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
            self._models[name] = AutoModelForSequenceClassification.from_pretrained(
                model_id
            ).to(self.device).eval()

            # Extract label mapping from model config
            config = self._models[name].config
            if hasattr(config, "id2label"):
                self._label_maps[name] = config.id2label
            else:
                self._label_maps[name] = {i: str(i) for i in range(config.num_labels)}

        # Toxicity model
        log.info("  Loading toxicity: %s", TOXICITY_MODEL)
        self._tokenizers["toxicity"] = AutoTokenizer.from_pretrained(TOXICITY_MODEL)
        self._models["toxicity"] = AutoModelForSequenceClassification.from_pretrained(
            TOXICITY_MODEL
        ).to(self.device).eval()
        self._label_maps["toxicity"] = {0: "neutral", 1: "toxic"}

        log.info("All %d classification heads loaded.", len(self._models))

    def _infer_batch(
        self, name: str, texts: list[str]
    ) -> list[dict[str, float]]:
        """Run a single model on a batch of texts. Returns list of {label: prob}."""
        tokenizer = self._tokenizers[name]
        model = self._models[name]
        label_map = self._label_maps[name]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for row in probs:
            results.append({label_map[i]: float(row[i]) for i in range(len(row))})
        return results

    def classify_tweets(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        id_col: str = "tweet_id",
        batch_size: int = CLASSIFICATION_BATCH_SIZE,
    ) -> pd.DataFrame:
        """
        Run all classification heads on a DataFrame of tweets.
        Returns a new DataFrame with classification columns added.
        """
        if not self._models:
            self.load_all()

        texts = df[text_col].tolist()
        ids = df[id_col].tolist() if id_col in df.columns else list(range(len(texts)))
        n = len(texts)

        # Collect all head results
        all_results = {name: [] for name in self._models}

        for name in self._models:
            log.info("Running %s classifier on %d tweets...", name, n)
            for i in tqdm(range(0, n, batch_size), desc=name, leave=False):
                batch = texts[i : i + batch_size]
                # Preprocess for CardiffNLP models
                batch = [_preprocess_tweet(t) for t in batch]
                results = self._infer_batch(name, batch)
                all_results[name].extend(results)

        # Build output columns
        out = df.copy()

        # Sentiment
        if "sentiment" in all_results:
            sent = all_results["sentiment"]
            out["sentiment_negative"] = [s.get("negative", s.get("LABEL_0", 0)) for s in sent]
            out["sentiment_neutral"] = [s.get("neutral", s.get("LABEL_1", 0)) for s in sent]
            out["sentiment_positive"] = [s.get("positive", s.get("LABEL_2", 0)) for s in sent]
            out["sentiment_label"] = [max(s, key=s.get) for s in sent]

        # Emotion
        if "emotion" in all_results:
            emo = all_results["emotion"]
            for label in ["anger", "joy", "optimism", "sadness"]:
                out[f"emotion_{label}"] = [
                    e.get(label, 0) for e in emo
                ]
            out["emotion_label"] = [max(e, key=e.get) for e in emo]

        # Offensive
        if "offensive" in all_results:
            off = all_results["offensive"]
            out["offensive_score"] = [
                o.get("offensive", o.get("LABEL_1", 0)) for o in off
            ]
            out["offensive_label"] = [max(o, key=o.get) for o in off]

        # Irony
        if "irony" in all_results:
            iro = all_results["irony"]
            out["irony_score"] = [
                i.get("irony", i.get("LABEL_1", 0)) for i in iro
            ]
            out["irony_label"] = [max(i, key=i.get) for i in iro]

        # Hate
        if "hate" in all_results:
            hate = all_results["hate"]
            out["hate_score"] = [
                1.0 - h.get("not-hate", h.get("LABEL_0", 1.0)) for h in hate
            ]
            out["hate_label"] = [max(h, key=h.get) for h in hate]

        # Toxicity
        if "toxicity" in all_results:
            tox = all_results["toxicity"]
            out["toxicity_score"] = [t.get("toxic", t.get(1, 0)) for t in tox]
            out["toxicity_label"] = [
                "toxic" if t.get("toxic", t.get(1, 0)) > 0.5 else "neutral"
                for t in tox
            ]

        log.info("Classification complete. Added %d columns.", len(out.columns) - len(df.columns))
        return out


def _preprocess_tweet(text: str) -> str:
    """
    Preprocess tweet text for CardiffNLP models.
    - Replace @mentions with @user
    - Replace URLs with http
    """
    tokens = text.split()
    processed = []
    for t in tokens:
        if t.startswith("@") and len(t) > 1:
            processed.append("@user")
        elif t.startswith("http"):
            processed.append("http")
        else:
            processed.append(t)
    return " ".join(processed)
