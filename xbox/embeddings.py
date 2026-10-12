"""
Embedding generation using MTEB-leading models (<1B params).

Primary: Qwen/Qwen3-Embedding-0.6B (MTEB Eng v2: 70.70, Classification: 85.76)
Fallback: dunzhang/stella_en_400M_v5
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_FALLBACK,
    EMBEDDING_MODEL,
    OUTPUT_DIR,
)

log = logging.getLogger(__name__)


class TweetEmbedder:
    """
    Generate dense embeddings for tweets using Qwen3-Embedding-0.6B.
    Embeddings are useful for:
    - SetFit few-shot classification
    - Clustering / topic modeling
    - Nearest-neighbor retrieval of similar tweets
    - Dimensionality reduction visualization (UMAP/t-SNE)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self._model = None
        self._tokenizer = None

    def load(self):
        """Load the embedding model."""
        try:
            log.info("Loading embedding model: %s", self.model_name)
            self._load_model(self.model_name)
        except Exception as e:
            log.warning("Failed to load %s: %s. Trying fallback...", self.model_name, e)
            self.model_name = EMBEDDING_FALLBACK
            self._load_model(self.model_name)
        log.info("Embedding model loaded on device=%s", self.device)

    def _load_model(self, model_name: str):
        """Load model using sentence-transformers or transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=self.device)
            self._use_st = True
            log.info("Using SentenceTransformer backend")
        except Exception:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name).to(self.device).eval()
            self._use_st = False
            log.info("Using raw transformers backend")

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
        instruction: str = "",
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of tweet texts
            batch_size: Batch size for inference
            show_progress: Show tqdm progress bar
            instruction: Optional instruction prefix (for instruction-aware models like Qwen3)

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if self._model is None:
            self.load()

        if self._use_st:
            # SentenceTransformer handles batching internally
            if instruction and hasattr(self._model, "encode"):
                # Qwen3 supports instruction-aware embeddings
                embeddings = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    prompt=instruction,
                    normalize_embeddings=True,
                )
            else:
                embeddings = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=True,
                )
            return np.array(embeddings)

        # Manual batching with transformers
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", leave=False)

        for i in iterator:
            batch = texts[i : i + batch_size]
            if instruction:
                batch = [f"{instruction}{t}" for t in batch]

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self._model(**encoded)
                # Mean pooling over last hidden state
                attention_mask = encoded["attention_mask"]
                hidden = output.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        batch_size: int = EMBEDDING_BATCH_SIZE,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embeddings for all tweets in a DataFrame.
        Optionally save to disk as .npy file.
        """
        texts = df[text_col].tolist()
        instruction = "Classify the sentiment and tone of this tweet: "

        embeddings = self.embed_texts(
            texts,
            batch_size=batch_size,
            instruction=instruction,
        )

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(p), embeddings)
            log.info("Saved embeddings (%s) to %s", embeddings.shape, p)

        return embeddings
