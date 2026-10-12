"""
Pipeline configuration — model IDs, paths, thresholds.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Windows mount path for existing data
XBOX_DATA = Path("/mnt/c/x_box") if Path("/mnt/c").exists() else Path("C:/x_box")

# ── Embedding model (MTEB #1 under 1B) ────────────────────────────
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024
EMBEDDING_FALLBACK = "dunzhang/stella_en_400M_v5"

# ── Classification models (CardiffNLP Twitter-RoBERTa suite) ──────
CLASSIFIER_MODELS = {
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "emotion": "cardiffnlp/twitter-roberta-base-emotion",
    "offensive": "cardiffnlp/twitter-roberta-base-offensive",
    "irony": "cardiffnlp/twitter-roberta-base-irony",
    "hate": "cardiffnlp/twitter-roberta-base-hate-multiclass-latest",
}

# ── Toxicity model ────────────────────────────────────────────────
TOXICITY_MODEL = "s-nlp/roberta_toxicity_classifier"

# ── Senator data sources ──────────────────────────────────────────
CONGRESS_LEGISLATORS_URL = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators"
    "/main/legislators-social-media.yaml"
)
CONGRESS_LEGISLATORS_CURRENT_URL = (
    "https://raw.githubusercontent.com/unitedstates/congress-legislators"
    "/main/legislators-current.yaml"
)
SENATOR_TWEETS_DATASET = "m-newhauser/senator-tweets"

# ── Behavioral thresholds ─────────────────────────────────────────
SESSION_GAP_MINUTES = 30       # gap before new session
NIGHT_START_HOUR = 0           # UTC
NIGHT_END_HOUR = 6             # UTC
BURST_WINDOW_MINUTES = 60      # window for burst detection

# ── Virulence score weights ───────────────────────────────────────
VIRULENCE_WEIGHTS = {
    "sentiment_negative": 0.15,
    "emotion_anger": 0.20,
    "offensive": 0.20,
    "toxicity": 0.15,
    "hate": 0.10,
    "irony": 0.05,
    "engagement_controversy": 0.10,
    "burst_bonus": 0.05,
}

# ── Compulsion score weights ──────────────────────────────────────
COMPULSION_WEIGHTS = {
    "activity": 0.20,
    "burstiness": 0.25,
    "night_activity": 0.15,
    "session_intensity": 0.15,
    "reply_reactivity": 0.10,
    "repetition": 0.10,
    "emoji_media_sparsity": 0.05,
}

# ── Batch sizes ───────────────────────────────────────────────────
CLASSIFICATION_BATCH_SIZE = 32
EMBEDDING_BATCH_SIZE = 16
