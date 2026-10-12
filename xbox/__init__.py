"""
X-Box Pipeline — Classifier-based tweet analysis for political accounts.

Modules:
    data        — Senator handle collection, tweet dataset loading
    classifiers — Multi-head text classification (sentiment, toxicity, etc.)
    embeddings  — Embedding generation using MTEB-leading models
    behavioral  — Compulsive usage pattern detection (temporal/metadata)
    fusion      — Score fusion and hierarchical modeling
    report      — Report generation (JSON, Markdown)
    cli         — Command-line orchestrator
"""

__version__ = "0.1.0"
