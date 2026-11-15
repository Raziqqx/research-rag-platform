"""Runtime configuration, read from environment variables.

Large artefacts (PDFs, the Hugging Face model cache) live outside the repo, under
RAG_DATA_DIR (default ~/rag-data), so nothing big is ever committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
    return Path(os.environ.get("RAG_DATA_DIR", Path.home() / "rag-data")).expanduser()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=_data_dir)
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    reranker_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    db_pool_size: int = field(default_factory=lambda: int(os.environ.get("RAG_DB_POOL_SIZE", "8")))
    strategy: str = field(default_factory=lambda: os.environ.get("RAG_STRATEGY", "recursive"))
    # Below this top cross-encoder score /ask abstains. Calibrated in results/abstention.md.
    abstain_threshold: float = field(
        default_factory=lambda: float(os.environ.get("RAG_ABSTAIN_THRESHOLD", "-1.31"))
    )

    @property
    def pdf_dir(self) -> Path:
        return Path(os.environ.get("RAG_PDF_DIR", self.data_dir / "pdfs")).expanduser()

    @property
    def manifest_path(self) -> Path:
        return REPO_ROOT / "corpus" / "manifest.json"


def get_settings() -> Settings:
    # Point the HF cache at the data dir unless the caller already chose one.
    os.environ.setdefault("HF_HOME", str(_data_dir() / "hf-cache"))
    return Settings()
