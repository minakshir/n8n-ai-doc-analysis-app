import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    service_token: str
    max_upload_mb: int
    max_pdf_pages: int
    retention_days: int
    ollama_url: str
    embed_model: str

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    token = os.environ.get("SERVICE_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("SERVICE_TOKEN must be set to at least 32 characters")
    return Settings(
        data_dir=Path(os.environ.get("DATA_DIR", "/data")),
        service_token=token,
        max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "50")),
        max_pdf_pages=int(os.environ.get("MAX_PDF_PAGES", "300")),
        retention_days=int(os.environ.get("RETENTION_DAYS", "7")),
        ollama_url=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text:v1.5"),
    )
