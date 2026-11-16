"""On-disk layout:

    {data_dir}/sessions/{session_id}/{dataset_id}/
        source.<ext>          original upload
        profile.json          what /ingest returned
        text.txt              extracted text (txt, pdf)
        tables/t1.parquet     typed tables
    {data_dir}/tmp/           upload staging (same filesystem, so moves are cheap)
"""

import json
import re
import secrets
import shutil
import time
from pathlib import Path

import pandas as pd

from .errors import ServiceError
from .parsers import ParsedFile

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
DATASET_RE = re.compile(r"^ds_[0-9a-f]{12}$")
TABLE_RE = re.compile(r"^t\d{1,4}$")


class Storage:
    def __init__(self, data_dir: Path):
        self.sessions = data_dir / "sessions"
        self.tmp_dir = data_dir / "tmp"
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_dataset_id() -> str:
        return "ds_" + secrets.token_hex(6)

    def session_dir(self, session_id: str) -> Path:
        if not SESSION_RE.match(session_id or ""):
            raise ServiceError("BAD_SESSION_ID", "session_id must be 1-128 letters, digits, '-' or '_'.")
        return self.sessions / session_id

    def dataset_dir(self, session_id: str, dataset_id: str) -> Path:
        if not DATASET_RE.match(dataset_id or ""):
            raise ServiceError("BAD_DATASET_ID", "Invalid dataset_id.")
        path = self.session_dir(session_id) / dataset_id
        if not path.is_dir():
            raise ServiceError("NOT_FOUND", f"Dataset {dataset_id} not found in this session.", status=404)
        return path

    def save(self, session_id: str, dataset_id: str, source: Path, parsed: ParsedFile, profile: dict) -> None:
        session = self.session_dir(session_id)
        staging = self.tmp_dir / f"{dataset_id}.staging"
        (staging / "tables").mkdir(parents=True)
        try:
            shutil.copyfile(source, staging / f"source{source.suffix}")
            for table, meta in zip(parsed.tables, profile["tables"]):
                table.df.to_parquet(staging / "tables" / f"{meta['table_id']}.parquet", index=False)
            if parsed.text is not None:
                (staging / "text.txt").write_text(parsed.text, encoding="utf-8")
            (staging / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=1))
            session.mkdir(parents=True, exist_ok=True)
            staging.rename(session / dataset_id)
            session.touch()  # mtime drives retention cleanup
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def list_profiles(self, session_id: str) -> list[dict]:
        session = self.session_dir(session_id)
        if not session.is_dir():
            return []
        profiles = [json.loads(p.read_text()) for p in session.glob("ds_*/profile.json")]
        return sorted(profiles, key=lambda p: p["created_at"])

    def read_text(self, session_id: str, dataset_id: str) -> str | None:
        path = self.dataset_dir(session_id, dataset_id) / "text.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def load_table(self, session_id: str, dataset_id: str, table_id: str) -> pd.DataFrame:
        if not TABLE_RE.match(table_id or ""):
            raise ServiceError("BAD_TABLE_ID", "Invalid table_id.")
        path = self.dataset_dir(session_id, dataset_id) / "tables" / f"{table_id}.parquet"
        if not path.exists():
            raise ServiceError("NOT_FOUND", f"Table {table_id} not found in {dataset_id}.", status=404)
        return pd.read_parquet(path)

    def delete_session(self, session_id: str) -> bool:
        session = self.session_dir(session_id)
        if not session.is_dir():
            return False
        shutil.rmtree(session)
        return True

    def cleanup(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86400
        removed = 0
        for session in self.sessions.iterdir():
            if session.is_dir() and session.stat().st_mtime < cutoff:
                shutil.rmtree(session, ignore_errors=True)
                removed += 1
        for staging in self.tmp_dir.iterdir():  # leftovers from crashed uploads
            if staging.stat().st_mtime >= time.time() - 3600:
                continue
            if staging.is_dir():
                shutil.rmtree(staging, ignore_errors=True)
            else:
                staging.unlink(missing_ok=True)
        return removed
