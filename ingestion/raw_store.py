"""Capa RAW: guarda las respuestas originales + un manifest con hash y fecha para reproducibilidad."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


class RawStore:
    def __init__(self, root: str | Path = "data/raw"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, source: str, dataset: str, key: str, ext: str) -> Path:
        p = self.root / source / dataset / f"{key}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, source: str, dataset: str, key: str, ext: str) -> bool:
        return self.path(source, dataset, key, ext).exists()

    def _log(self, p: Path, source: str, dataset: str, key: str) -> None:
        entry = {
            "path": str(p.relative_to(self.root)), "source": source, "dataset": dataset, "key": key,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.root / "manifest.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def write_json(self, source: str, dataset: str, key: str, obj) -> Path:
        p = self.path(source, dataset, key, "json")
        p.write_text(json.dumps(obj))
        self._log(p, source, dataset, key)
        return p

    def read_json(self, source: str, dataset: str, key: str):
        return json.loads(self.path(source, dataset, key, "json").read_text())

    def write_parquet(self, source: str, dataset: str, key: str, df: pd.DataFrame) -> Path:
        p = self.path(source, dataset, key, "parquet")
        df.to_parquet(p)
        self._log(p, source, dataset, key)
        return p
