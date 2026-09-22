"""Atomic, checksum-bound writers for complete evaluation runs."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from mcp_route_reliability.hashing import canonical_json_bytes
from mcp_route_reliability.models import ScoreMetadata, StrictModel

REQUIRED_FILES = (
    "predictions.jsonl",
    "features.parquet",
    "metrics.json",
    "errors.csv",
    "policy.json",
    "policy-empirical.json",
    "policy-conservative.json",
    "reliability-card.md",
)
REQUIRED_FIGURES = (
    "figures/reliability-diagram.png",
    "figures/risk-coverage.png",
    "figures/method-comparison.png",
    "figures/robustness.png",
)
RAW_GOLD_COLUMNS = {
    "relevant_tool_ids",
    "should_abstain",
    "abstain_reason",
    "label_rationale",
    "candidate_conflict_note",
}


class RunManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    source_commit: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    tool_corpus_version: str = Field(min_length=1)
    tool_corpus_hash: str = Field(min_length=1)
    router_id: str = Field(min_length=1)
    router_version: str = Field(min_length=1)
    model_version: str | None
    score_metadata: ScoreMetadata
    feature_version: str = Field(min_length=1)
    prediction_importer_version: str = Field(min_length=1)
    random_seed: int
    python_version: str = Field(min_length=1)
    dependency_versions: dict[str, str]
    started_at: str = Field(min_length=1)
    finished_at: str | None = None
    status: Literal["running", "complete", "failed"] = "running"
    error: str | None = None
    commands: tuple[str, ...]
    artifact_checksums: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "complete":
            if self.finished_at is None:
                raise ValueError("complete manifests require finished_at")
            if self.error is not None:
                raise ValueError("complete manifests cannot contain an error")
        if self.status == "failed" and not self.error:
            raise ValueError("failed manifests require an error")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class RunWriter:
    """Build a run privately, validate it, then atomically expose it."""

    def __init__(self, runs_dir: Path, run_id: str) -> None:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a nonempty path-safe name")
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id
        self.final_path = self.runs_dir / run_id
        self._temporary_path: Path | None = None
        self._manifest: RunManifest | None = None
        self._finalized = False

    def __enter__(self) -> RunWriter:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            raise FileExistsError(f"run already exists: {self.final_path}")
        temporary = tempfile.mkdtemp(prefix=f".tmp-{self.run_id}-", dir=self.runs_dir)
        self._temporary_path = Path(temporary)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._finalized and self._temporary_path is not None:
            shutil.rmtree(self._temporary_path, ignore_errors=True)

    def _path(self, relative_path: str) -> Path:
        if self._temporary_path is None:
            raise RuntimeError("RunWriter must be used as a context manager")
        if self._finalized:
            raise RuntimeError("run has already been finalized")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("artifact path must stay inside the run directory")
        target = self._temporary_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_manifest(self, manifest: RunManifest) -> None:
        if manifest.run_id != self.run_id:
            raise ValueError("manifest run_id does not match RunWriter")
        if manifest.status != "running":
            raise ValueError("initial manifest must have status='running'")
        self._manifest = manifest
        self.write_json("manifest.json", manifest)

    def write_json(self, relative_path: str, value: object) -> None:
        self._path(relative_path).write_bytes(canonical_json_bytes(value) + b"\n")

    def write_jsonl(self, relative_path: str, rows: Sequence[object]) -> None:
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        self._path(relative_path).write_bytes(payload)

    def write_text(self, relative_path: str, value: str) -> None:
        self._path(relative_path).write_text(value, encoding="utf-8", newline="\n")

    def write_bytes(self, relative_path: str, value: bytes) -> None:
        self._path(relative_path).write_bytes(value)

    def write_csv(self, relative_path: str, rows: Sequence[Mapping[str, object]]) -> None:
        path = self._path(relative_path)
        fieldnames = list(rows[0]) if rows else []
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)

    def write_features_parquet(self, rows: Sequence[Mapping[str, object]]) -> None:
        frame = pd.DataFrame(rows)
        forbidden = RAW_GOLD_COLUMNS & set(frame.columns)
        if forbidden:
            raise ValueError(
                f"feature artifact contains raw gold-label columns: {sorted(forbidden)}"
            )
        frame.to_parquet(self._path("features.parquet"), index=False)

    def finalize(self) -> Path:
        if self._temporary_path is None:
            raise RuntimeError("RunWriter must be used as a context manager")
        if self._manifest is None:
            raise ValueError("run manifest has not been written")
        required = (*REQUIRED_FILES, *REQUIRED_FIGURES)
        missing = [name for name in required if not (self._temporary_path / name).is_file()]
        if missing:
            raise ValueError(f"missing required artifacts: {missing}")
        if self.final_path.exists():
            raise FileExistsError(f"run already exists: {self.final_path}")

        checksums = {
            path.relative_to(self._temporary_path).as_posix(): _sha256(path)
            for path in sorted(self._temporary_path.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        completed = self._manifest.model_copy(
            update={
                "finished_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "status": "complete",
                "error": None,
                "artifact_checksums": checksums,
            }
        )
        self._manifest = RunManifest.model_validate(completed.model_dump())
        self.write_json("manifest.json", self._manifest)

        os.replace(self._temporary_path, self.final_path)
        self._finalized = True
        self._temporary_path = None
        return self.final_path
