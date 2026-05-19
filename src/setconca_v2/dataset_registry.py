from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .io_utils import read_json


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    hf_dataset: str
    builder: str
    status: str
    primary_use: str
    splits: List[str]
    latent_types: List[str]
    license: str | None = None
    notes: str | None = None
    config: str | None = None
    metadata: Dict[str, Any] | None = None


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "dataset_sources.json"


def source_from_dict(row: Mapping[str, Any]) -> DatasetSource:
    return DatasetSource(
        source_id=str(row.get("source_id", "")).strip(),
        hf_dataset=str(row.get("hf_dataset", "")).strip(),
        builder=str(row.get("builder", "")).strip(),
        status=str(row.get("status", "")).strip(),
        primary_use=str(row.get("primary_use", "")).strip(),
        splits=[str(item) for item in row.get("splits", [])],
        latent_types=[str(item) for item in row.get("latent_types", [])],
        license=str(row["license"]).strip() if row.get("license") is not None else None,
        notes=str(row["notes"]).strip() if row.get("notes") is not None else None,
        config=str(row["config"]).strip() if row.get("config") is not None else None,
        metadata=dict(row.get("metadata", {})),
    )


def validate_source(source: DatasetSource) -> List[str]:
    issues: List[str] = []
    if not source.source_id:
        issues.append("missing_source_id")
    if not source.hf_dataset:
        issues.append(f"{source.source_id}:missing_hf_dataset")
    if not source.builder:
        issues.append(f"{source.source_id}:missing_builder")
    if source.status not in {"build_now", "optional_eval", "future_work"}:
        issues.append(f"{source.source_id}:unknown_status:{source.status}")
    if not source.primary_use:
        issues.append(f"{source.source_id}:missing_primary_use")
    if not source.splits:
        issues.append(f"{source.source_id}:missing_splits")
    if not source.latent_types:
        issues.append(f"{source.source_id}:missing_latent_types")
    return issues


def load_dataset_registry(path: str | Path | None = None) -> List[DatasetSource]:
    registry_path = Path(path) if path is not None else default_registry_path()
    raw = read_json(registry_path)
    rows = raw.get("sources", raw)
    if not isinstance(rows, list):
        raise ValueError("dataset registry must be a list or an object with a 'sources' list")
    sources = [source_from_dict(row) for row in rows]
    issues = []
    seen = set()
    for source in sources:
        issues.extend(validate_source(source))
        if source.source_id in seen:
            issues.append(f"duplicate_source_id:{source.source_id}")
        seen.add(source.source_id)
    if issues:
        raise ValueError("invalid dataset registry: " + "; ".join(issues))
    return sources


def filter_sources(sources: Iterable[DatasetSource], *, status: str | None = None) -> List[DatasetSource]:
    return [source for source in sources if status is None or source.status == status]

