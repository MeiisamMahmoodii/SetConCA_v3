from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class LatentTextView:
    id: str
    text: str
    view_type: str
    source_id: str | None = None
    generated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LatentEdge:
    source: str
    target: str
    relation: str
    evidence: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardNegative:
    id: str
    text: str
    negative_type: str
    source_id: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LatentSetRow:
    set_id: str
    latent_type: str
    latent_key: str
    source_dataset: str
    source_ids: List[str]
    texts: List[LatentTextView]
    domain: str | None = None
    positive_edges: List[LatentEdge] = field(default_factory=list)
    hard_negatives: List[HardNegative] = field(default_factory=list)
    risk_tag: str | None = None
    usage: str = "train"
    confidence: float | None = None
    validation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "set_id": self.set_id,
            "latent_type": self.latent_type,
            "latent_key": self.latent_key,
            "source_dataset": self.source_dataset,
            "source_ids": list(self.source_ids),
            "texts": [
                {
                    "id": view.id,
                    "text": view.text,
                    "view_type": view.view_type,
                    **({"source_id": view.source_id} if view.source_id is not None else {}),
                    **({"generated": True} if view.generated else {}),
                    **({"metadata": view.metadata} if view.metadata else {}),
                }
                for view in self.texts
            ],
            "usage": self.usage,
        }
        if self.domain is not None:
            row["domain"] = self.domain
        if self.positive_edges:
            row["positive_edges"] = [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    **({"evidence": edge.evidence} if edge.evidence is not None else {}),
                    **({"metadata": edge.metadata} if edge.metadata else {}),
                }
                for edge in self.positive_edges
            ]
        if self.hard_negatives:
            row["hard_negatives"] = [
                {
                    "id": neg.id,
                    "text": neg.text,
                    "negative_type": neg.negative_type,
                    **({"source_id": neg.source_id} if neg.source_id is not None else {}),
                    **({"metadata": neg.metadata} if neg.metadata else {}),
                }
                for neg in self.hard_negatives
            ]
        if self.risk_tag is not None:
            row["risk_tag"] = self.risk_tag
        if self.confidence is not None:
            row["confidence"] = self.confidence
        if self.validation:
            row["validation"] = self.validation
        if self.metadata:
            row["metadata"] = self.metadata
        return row


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _normalize_text_views(raw_views: Iterable[Mapping[str, Any]]) -> List[LatentTextView]:
    views: List[LatentTextView] = []
    for idx, raw in enumerate(raw_views):
        view_id = str(raw.get("id") or f"view_{idx:02d}")
        views.append(
            LatentTextView(
                id=view_id,
                text=str(raw.get("text", "")).strip(),
                view_type=str(raw.get("view_type", "")).strip(),
                source_id=str(raw["source_id"]) if raw.get("source_id") is not None else None,
                generated=bool(raw.get("generated", False)),
                metadata=dict(raw.get("metadata", {})),
            )
        )
    return views


def _normalize_edges(raw_edges: Iterable[Any]) -> List[LatentEdge]:
    edges: List[LatentEdge] = []
    for raw in raw_edges:
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            source, target = raw[0], raw[1]
            relation = raw[2] if len(raw) >= 3 else "positive"
            edges.append(LatentEdge(source=str(source), target=str(target), relation=str(relation)))
        else:
            item = dict(raw)
            edges.append(
                LatentEdge(
                    source=str(item.get("source", "")),
                    target=str(item.get("target", "")),
                    relation=str(item.get("relation", "positive")),
                    evidence=str(item["evidence"]) if item.get("evidence") is not None else None,
                    metadata=dict(item.get("metadata", {})),
                )
            )
    return edges


def _normalize_hard_negatives(raw_negatives: Iterable[Mapping[str, Any]]) -> List[HardNegative]:
    negatives: List[HardNegative] = []
    for idx, raw in enumerate(raw_negatives):
        negatives.append(
            HardNegative(
                id=str(raw.get("id") or f"negative_{idx:02d}"),
                text=str(raw.get("text", "")).strip(),
                negative_type=str(raw.get("negative_type", "hard_negative")).strip(),
                source_id=str(raw["source_id"]) if raw.get("source_id") is not None else None,
                metadata=dict(raw.get("metadata", {})),
            )
        )
    return negatives


def latent_set_from_dict(row: Mapping[str, Any]) -> LatentSetRow:
    return LatentSetRow(
        set_id=str(row.get("set_id", "")).strip(),
        latent_type=str(row.get("latent_type", "")).strip(),
        latent_key=str(row.get("latent_key", "")).strip(),
        source_dataset=str(row.get("source_dataset", "")).strip(),
        source_ids=[str(item) for item in _as_list(row.get("source_ids"))],
        domain=str(row["domain"]).strip() if row.get("domain") is not None else None,
        texts=_normalize_text_views(_as_list(row.get("texts"))),
        positive_edges=_normalize_edges(_as_list(row.get("positive_edges"))),
        hard_negatives=_normalize_hard_negatives(_as_list(row.get("hard_negatives"))),
        risk_tag=str(row["risk_tag"]).strip() if row.get("risk_tag") is not None else None,
        usage=str(row.get("usage", "train")).strip() or "train",
        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
        validation=dict(row.get("validation", {})),
        metadata=dict(row.get("metadata", {})),
    )


def validate_latent_set(row: LatentSetRow, *, min_views: int = 2, min_view_types: int = 2) -> List[str]:
    issues: List[str] = []
    if not row.set_id:
        issues.append("missing_set_id")
    if not row.latent_type:
        issues.append("missing_latent_type")
    if not row.latent_key:
        issues.append("missing_latent_key")
    if not row.source_dataset:
        issues.append("missing_source_dataset")
    if not row.source_ids:
        issues.append("missing_source_ids")
    if len(row.texts) < min_views:
        issues.append(f"too_few_views<{min_views}")

    view_ids = [view.id for view in row.texts]
    view_id_set = set(view_ids)
    if len(view_ids) != len(view_id_set):
        issues.append("duplicate_view_ids")

    for view in row.texts:
        if not view.id:
            issues.append("missing_view_id")
        if not view.text:
            issues.append(f"empty_view_text:{view.id}")
        if not view.view_type:
            issues.append(f"missing_view_type:{view.id}")

    view_types = {view.view_type for view in row.texts if view.view_type}
    if len(view_types) < min_view_types:
        issues.append(f"too_few_view_types<{min_view_types}")

    valid_refs = view_id_set | {str(i) for i in range(len(row.texts))}
    for edge in row.positive_edges:
        if edge.source not in valid_refs:
            issues.append(f"edge_unknown_source:{edge.source}")
        if edge.target not in valid_refs:
            issues.append(f"edge_unknown_target:{edge.target}")
        if not edge.relation:
            issues.append("edge_missing_relation")

    for neg in row.hard_negatives:
        if not neg.id:
            issues.append("missing_negative_id")
        if not neg.text:
            issues.append(f"empty_negative_text:{neg.id}")
        if not neg.negative_type:
            issues.append(f"missing_negative_type:{neg.id}")

    if row.risk_tag and row.usage == "train":
        issues.append("risk_tagged_row_marked_train")

    return issues


def validate_latent_set_dict(row: Mapping[str, Any], *, min_views: int = 2, min_view_types: int = 2) -> List[str]:
    return validate_latent_set(latent_set_from_dict(row), min_views=min_views, min_view_types=min_view_types)

