from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

from .latent_set_schema import HardNegative, LatentEdge, LatentSetRow, LatentTextView, validate_latent_set


SUPPORTED_LABELS = {"SUPPORTS", "REFUTES"}


@dataclass(frozen=True)
class FeverBuildResult:
    accepted: List[LatentSetRow] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)


def normalize_fever_title(title: Any) -> str:
    return str(title or "").strip()


def clean_fever_wiki_sentence(line_text: str) -> str:
    parts = str(line_text or "").split("\t")
    sentence = parts[0] if parts else ""
    return " ".join(sentence.split())


def parse_fever_wiki_lines(lines: str) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for raw_line in str(lines or "").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t", 1)
        if len(parts) != 2:
            continue
        try:
            sent_id = int(parts[0])
        except ValueError:
            continue
        text = clean_fever_wiki_sentence(parts[1])
        if text:
            parsed[sent_id] = text
    return parsed


def build_wiki_sentence_index(wiki_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[int, str]]:
    index: Dict[str, Dict[int, str]] = {}
    for row in wiki_rows:
        page_id = normalize_fever_title(row.get("id"))
        if not page_id:
            continue
        index[page_id] = parse_fever_wiki_lines(str(row.get("lines", "")))
    return index


def evidence_text_for_record(record: Mapping[str, Any], wiki_sentence_index: Mapping[str, Mapping[int, str]]) -> str | None:
    page = normalize_fever_title(record.get("evidence_wiki_url"))
    try:
        sent_id = int(record.get("evidence_sentence_id", -1))
    except (TypeError, ValueError):
        return None
    return wiki_sentence_index.get(page, {}).get(sent_id)


def first_fever_evidence_item(record: Mapping[str, Any]) -> tuple[Any, Any, Any, Any] | None:
    evidence_groups = record.get("evidence")
    if not isinstance(evidence_groups, list):
        return None
    for group in evidence_groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, (list, tuple)) and len(item) >= 4 and item[2] is not None and item[3] is not None:
                return item[0], item[1], item[2], item[3]
    return None


def normalize_fever_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    if normalized.get("evidence_wiki_url") is None or normalized.get("evidence_sentence_id") is None:
        evidence_item = first_fever_evidence_item(record)
        if evidence_item is not None:
            annotation_id, evidence_id, wiki_url, sentence_id = evidence_item
            normalized.setdefault("evidence_annotation_id", annotation_id)
            normalized.setdefault("evidence_id", evidence_id)
            normalized["evidence_wiki_url"] = wiki_url
            normalized["evidence_sentence_id"] = sentence_id
    return normalized


def required_fever_pages(records: Iterable[Mapping[str, Any]]) -> set[str]:
    pages: set[str] = set()
    for record in records:
        normalized = normalize_fever_record(record)
        page = normalize_fever_title(normalized.get("evidence_wiki_url"))
        if page:
            pages.add(page)
    return pages


def _label_relation(label: str) -> str:
    if label == "SUPPORTS":
        return "supported_by"
    if label == "REFUTES":
        return "refuted_by"
    return "evidence_for"


def _source_id(record: Mapping[str, Any], *, split: str) -> str:
    return f"fever_{split}_{int(record.get('id', -1)):06d}"


def fever_record_to_latent_set(
    record: Mapping[str, Any],
    *,
    split: str,
    wiki_sentence_index: Mapping[str, Mapping[int, str]] | None = None,
    include_reference_view: bool = False,
) -> tuple[LatentSetRow | None, List[str]]:
    record = normalize_fever_record(record)
    reasons: List[str] = []
    label = str(record.get("label", "")).strip().upper()
    if label not in SUPPORTED_LABELS:
        reasons.append(f"unsupported_label:{label or 'missing'}")

    claim = " ".join(str(record.get("claim", "")).split())
    if not claim:
        reasons.append("missing_claim")

    page = normalize_fever_title(record.get("evidence_wiki_url"))
    if not page:
        reasons.append("missing_evidence_wiki_url")

    try:
        claim_id = int(record.get("id"))
    except (TypeError, ValueError):
        claim_id = -1
        reasons.append("missing_or_invalid_id")

    try:
        evidence_sentence_id = int(record.get("evidence_sentence_id", -1))
    except (TypeError, ValueError):
        evidence_sentence_id = -1
    if evidence_sentence_id < 0:
        reasons.append("missing_evidence_sentence_id")

    if reasons:
        return None, reasons

    evidence_text = None
    if wiki_sentence_index is not None:
        evidence_text = evidence_text_for_record(record, wiki_sentence_index)

    evidence_source_id = f"wiki:{page}:{evidence_sentence_id}"
    evidence_missing = False
    if evidence_text is None:
        if not include_reference_view:
            return None, ["missing_evidence_text"]
        evidence_missing = True
        relation_word = "supports" if label == "SUPPORTS" else "refutes"
        evidence_text = f"Evidence sentence {evidence_sentence_id} on Wikipedia page {page} {relation_word} this claim."

    evidence_annotation_id = record.get("evidence_annotation_id", "unknown")
    set_id = f"fever_{split}_{claim_id:06d}_{evidence_annotation_id}"
    source_id = _source_id(record, split=split)

    row = LatentSetRow(
        set_id=set_id,
        latent_type="fact",
        latent_key=f"fever:{claim_id}:{label}:{page}:{evidence_sentence_id}",
        source_dataset="fever/fever",
        source_ids=[source_id, evidence_source_id],
        domain="wikipedia_fact_checking",
        texts=[
            LatentTextView(
                id="claim",
                text=claim,
                view_type="claim",
                source_id=source_id,
                metadata={"label": label},
            ),
            LatentTextView(
                id="evidence",
                text=evidence_text,
                view_type="evidence" if not evidence_missing else "evidence_reference",
                source_id=evidence_source_id,
                metadata={
                    "page": page,
                    "sentence_id": evidence_sentence_id,
                    "evidence_text_missing": evidence_missing,
                },
            ),
        ],
        positive_edges=[
            LatentEdge(
                source="claim",
                target="evidence",
                relation=_label_relation(label),
                evidence=evidence_source_id,
            )
        ],
        hard_negatives=[
            HardNegative(
                id="label_flip_control",
                text=f"A nearby hard negative should use the same topic but a different evidence relation for claim {claim_id}.",
                negative_type="placeholder_requires_mining",
                metadata={"needs_replacement": True},
            )
        ],
        validation={
            "source_annotation_ok": True,
            "evidence_text_present": not evidence_missing,
            "needs_hard_negative_mining": True,
        },
        metadata={
            "label": label,
            "evidence_wiki_url": page,
            "evidence_sentence_id": evidence_sentence_id,
            "evidence_annotation_id": evidence_annotation_id,
        },
    )
    issues = validate_latent_set(row)
    return (None, issues) if issues else (row, [])


def build_fever_latent_sets(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str,
    wiki_sentence_index: Mapping[str, Mapping[int, str]] | None = None,
    include_reference_view: bool = False,
    limit: int | None = None,
) -> FeverBuildResult:
    accepted: List[LatentSetRow] = []
    rejected: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        if limit is not None and len(accepted) >= limit:
            break
        row, reasons = fever_record_to_latent_set(
            record,
            split=split,
            wiki_sentence_index=wiki_sentence_index,
            include_reference_view=include_reference_view,
        )
        if row is None:
            rejected.append(
                {
                    "source_index": idx,
                    "source_id": record.get("id"),
                    "reasons": reasons,
                    "claim": record.get("claim"),
                    "label": record.get("label"),
                }
            )
        else:
            accepted.append(row)
    return FeverBuildResult(accepted=accepted, rejected=rejected)
