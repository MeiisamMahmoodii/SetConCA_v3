from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping

from .latent_set_schema import latent_set_from_dict


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class LatentDatasetSummary:
    n_sets: int
    latent_type_counts: Dict[str, int]
    source_dataset_counts: Dict[str, int]
    usage_counts: Dict[str, int]
    view_type_counts: Dict[str, int]
    label_counts: Dict[str, int]
    evidence_present_count: int
    evidence_missing_count: int
    placeholder_negative_count: int
    mean_views_per_set: float
    mean_claim_evidence_jaccard: float | None
    mean_claim_tokens: float | None
    mean_evidence_tokens: float | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_sets": self.n_sets,
            "latent_type_counts": self.latent_type_counts,
            "source_dataset_counts": self.source_dataset_counts,
            "usage_counts": self.usage_counts,
            "view_type_counts": self.view_type_counts,
            "label_counts": self.label_counts,
            "evidence_present_count": self.evidence_present_count,
            "evidence_missing_count": self.evidence_missing_count,
            "placeholder_negative_count": self.placeholder_negative_count,
            "mean_views_per_set": self.mean_views_per_set,
            "mean_claim_evidence_jaccard": self.mean_claim_evidence_jaccard,
            "mean_claim_tokens": self.mean_claim_tokens,
            "mean_evidence_tokens": self.mean_evidence_tokens,
        }


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _first_view(row: Mapping[str, Any], view_type: str) -> Mapping[str, Any] | None:
    for view in row.get("texts", []):
        if view.get("view_type") == view_type:
            return view
    return None


def summarize_latent_rows(rows: Iterable[Mapping[str, Any]]) -> LatentDatasetSummary:
    row_list = list(rows)
    latent_type_counts = Counter()
    source_dataset_counts = Counter()
    usage_counts = Counter()
    view_type_counts = Counter()
    label_counts = Counter()
    evidence_present_count = 0
    evidence_missing_count = 0
    placeholder_negative_count = 0
    views_per_set: List[int] = []
    claim_evidence_jaccards: List[float] = []
    claim_token_counts: List[int] = []
    evidence_token_counts: List[int] = []

    for raw in row_list:
        row = latent_set_from_dict(raw)
        latent_type_counts[row.latent_type] += 1
        source_dataset_counts[row.source_dataset] += 1
        usage_counts[row.usage] += 1
        views_per_set.append(len(row.texts))

        label = row.metadata.get("label")
        if label is not None:
            label_counts[str(label)] += 1

        for view in row.texts:
            view_type_counts[view.view_type] += 1
            if view.view_type == "evidence":
                if view.metadata.get("evidence_text_missing"):
                    evidence_missing_count += 1
                else:
                    evidence_present_count += 1
            elif view.view_type == "evidence_reference":
                evidence_missing_count += 1

        for neg in row.hard_negatives:
            if neg.metadata.get("needs_replacement") or neg.negative_type == "placeholder_requires_mining":
                placeholder_negative_count += 1

        claim = _first_view(raw, "claim")
        evidence = _first_view(raw, "evidence")
        if claim and evidence:
            claim_tokens = tokenize(str(claim.get("text", "")))
            evidence_tokens = tokenize(str(evidence.get("text", "")))
            claim_token_counts.append(len(claim_tokens))
            evidence_token_counts.append(len(evidence_tokens))
            claim_evidence_jaccards.append(jaccard(claim_tokens, evidence_tokens))

    return LatentDatasetSummary(
        n_sets=len(row_list),
        latent_type_counts=dict(sorted(latent_type_counts.items())),
        source_dataset_counts=dict(sorted(source_dataset_counts.items())),
        usage_counts=dict(sorted(usage_counts.items())),
        view_type_counts=dict(sorted(view_type_counts.items())),
        label_counts=dict(sorted(label_counts.items())),
        evidence_present_count=evidence_present_count,
        evidence_missing_count=evidence_missing_count,
        placeholder_negative_count=placeholder_negative_count,
        mean_views_per_set=mean(views_per_set) if views_per_set else 0.0,
        mean_claim_evidence_jaccard=mean(claim_evidence_jaccards) if claim_evidence_jaccards else None,
        mean_claim_tokens=mean(claim_token_counts) if claim_token_counts else None,
        mean_evidence_tokens=mean(evidence_token_counts) if evidence_token_counts else None,
    )


def rejection_reason_counts(rejected_rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for row in rejected_rows:
        for reason in row.get("reasons", []):
            counts[str(reason)] += 1
    return dict(counts.most_common())


def write_summary_report(
    *,
    summary: LatentDatasetSummary,
    rejection_counts: Dict[str, int],
    examples: List[Mapping[str, Any]],
) -> str:
    data = summary.to_dict()
    lines = [
        "# Latent Dataset Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Sets | {data['n_sets']} |",
        f"| Mean views per set | {data['mean_views_per_set']:.2f} |",
        f"| Evidence present views | {data['evidence_present_count']} |",
        f"| Evidence missing/reference views | {data['evidence_missing_count']} |",
        f"| Placeholder hard negatives | {data['placeholder_negative_count']} |",
    ]
    if data["mean_claim_evidence_jaccard"] is not None:
        lines.append(f"| Mean claim/evidence Jaccard | {data['mean_claim_evidence_jaccard']:.4f} |")
    if data["mean_claim_tokens"] is not None:
        lines.append(f"| Mean claim tokens | {data['mean_claim_tokens']:.2f} |")
    if data["mean_evidence_tokens"] is not None:
        lines.append(f"| Mean evidence tokens | {data['mean_evidence_tokens']:.2f} |")

    for title, counts in [
        ("Latent Type Counts", summary.latent_type_counts),
        ("Label Counts", summary.label_counts),
        ("View Type Counts", summary.view_type_counts),
        ("Usage Counts", summary.usage_counts),
        ("Rejection Reason Counts", rejection_counts),
    ]:
        lines.extend(["", f"## {title}", "", "| Item | Count |", "| --- | ---: |"])
        for key, count in counts.items():
            lines.append(f"| {str(key).replace('|', '\\|')} | {count} |")

    lines.extend(["", "## Review Examples", "", "| Set ID | Label | Claim | Evidence |", "| --- | --- | --- | --- |"])
    for raw in examples:
        claim = _first_view(raw, "claim") or {}
        evidence = _first_view(raw, "evidence") or _first_view(raw, "evidence_reference") or {}
        label = raw.get("metadata", {}).get("label", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(raw.get("set_id", "")).replace("|", "\\|"),
                    str(label).replace("|", "\\|"),
                    str(claim.get("text", "")).replace("|", "\\|"),
                    str(evidence.get("text", "")).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"

