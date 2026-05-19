import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.latent_dataset_summary import (
    jaccard,
    rejection_reason_counts,
    summarize_latent_rows,
    tokenize,
    write_summary_report,
)


def sample_rows():
    return [
        {
            "set_id": "fever_a",
            "latent_type": "fact",
            "latent_key": "a",
            "source_dataset": "fever/fever",
            "source_ids": ["fever_a", "wiki:a:0"],
            "usage": "train",
            "metadata": {"label": "SUPPORTS"},
            "texts": [
                {"id": "claim", "view_type": "claim", "text": "Fox released Soul Food."},
                {"id": "evidence", "view_type": "evidence", "text": "Soul Food was released by Fox."},
            ],
            "hard_negatives": [
                {
                    "id": "placeholder",
                    "negative_type": "placeholder_requires_mining",
                    "text": "placeholder",
                    "metadata": {"needs_replacement": True},
                }
            ],
        },
        {
            "set_id": "fever_b",
            "latent_type": "fact",
            "latent_key": "b",
            "source_dataset": "fever/fever",
            "source_ids": ["fever_b", "wiki:b:0"],
            "usage": "train",
            "metadata": {"label": "REFUTES"},
            "texts": [
                {"id": "claim", "view_type": "claim", "text": "Telemundo is English language."},
                {
                    "id": "evidence",
                    "view_type": "evidence_reference",
                    "text": "Evidence reference text.",
                },
            ],
        },
    ]


def test_tokenize_and_jaccard_are_case_insensitive():
    assert tokenize("Fox released FOX.") == ["fox", "released", "fox"]
    assert jaccard(["fox", "released"], ["fox", "film"]) == 1 / 3


def test_summarize_latent_rows_counts_core_metrics():
    summary = summarize_latent_rows(sample_rows())
    assert summary.n_sets == 2
    assert summary.latent_type_counts == {"fact": 2}
    assert summary.label_counts == {"REFUTES": 1, "SUPPORTS": 1}
    assert summary.view_type_counts == {"claim": 2, "evidence": 1, "evidence_reference": 1}
    assert summary.evidence_present_count == 1
    assert summary.evidence_missing_count == 1
    assert summary.placeholder_negative_count == 1
    assert summary.mean_views_per_set == 2
    assert summary.mean_claim_evidence_jaccard is not None


def test_rejection_reason_counts_flattens_reasons():
    counts = rejection_reason_counts(
        [
            {"reasons": ["missing_evidence_text", "unsupported_label:NOT ENOUGH INFO"]},
            {"reasons": ["missing_evidence_text"]},
        ]
    )
    assert counts["missing_evidence_text"] == 2
    assert counts["unsupported_label:NOT ENOUGH INFO"] == 1


def test_write_summary_report_includes_review_examples():
    summary = summarize_latent_rows(sample_rows())
    report = write_summary_report(summary=summary, rejection_counts={"missing": 1}, examples=sample_rows()[:1])
    assert "# Latent Dataset Summary" in report
    assert "fever_a" in report
    assert "Placeholder hard negatives" in report

