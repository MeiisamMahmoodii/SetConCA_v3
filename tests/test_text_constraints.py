import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.io_utils import group_accepted, write_review_table
from setconca_v2.semantic_validation import SemanticValidator
from setconca_v2.text_constraints import (  # noqa: E402
    LengthBand,
    clean_model_output,
    contains_banned_word,
    count_words,
    extract_banned_words,
    validate_rewrite,
)


def test_extract_banned_words_removes_stopwords_and_keeps_content():
    text = "Quarterly earnings rose after the chipmaker expanded cloud server sales in Asia."
    banned = extract_banned_words(text, max_words=5)
    assert "earnings" in banned
    assert "chipmaker" in banned
    assert "the" not in banned


def test_validate_rewrite_checks_length_and_banned_words():
    band = LengthBand("5-7", 5, 7)
    ok, reasons = validate_rewrite("Profits grew as overseas demand improved", ["profits"], band)
    assert not ok
    assert any("banned_words" in reason for reason in reasons)

    ok, reasons = validate_rewrite("Overseas demand lifted company income", ["profits"], band)
    assert ok
    assert reasons == []


def test_count_words_handles_punctuation():
    assert count_words("A fast, clear rewrite works well.") == 6


def test_clean_model_output_removes_common_prefixes():
    assert clean_model_output("Rewrite: Fresh demand lifted company income.") == "Fresh demand lifted company income."


def test_contains_banned_word_is_token_based():
    assert contains_banned_word("Clouds gathered at dusk.", ["cloud"]) == []
    assert contains_banned_word("Cloud demand increased.", ["cloud"]) == ["cloud"]


def test_group_accepted_collects_rewrites_by_original():
    rows = [
        {
            "original_id": "x",
            "original_text": "Original sentence.",
            "label": "demo",
            "source": "unit",
            "banned_words": ["original"],
            "rewrite": "Fresh wording appears here",
            "model_name": "m1",
            "model_id": "m1-id",
            "length_band": "5-7",
            "word_count": 4,
        }
    ]
    grouped = group_accepted(rows)
    assert len(grouped) == 1
    assert grouped[0]["rewrites"][0]["model_name"] == "m1"


def test_semantic_validator_disabled_passes():
    validator = SemanticValidator({"enabled": False})
    result = validator.validate("The market rose.", "Stocks climbed.")
    assert result.passed
    assert result.metrics["embedding_cosine"] is None


def test_write_review_table_contains_required_columns():
    rows = [
        {
            "original_id": "x",
            "original_text": "Original sentence.",
            "banned_words": ["original"],
            "rewrites": [
                {
                    "text": "Fresh wording appears here",
                    "model_name": "m1",
                    "model_id": "m1-id",
                    "length_band": "5-7",
                    "word_count": 4,
                }
            ],
        }
    ]
    out = ROOT / "data" / "generated" / "test_review_table.md"
    write_review_table(out, rows)
    text = out.read_text(encoding="utf-8")
    assert "Original sentence" in text
    assert "Banned words" in text
    assert "Fresh wording appears here" in text
