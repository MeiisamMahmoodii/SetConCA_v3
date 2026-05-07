import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.dataset_download import format_ag_news_rows, normalize_news_text, rows_to_jsonl_dicts


def test_normalize_news_text_strips_whitespace_and_truncates():
    text = "  Hello\n\nworld   " + "x " * 100
    out = normalize_news_text(text, max_chars=20)
    assert "\n" not in out
    assert len(out) <= 20


def test_format_ag_news_rows_maps_labels():
    rows = format_ag_news_rows(
        [
            {"text": "Markets rose after a strong earnings report.", "label": 2},
            {"text": "The team won the final.", "label": 1},
        ],
        split="train",
    )
    assert rows[0].id == "ag_news_train_000000"
    assert rows[0].label == "business"
    assert rows[1].label == "sports"


def test_rows_to_jsonl_dicts_has_v2_schema():
    rows = format_ag_news_rows([{"text": "Scientists tested a new engine.", "label": 3}], split="test")
    item = rows_to_jsonl_dicts(rows)[0]
    assert set(item) == {"id", "text", "source", "label"}
    assert item["source"] == "hf:ag_news:test"

