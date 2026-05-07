import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.activation_extraction import load_semantic_views
from setconca_v2.io_utils import write_jsonl
from setconca_v2.set_dataset import compute_set_stats, filter_sets_by_min_rewrites


def sample_rows():
    return [
        {
            "original_id": "a",
            "original_text": "Original A.",
            "label": "business",
            "source": "unit",
            "rewrites": [
                {"text": "A rewrite one.", "model_name": "m1", "length_band": "5-7", "word_count": 3},
                {"text": "A rewrite two.", "model_name": "m2", "length_band": "10-12", "word_count": 3},
            ],
        },
        {
            "original_id": "b",
            "original_text": "Original B.",
            "label": "sports",
            "source": "unit",
            "rewrites": [
                {"text": "B rewrite one.", "model_name": "m1", "length_band": "5-7", "word_count": 3},
            ],
        },
    ]


def test_compute_set_stats_counts_rewrites():
    path = ROOT / ".test_tmp" / "unit_test_sets.jsonl"
    write_jsonl(path, sample_rows())
    stats = compute_set_stats(path, thresholds=(1, 2))
    assert stats.n_sets == 2
    assert stats.n_rewrites == 3
    assert stats.sets_at_least == {"1": 2, "2": 1}
    assert stats.model_counts["m1"] == 2
    assert stats.length_band_counts["5-7"] == 2


def test_filter_sets_by_min_rewrites_keeps_large_sets():
    filtered = filter_sets_by_min_rewrites(sample_rows(), min_rewrites=2)
    assert [row["original_id"] for row in filtered] == ["a"]


def test_load_semantic_views_samples_original_plus_rewrites():
    path = ROOT / ".test_tmp" / "unit_test_sets_views.jsonl"
    write_jsonl(path, sample_rows())
    views = load_semantic_views(path, views=2, seed=123)
    assert len(views) == 2
    assert views[0].texts[0] == "Original A."
    assert len(views[0].texts) == 2
    assert views[0].rewrite_meta
