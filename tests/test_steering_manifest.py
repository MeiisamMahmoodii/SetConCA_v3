from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.steering import candidate_rows_from_review, keyword_score, model_dir_from_key


def test_keyword_score_counts_distinct_hits():
    score, hits = keyword_score("Microsoft released a Windows security update.", ["windows", "security", "ipo"])
    assert score == 2
    assert hits == ["windows", "security"]


def test_model_dir_from_key_reconstructs_transfer_path():
    path = model_dir_from_key(
        Path("results/run"),
        "setconca__llama3__mid_3b__meta-llama__llama-3.2-3b__layer_17_60pct__s16",
    )
    assert path.as_posix().endswith(
        "results/run/models/setconca/llama3/mid_3b/meta-llama__llama-3.2-3b/layer_17_60pct/s16"
    )


def test_candidate_rows_from_review_merges_ranked_labels():
    tmp_dir = ROOT / ".test_tmp" / f"steering_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    concept_summary = tmp_dir / "concept_summary.csv"
    labels = tmp_dir / "labels.csv"
    concept_summary.write_text(
        "source_key,target_key,source_family,target_family,source_size,target_size,source_layer,target_layer,"
        "bridge,pair_controlled_topk,target_concept_dim,source_concept_dim,inspection_score\n"
        "src,tgt,llama3,qwen3,small,big,layer_1,layer_2,ridge,0.2,7,3,1.5\n",
        encoding="utf-8",
    )
    labels.write_text(
        "rank,candidate,label,use_for_steering,notes\n"
        "1,Windows security,clean_semantic,yes,good\n",
        encoding="utf-8",
    )
    rows = candidate_rows_from_review(concept_summary_path=concept_summary, label_path=labels)
    assert len(rows) == 1
    assert rows[0]["target_concept_dim"] == 7
    assert "windows" in rows[0]["keywords"]
