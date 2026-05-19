import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.dataset_registry import filter_sources, load_dataset_registry
from setconca_v2.latent_set_schema import latent_set_from_dict, validate_latent_set_dict


def valid_latent_row():
    return {
        "set_id": "fever_fact_000001",
        "latent_type": "fact",
        "latent_key": "claim:000001",
        "source_dataset": "fever/fever",
        "source_ids": ["fever_train_000001"],
        "domain": "wikipedia_fact_checking",
        "texts": [
            {
                "id": "claim",
                "view_type": "claim",
                "text": "The Eiffel Tower is located in Paris.",
                "source_id": "fever_train_000001",
            },
            {
                "id": "evidence",
                "view_type": "evidence",
                "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
                "source_id": "wiki:eiffel_tower:0",
            },
        ],
        "positive_edges": [
            {"source": "claim", "target": "evidence", "relation": "supported_by"},
        ],
        "hard_negatives": [
            {
                "id": "topic_negative",
                "negative_type": "topic_negative",
                "text": "The Eiffel Tower is located in Berlin.",
            }
        ],
    }


def test_latent_set_from_dict_round_trips_required_shape():
    row = latent_set_from_dict(valid_latent_row())
    assert row.set_id == "fever_fact_000001"
    assert row.latent_type == "fact"
    assert row.source_ids == ["fever_train_000001"]
    assert {view.view_type for view in row.texts} == {"claim", "evidence"}
    assert row.to_dict()["positive_edges"][0]["relation"] == "supported_by"


def test_validate_latent_set_accepts_valid_row():
    assert validate_latent_set_dict(valid_latent_row()) == []


def test_validate_latent_set_requires_multiple_view_types():
    row = valid_latent_row()
    row["texts"][1]["view_type"] = "claim"
    issues = validate_latent_set_dict(row)
    assert "too_few_view_types<2" in issues


def test_validate_latent_set_rejects_unknown_edge_reference():
    row = valid_latent_row()
    row["positive_edges"] = [{"source": "claim", "target": "missing", "relation": "supported_by"}]
    issues = validate_latent_set_dict(row)
    assert "edge_unknown_target:missing" in issues


def test_validate_latent_set_flags_risk_tagged_training_rows():
    row = valid_latent_row()
    row["risk_tag"] = "sensitive_bias"
    row["usage"] = "train"
    issues = validate_latent_set_dict(row)
    assert "risk_tagged_row_marked_train" in issues


def test_dataset_registry_loads_build_now_sources():
    sources = load_dataset_registry(ROOT / "configs" / "dataset_sources.json")
    build_now = filter_sources(sources, status="build_now")
    assert {source.source_id for source in build_now} >= {"fever", "hotpotqa", "allnli", "paws"}
    assert all(source.primary_use for source in sources)


def test_dataset_registry_keeps_truthfulqa_eval_only():
    sources = load_dataset_registry(ROOT / "configs" / "dataset_sources.json")
    truthfulqa = [source for source in sources if source.source_id == "truthfulqa"][0]
    assert truthfulqa.status == "optional_eval"
    assert truthfulqa.primary_use == "intervention_benchmark"

