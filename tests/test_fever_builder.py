import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.fever_builder import (
    build_fever_latent_sets,
    build_wiki_sentence_index,
    clean_fever_wiki_sentence,
    evidence_text_for_record,
    first_fever_evidence_item,
    normalize_fever_record,
    parse_fever_wiki_lines,
    required_fever_pages,
)
from scripts.build_latent_sets import FEVER_DIRECT_URLS, build_wiki_sentence_index_from_zip


def fever_record(label="SUPPORTS"):
    return {
        "id": 75397,
        "label": label,
        "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        "evidence_wiki_url": "Nikolaj_Coster-Waldau",
        "evidence_id": 104971,
        "evidence_sentence_id": 7,
        "evidence_annotation_id": 92206,
    }


def wiki_record():
    return {
        "id": "Nikolaj_Coster-Waldau",
        "lines": "0\tNikolaj Coster-Waldau is a Danish actor.\n7\tHe worked with the Fox Broadcasting Company.\tFox Broadcasting Company\tFox Broadcasting Company\n",
    }


def raw_fever_record():
    return {
        "id": 75397,
        "verifiable": "VERIFIABLE",
        "label": "SUPPORTS",
        "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
        "evidence": [[[92206, 104971, "Nikolaj_Coster-Waldau", 7]]],
    }


def test_parse_fever_wiki_lines_extracts_sentence_ids():
    parsed = parse_fever_wiki_lines(wiki_record()["lines"])
    assert parsed[0] == "Nikolaj Coster-Waldau is a Danish actor."
    assert parsed[7] == "He worked with the Fox Broadcasting Company."


def test_clean_fever_wiki_sentence_removes_entity_tail():
    line = "Soul Food is a 1997 American film .\tFox 2000 Pictures\tFox 2000 Pictures"
    assert clean_fever_wiki_sentence(line) == "Soul Food is a 1997 American film ."


def test_evidence_text_for_record_uses_page_and_sentence_id():
    index = build_wiki_sentence_index([wiki_record()])
    assert evidence_text_for_record(fever_record(), index) == "He worked with the Fox Broadcasting Company."


def test_normalize_fever_record_supports_original_nested_evidence_shape():
    item = first_fever_evidence_item(raw_fever_record())
    assert item == (92206, 104971, "Nikolaj_Coster-Waldau", 7)
    normalized = normalize_fever_record(raw_fever_record())
    assert normalized["evidence_wiki_url"] == "Nikolaj_Coster-Waldau"
    assert normalized["evidence_sentence_id"] == 7


def test_required_fever_pages_supports_nested_evidence_shape():
    assert required_fever_pages([raw_fever_record()]) == {"Nikolaj_Coster-Waldau"}


def test_build_fever_latent_sets_with_evidence_text():
    index = build_wiki_sentence_index([wiki_record()])
    result = build_fever_latent_sets([fever_record()], split="train", wiki_sentence_index=index)
    assert len(result.accepted) == 1
    assert result.rejected == []
    row = result.accepted[0]
    assert row.latent_type == "fact"
    assert row.texts[0].view_type == "claim"
    assert row.texts[1].view_type == "evidence"
    assert row.positive_edges[0].relation == "supported_by"
    assert row.validation["evidence_text_present"] is True


def test_build_fever_latent_sets_from_original_nested_evidence_shape():
    index = build_wiki_sentence_index([wiki_record()])
    result = build_fever_latent_sets([raw_fever_record()], split="labelled_dev", wiki_sentence_index=index)
    assert len(result.accepted) == 1
    row = result.accepted[0]
    assert row.source_ids[0] == "fever_labelled_dev_075397"
    assert row.texts[1].text == "He worked with the Fox Broadcasting Company."


def test_build_fever_latent_sets_rejects_missing_evidence_without_reference_mode():
    result = build_fever_latent_sets([fever_record()], split="train")
    assert result.accepted == []
    assert result.rejected[0]["reasons"] == ["missing_evidence_text"]


def test_build_fever_latent_sets_reference_mode_for_smoke_tests():
    result = build_fever_latent_sets([fever_record("REFUTES")], split="train", include_reference_view=True)
    assert len(result.accepted) == 1
    row = result.accepted[0]
    assert row.texts[1].view_type == "evidence_reference"
    assert row.positive_edges[0].relation == "refuted_by"
    assert row.validation["evidence_text_present"] is False


def test_build_latent_sets_cli_from_local_jsonl(tmp_path):
    input_path = tmp_path / "fever.jsonl"
    wiki_path = tmp_path / "wiki.jsonl"
    out_dir = tmp_path / "out"
    input_path.write_text(json.dumps(fever_record()) + "\n", encoding="utf-8")
    wiki_path.write_text(json.dumps(wiki_record()) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_latent_sets.py"),
            "fever",
            "--input",
            str(input_path),
            "--wiki-jsonl",
            str(wiki_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"accepted": 1' in proc.stdout
    rows = [json.loads(line) for line in (out_dir / "sets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["texts"][1]["text"] == "He worked with the Fox Broadcasting Company."
    assert (out_dir / "manifest.json").exists()


def test_build_wiki_sentence_index_from_zip_selects_required_pages(tmp_path):
    zip_path = tmp_path / "wiki-pages.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("wiki-pages/wiki-001.jsonl", json.dumps(wiki_record()) + "\n")
        zf.writestr("__MACOSX/wiki-pages/._wiki-001.jsonl", b"\x00\x05\x16\x07\x00\x02Mac OS X\x00\xb0")
        zf.writestr(
            "wiki-pages/wiki-002.jsonl",
            json.dumps({"id": "Unused", "lines": "0\tUnused sentence.\n"}) + "\n",
        )

    index = build_wiki_sentence_index_from_zip(zip_path, {"Nikolaj_Coster-Waldau"})
    assert set(index) == {"Nikolaj_Coster-Waldau"}
    assert index["Nikolaj_Coster-Waldau"][7] == "He worked with the Fox Broadcasting Company."


def test_fever_direct_urls_include_labelled_dev():
    assert FEVER_DIRECT_URLS["labelled_dev"].endswith("shared_task_dev.jsonl")
