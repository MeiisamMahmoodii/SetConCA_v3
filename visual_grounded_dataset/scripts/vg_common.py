from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def stable_hash(value: Any, *, length: int = 16) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def normalize_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[|,;]", raw) if part.strip()]


def render_prompt(language: dict[str, Any], view: dict[str, Any], template: str) -> str:
    return template.format(
        language_name=language["name"],
        language_code=language["code"],
        view_id=view["view_id"],
        view_instruction=view["instruction"],
    )


def load_enabled_models(path: str | Path | None = None) -> list[dict[str, Any]]:
    config = read_json(path or CONFIG_DIR / "models.json")
    return [model for model in config["models"] if model.get("enabled", True)]


def load_languages(path: str | Path | None = None) -> list[dict[str, Any]]:
    return read_json(path or CONFIG_DIR / "languages.json")["languages"]


def load_views(path: str | Path | None = None) -> tuple[list[dict[str, Any]], str]:
    config = read_json(path or CONFIG_DIR / "views.json")
    return config["views"], config["template"]


def refusal_like(text: str) -> bool:
    lowered = text.lower()
    bad_fragments = [
        "i cannot see",
        "i can't see",
        "cannot view",
        "can't view",
        "as an ai",
        "i do not have access",
        "no image provided",
        "unable to determine",
        "sorry",
    ]
    return any(fragment in lowered for fragment in bad_fragments)


def leaks_metadata(text: str) -> bool:
    lowered = text.lower()
    bad_fragments = [
        "dataset",
        "filename",
        "prompt",
        "instruction",
        "language_name",
        "view_instruction",
    ]
    return any(fragment in lowered for fragment in bad_fragments)


def script_matches(text: str, script_hint: str) -> bool:
    if script_hint == "arabic":
        return bool(re.search(r"[\u0600-\u06ff]", text))
    if script_hint == "devanagari":
        return bool(re.search(r"[\u0900-\u097f]", text))
    if script_hint == "cjk":
        return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))
    if script_hint == "hangul":
        return bool(re.search(r"[\uac00-\ud7af]", text))
    if script_hint == "latin":
        latin_chars = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text))
        letters = len(re.findall(r"[^\W\d_]", text, flags=re.UNICODE))
        return letters == 0 or latin_chars / max(letters, 1) >= 0.75
    return True


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

