from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class DatasetRow:
    id: str
    text: str
    source: str
    label: str | None = None
    title: str | None = None


AG_NEWS_LABELS = {
    0: "world",
    1: "sports",
    2: "business",
    3: "science_technology",
}


def normalize_news_text(text: str, *, max_chars: int = 500) -> str:
    text = " ".join(str(text).replace("\\n", " ").split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def format_ag_news_rows(records: Iterable[Dict[str, Any]], *, split: str, limit: int | None = None) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    for idx, rec in enumerate(records):
        if limit is not None and len(rows) >= limit:
            break
        text = normalize_news_text(rec.get("text", ""))
        if not text:
            continue
        label_id = rec.get("label")
        label = AG_NEWS_LABELS.get(int(label_id), str(label_id)) if label_id is not None else None
        rows.append(
            DatasetRow(
                id=f"ag_news_{split}_{idx:06d}",
                text=text,
                source=f"hf:ag_news:{split}",
                label=label,
            )
        )
    return rows


def rows_to_jsonl_dicts(rows: Iterable[DatasetRow]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        item = {
            "id": row.id,
            "text": row.text,
            "source": row.source,
            "label": row.label,
        }
        if row.title is not None:
            item["title"] = row.title
        out.append(item)
    return out

