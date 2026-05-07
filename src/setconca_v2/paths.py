from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(raw: str | Path) -> Path:
    """Resolve CLI paths so scripts work from repo root or SetConCA_V2 root.

    If a user runs from inside SetConCA_V2 but still passes
    `SetConCA_V2/data/...`, strip the redundant leading project folder.
    """
    path = Path(raw)
    if path.is_absolute():
        return path

    root = project_root()
    parts = path.parts
    if parts and parts[0].lower() == root.name.lower():
        path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return root / path

