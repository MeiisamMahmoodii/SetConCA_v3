from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class ActivationSetBank:
    hidden: torch.Tensor
    texts: list[str]
    meta: dict[str, Any]

    def capped(self, n: int | None = None, views: int | None = None) -> "ActivationSetBank":
        h = self.hidden
        if n is not None:
            h = h[:n]
        if views is not None:
            h = h[:, :views, :]
        return ActivationSetBank(h.contiguous(), self.texts[: len(h)], dict(self.meta))

    def train_test(self, train_frac: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
        n_train = int(len(self.hidden) * train_frac)
        return self.hidden[:n_train].contiguous(), self.hidden[n_train:].contiguous()


def load_activation_bank(path: str | Path) -> ActivationSetBank:
    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "hidden" not in obj:
        raise ValueError(f"{path} must contain a dict with key 'hidden'")
    hidden = obj["hidden"].float()
    texts = obj.get("texts", [])
    if not isinstance(texts, list):
        texts = []
    meta = obj.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta = {**meta, "path": str(path), "shape": list(hidden.shape)}
    return ActivationSetBank(hidden=hidden, texts=texts, meta=meta)

