from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from data.activation_sets import ActivationSetBank, load_activation_bank
from model.setconca_v2 import SetConCAV2
from setconca_v2.paths import resolve_project_path
from training.losses import LossWeights, compute_v2_loss, offdiag_decorrelation


DEPTH_RE = re.compile(r"_(20|60|90)pct")


@dataclass(frozen=True)
class BankSpec:
    bank_id: str
    family: str
    size: str
    size_label: str
    model_slug: str
    layer_name: str
    path: Path
    model_id: str
    layer: int
    hidden_dim: int
    n_sets: int
    views: int


@dataclass(frozen=True)
class TrainSpec:
    bank: BankSpec
    set_size: int
    out_dir: Path


class PointwiseTopKModel(torch.nn.Module):
    """Pointwise sparse baseline: train on individual views, pool codes only at evaluation time."""

    def __init__(self, hidden_dim: int, concept_dim: int, topk: int, dropout: float = 0.0, topk_abs: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.concept_dim = concept_dim
        self.topk = min(topk, concept_dim)
        self.topk_abs = topk_abs
        self.encoder = torch.nn.Linear(hidden_dim, concept_dim)
        self.norm = torch.nn.LayerNorm(concept_dim, elementwise_affine=False)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.decoder = torch.nn.Linear(concept_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)

    def topk_sparse(self, z_dense: torch.Tensor) -> torch.Tensor:
        score = z_dense.abs() if self.topk_abs else z_dense
        _, idx = torch.topk(score, self.topk, dim=-1)
        mask = torch.zeros_like(z_dense)
        mask.scatter_(-1, idx, 1.0)
        return z_dense * mask

    def encode_points(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_dense = self.dropout(self.norm(self.encoder(x)))
        z = self.topk_sparse(z_dense)
        return z, z_dense

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, z_dense = self.encode_points(x)
        return self.decoder(z), z, z_dense

    @torch.no_grad()
    def encode_set(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape
        _, dense_points = self.encode_points(x.reshape(b * s, d))
        dense_set = dense_points.reshape(b, s, -1).mean(dim=1)
        z_set = self.topk_sparse(dense_set)
        return z_set, dense_set


def stable_id(text: str) -> str:
    return (
        text.lower()
        .replace("\\", "/")
        .replace("/", "__")
        .replace(" ", "_")
        .replace(":", "")
    )


def depth_pct_from_name(text: str) -> str:
    match = DEPTH_RE.search(text)
    return match.group(1) if match else "unknown"


def parse_bank(path: Path, root: Path) -> BankSpec:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hidden = payload["hidden"]
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) < 5:
        raise ValueError(f"Activation bank path is too shallow: {path}")
    family = parts[0]
    size_part = parts[1]
    if "_" in size_part:
        size, size_label = size_part.split("_", 1)
    else:
        size, size_label = size_part, size_part
    model_slug = parts[2]
    layer_name = parts[3]
    model_id = str(meta.get("model_id", model_slug))
    layer = int(meta.get("layer", -1))
    bank_id = stable_id("/".join([family, size_part, model_slug, layer_name]))
    return BankSpec(
        bank_id=bank_id,
        family=family,
        size=size,
        size_label=size_label,
        model_slug=model_slug,
        layer_name=layer_name,
        path=path,
        model_id=model_id,
        layer=layer,
        hidden_dim=int(hidden.shape[-1]),
        n_sets=int(hidden.shape[0]),
        views=int(hidden.shape[1]),
    )


def discover_banks(root: Path) -> list[BankSpec]:
    banks = []
    for path in sorted(root.rglob("activation_bank.pt")):
        banks.append(parse_bank(path, root))
    return banks


def filter_banks(
    banks: list[BankSpec],
    *,
    only_family: str | None,
    exclude_family: str | None,
    only_size: str | None,
    only_layer_pct: str | None,
    max_banks: int | None,
) -> list[BankSpec]:
    out = banks
    if only_family:
        allowed = {x.strip() for x in only_family.split(",") if x.strip()}
        out = [b for b in out if b.family in allowed]
    if exclude_family:
        blocked = {x.strip() for x in exclude_family.split(",") if x.strip()}
        out = [b for b in out if b.family not in blocked]
    if only_size:
        out = [b for b in out if b.size == only_size or b.size_label == only_size]
    if only_layer_pct:
        out = [b for b in out if b.layer_name.endswith(f"_{only_layer_pct}pct")]
    if max_banks is not None:
        out = out[:max_banks]
    return out


def batch_indices(n: int, batch_size: int, *, shuffle: bool, device: torch.device) -> list[torch.Tensor]:
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    return [idx[start : start + batch_size] for start in range(0, n, batch_size)]


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def slice_bank(bank: ActivationSetBank, *, set_size: int, max_sets: int | None) -> ActivationSetBank:
    if set_size > bank.hidden.shape[1]:
        raise ValueError(f"Requested S={set_size}, but bank only has {bank.hidden.shape[1]} views")
    return bank.capped(n=max_sets, views=set_size)


def train_model(
    spec: TrainSpec,
    args: argparse.Namespace,
    device: torch.device,
    weights: LossWeights,
) -> tuple[SetConCAV2, dict[str, Any]]:
    t0 = time.time()
    bank = slice_bank(load_activation_bank(spec.bank.path), set_size=spec.set_size, max_sets=args.max_sets)
    hidden = bank.hidden.to(device)
    train_x, test_x = bank.__class__(hidden, bank.texts[: len(hidden)], bank.meta).train_test(args.train_frac)

    model = SetConCAV2(
        hidden_dim=hidden.shape[-1],
        concept_dim=args.concept_dim,
        topk=args.topk,
        residual_scale=args.residual_scale,
        residual_rank=args.residual_rank,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        rows: list[dict[str, float]] = []
        for idx in batch_indices(len(train_x), args.batch_size, shuffle=True, device=device):
            xb = train_x[idx]
            neg_idx = torch.randperm(len(train_x), device=device)[: len(idx)]
            xneg = train_x[neg_idx]
            loss, parts = compute_v2_loss(model, xb, xneg, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            rows.append({key: float(value.detach().cpu()) for key, value in parts.items()})
        train_metrics = average(rows)
        test_metrics = evaluate_model(model, test_x, weights, args.batch_size)
        history.append({"epoch": epoch, "train": train_metrics, "test": test_metrics})
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(
                f"train {spec.bank.bank_id} S={spec.set_size} epoch={epoch} "
                f"train_total={train_metrics.get('total', 0):.6f} "
                f"test_total={test_metrics.get('total', 0):.6f}",
                flush=True,
            )

    spec.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = spec.out_dir / "model.pt"
    metrics_path = spec.out_dir / "metrics.json"
    manifest = {
        "bank": asdict(spec.bank) | {"path": str(spec.bank.path)},
        "set_size": spec.set_size,
        "hidden_shape": list(hidden.shape),
        "train_shape": list(train_x.shape),
        "test_shape": list(test_x.shape),
        "concept_dim": args.concept_dim,
        "topk": args.topk,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "elapsed_s": time.time() - t0,
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
    }
    torch.save({"model_state_dict": model.state_dict(), "manifest": manifest}, checkpoint_path)
    metrics_path.write_text(json.dumps({"manifest": manifest, "history": history}, indent=2), encoding="utf-8")
    return model, {"manifest": manifest, "history": history}


def train_pointwise_model(
    spec: TrainSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PointwiseTopKModel, dict[str, Any]]:
    t0 = time.time()
    bank = slice_bank(load_activation_bank(spec.bank.path), set_size=spec.set_size, max_sets=args.max_sets)
    hidden = bank.hidden.to(device)
    train_x, test_x = bank.__class__(hidden, bank.texts[: len(hidden)], bank.meta).train_test(args.train_frac)
    train_points = train_x.reshape(-1, hidden.shape[-1])
    test_points = test_x.reshape(-1, hidden.shape[-1])

    model = PointwiseTopKModel(
        hidden_dim=hidden.shape[-1],
        concept_dim=args.concept_dim,
        topk=args.topk,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        rows: list[dict[str, float]] = []
        for idx in batch_indices(len(train_points), args.batch_size, shuffle=True, device=device):
            xb = train_points[idx]
            recon, z, _ = model(xb)
            recon_loss = ((recon - xb) ** 2).mean()
            offdiag = offdiag_decorrelation(z)
            total = recon_loss + args.offdiag_weight * offdiag
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            rows.append(
                {
                    "total": float(total.detach().cpu()),
                    "point_recon": float(recon_loss.detach().cpu()),
                    "offdiag": float(offdiag.detach().cpu()),
                }
            )
        train_metrics = average(rows)
        test_metrics = evaluate_pointwise_model(model, test_points, args.batch_size, args.offdiag_weight)
        history.append({"epoch": epoch, "train": train_metrics, "test": test_metrics})
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(
                f"train pointwise {spec.bank.bank_id} S={spec.set_size} epoch={epoch} "
                f"train_total={train_metrics.get('total', 0):.6f} "
                f"test_total={test_metrics.get('total', 0):.6f}",
                flush=True,
            )

    spec.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = spec.out_dir / "model.pt"
    metrics_path = spec.out_dir / "metrics.json"
    manifest = {
        "method": "pointwise_topk",
        "bank": asdict(spec.bank) | {"path": str(spec.bank.path)},
        "set_size": spec.set_size,
        "hidden_shape": list(hidden.shape),
        "train_shape": list(train_x.shape),
        "test_shape": list(test_x.shape),
        "concept_dim": args.concept_dim,
        "topk": args.topk,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "elapsed_s": time.time() - t0,
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
    }
    torch.save({"model_state_dict": model.state_dict(), "manifest": manifest}, checkpoint_path)
    metrics_path.write_text(json.dumps({"manifest": manifest, "history": history}, indent=2), encoding="utf-8")
    return model, {"manifest": manifest, "history": history}


def evaluate_model(
    model: SetConCAV2,
    x: torch.Tensor,
    weights: LossWeights,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for idx in batch_indices(len(x), batch_size, shuffle=False, device=x.device):
            xb = x[idx]
            _, parts = compute_v2_loss(model, xb, None, weights)
            item = {key: float(value.detach().cpu()) for key, value in parts.items()}
            input_energy = float(xb.pow(2).mean().detach().cpu())
            item["input_energy"] = input_energy
            if input_energy > 0:
                item["shared_recon_norm"] = item.get("shared_recon", 0.0) / input_energy
                item["full_recon_norm"] = item.get("full_recon", 0.0) / input_energy
            else:
                item["shared_recon_norm"] = float("nan")
                item["full_recon_norm"] = float("nan")
            rows.append(item)
    return average(rows)


def evaluate_pointwise_model(
    model: PointwiseTopKModel,
    x: torch.Tensor,
    batch_size: int,
    offdiag_weight: float,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for idx in batch_indices(len(x), batch_size, shuffle=False, device=x.device):
            xb = x[idx]
            recon, z, _ = model(xb)
            recon_loss = ((recon - xb) ** 2).mean()
            offdiag = offdiag_decorrelation(z)
            input_energy = float(xb.pow(2).mean().detach().cpu())
            item = {
                "total": float((recon_loss + offdiag_weight * offdiag).detach().cpu()),
                "point_recon": float(recon_loss.detach().cpu()),
                "offdiag": float(offdiag.detach().cpu()),
                "input_energy": input_energy,
            }
            item["point_recon_norm"] = item["point_recon"] / input_energy if input_energy > 0 else float("nan")
            rows.append(item)
    return average(rows)


@torch.no_grad()
def extract_codes(
    model: SetConCAV2,
    bank: ActivationSetBank,
    *,
    train_frac: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    hidden = bank.hidden.to(device)
    n_train = int(len(hidden) * train_frac)
    model.eval()
    codes = []
    dense_codes = []
    for idx in batch_indices(len(hidden), batch_size, shuffle=False, device=device):
        out = model(hidden[idx])
        codes.append(out.z.detach().cpu())
        dense_codes.append(out.z_dense.detach().cpu())
    z = torch.cat(codes, dim=0)
    z_dense = torch.cat(dense_codes, dim=0)
    return {
        "train_z": z[:n_train].contiguous(),
        "test_z": z[n_train:].contiguous(),
        "train_z_dense": z_dense[:n_train].contiguous(),
        "test_z_dense": z_dense[n_train:].contiguous(),
    }


@torch.no_grad()
def extract_pointwise_codes(
    model: PointwiseTopKModel,
    bank: ActivationSetBank,
    *,
    train_frac: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    hidden = bank.hidden.to(device)
    n_train = int(len(hidden) * train_frac)
    model.eval()
    codes = []
    dense_codes = []
    for idx in batch_indices(len(hidden), batch_size, shuffle=False, device=device):
        z, z_dense = model.encode_set(hidden[idx])
        codes.append(z.detach().cpu())
        dense_codes.append(z_dense.detach().cpu())
    z = torch.cat(codes, dim=0)
    z_dense = torch.cat(dense_codes, dim=0)
    return {
        "train_z": z[:n_train].contiguous(),
        "test_z": z[n_train:].contiguous(),
        "train_z_dense": z_dense[:n_train].contiguous(),
        "test_z_dense": z_dense[n_train:].contiguous(),
    }


def fit_bridge(x: torch.Tensor, y: torch.Tensor, method: str, *, ridge_alpha: float, mlp_epochs: int) -> dict[str, Any]:
    x = x.float()
    y = y.float()
    if method == "identity":
        return {"method": method, "matrix": torch.eye(x.shape[1])}
    if method == "procrustes":
        u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
        return {"method": method, "matrix": u @ vh}
    if method == "ridge":
        eye = torch.eye(x.shape[1])
        matrix = torch.linalg.solve(x.T @ x + ridge_alpha * eye, x.T @ y)
        return {"method": method, "matrix": matrix}
    if method == "mlp":
        return fit_mlp_bridge(x, y, epochs=mlp_epochs)
    raise ValueError(f"Unsupported bridge method: {method}")


def apply_bridge(x: torch.Tensor, bridge: dict[str, Any]) -> torch.Tensor:
    if bridge["method"] in {"identity", "procrustes", "ridge"}:
        return x @ bridge["matrix"]
    if bridge["method"] == "mlp":
        model = bridge["model"]
        model.eval()
        with torch.no_grad():
            return model(x.float()).cpu()
    raise ValueError(f"Unsupported bridge method: {bridge['method']}")


def fit_mlp_bridge(x: torch.Tensor, y: torch.Tensor, *, epochs: int) -> dict[str, Any]:
    hidden = max(32, min(512, x.shape[1] * 2))
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, y.shape[1]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {"method": "mlp", "model": model}


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, a.shape[1], b.shape[1])
    ia = torch.topk(a.abs(), k, dim=-1).indices
    ib = torch.topk(b.abs(), k, dim=-1).indices
    overlaps = []
    for row_a, row_b in zip(ia, ib):
        overlaps.append(len(set(row_a.tolist()) & set(row_b.tolist())) / k)
    return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1).mean())


def steering_proxy(
    source_test: torch.Tensor,
    target_test: torch.Tensor,
    mapped_source_test: torch.Tensor,
    *,
    alphas: Iterable[float],
) -> dict[str, Any]:
    target_n = F.normalize(target_test.float(), dim=-1)
    direction = F.normalize(mapped_source_test.float(), dim=-1)
    perm = torch.randperm(len(direction))
    random_direction = direction[perm]
    rows = []
    for alpha in alphas:
        structured = F.normalize(target_n + float(alpha) * direction, dim=-1)
        random = F.normalize(target_n + float(alpha) * random_direction, dim=-1)
        rows.append(
            {
                "alpha": float(alpha),
                "structured_similarity": cosine_mean(structured, direction),
                "random_similarity": cosine_mean(random, direction),
            }
        )
    return {"rows": rows}


def evaluate_pairs(
    trained: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    keys = sorted(trained)
    bridge_methods = [x.strip() for x in args.bridges.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.steering_alphas.split(",") if x.strip()]
    for src_key in keys:
        for tgt_key in keys:
            if src_key == tgt_key and not args.include_self_pairs:
                continue
            src = trained[src_key]
            tgt = trained[tgt_key]
            if src["method"] != tgt["method"] and not args.include_cross_method_pairs:
                continue
            if src["set_size"] != tgt["set_size"]:
                continue
            n_train = min(len(src["codes"]["train_z"]), len(tgt["codes"]["train_z"]))
            n_test = min(len(src["codes"]["test_z"]), len(tgt["codes"]["test_z"]))
            if n_train < 2 or n_test < 1:
                continue
            x_train = src["codes"]["train_z"][:n_train]
            y_train = tgt["codes"]["train_z"][:n_train]
            x_test = src["codes"]["test_z"][:n_test]
            y_test = tgt["codes"]["test_z"][:n_test]
            for method in bridge_methods:
                if method == "identity" and x_train.shape[1] != y_train.shape[1]:
                    continue
                bridge = fit_bridge(
                    x_train,
                    y_train,
                    method,
                    ridge_alpha=args.ridge_alpha,
                    mlp_epochs=args.mlp_epochs,
                )
                mapped_train = apply_bridge(x_train, bridge)
                mapped = apply_bridge(x_test, bridge)
                mse = float(F.mse_loss(mapped, y_test.float()))
                train_overlap = topk_overlap(mapped_train, y_train, args.topk)
                overlap = topk_overlap(mapped, y_test, args.topk)
                shuffled_overlap = topk_overlap(mapped, y_test[torch.randperm(len(y_test))], args.topk)
                cos = cosine_mean(mapped, y_test)
                steering = steering_proxy(x_test, y_test, mapped, alphas=alphas)
                for steer_row in steering["rows"]:
                    rows.append(
                        {
                            "source_key": src_key,
                            "target_key": tgt_key,
                            "method": src["method"],
                            "source_method": src["method"],
                            "target_method": tgt["method"],
                            "source_family": src["bank"].family,
                            "target_family": tgt["bank"].family,
                            "source_size": src["bank"].size_label,
                            "target_size": tgt["bank"].size_label,
                            "source_layer": src["bank"].layer,
                            "target_layer": tgt["bank"].layer,
                            "source_depth_pct": depth_pct_from_name(src["bank"].layer_name),
                            "target_depth_pct": depth_pct_from_name(tgt["bank"].layer_name),
                            "relation": "within_family" if src["bank"].family == tgt["bank"].family else "cross_family",
                            "set_size": src["set_size"],
                            "bridge": method,
                            "transfer_mse": mse,
                            "train_topk_overlap": train_overlap,
                            "topk_overlap": overlap,
                            "train_test_topk_gap": train_overlap - overlap,
                            "shuffled_topk_overlap": shuffled_overlap,
                            "real_minus_shuffled_topk": overlap - shuffled_overlap,
                            "cosine": cos,
                            "steering_alpha": steer_row["alpha"],
                            "steering_structured_similarity": steer_row["structured_similarity"],
                            "steering_random_similarity": steer_row["random_similarity"],
                            "steering_gain_vs_random": steer_row["structured_similarity"] - steer_row["random_similarity"],
                        }
                    )
    write_rows(out_dir / "transfer_steering_results.csv", rows)
    (out_dir / "transfer_steering_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_training_summary(path: Path, trained: dict[str, dict[str, Any]]) -> None:
    rows = []
    for key, item in sorted(trained.items()):
        history = item["training"]["history"]
        final = history[-1] if history else {"train": {}, "test": {}}
        rows.append(
            {
                "key": key,
                "method": item["method"],
                "family": item["bank"].family,
                "size": item["bank"].size_label,
                "model_id": item["bank"].model_id,
                "layer": item["bank"].layer,
                "set_size": item["set_size"],
                "train_total": final["train"].get("total", math.nan),
                "test_total": final["test"].get("total", math.nan),
                "test_shared_recon": final["test"].get("shared_recon", math.nan),
                "test_full_recon": final["test"].get("full_recon", math.nan),
                "test_input_energy": final["test"].get("input_energy", math.nan),
                "test_shared_recon_norm": final["test"].get("shared_recon_norm", math.nan),
                "test_full_recon_norm": final["test"].get("full_recon_norm", math.nan),
                "test_point_recon": final["test"].get("point_recon", math.nan),
                "test_point_recon_norm": final["test"].get("point_recon_norm", math.nan),
            }
        )
    write_rows(path, rows)


def support_entropy(freq: torch.Tensor) -> float:
    total = freq.sum()
    if float(total) <= 0:
        return 0.0
    p = freq / total
    p = p[p > 0]
    return float(-(p * p.log()).sum() / math.log(len(freq)))


def code_diagnostics_for_item(key: str, item: dict[str, Any]) -> dict[str, Any]:
    z = torch.cat([item["codes"]["train_z"], item["codes"]["test_z"]], dim=0)
    z_dense = torch.cat([item["codes"]["train_z_dense"], item["codes"]["test_z_dense"]], dim=0)
    active = z.ne(0)
    freq = active.float().mean(dim=0)
    return {
        "key": key,
        "method": item["method"],
        "family": item["bank"].family,
        "size": item["bank"].size_label,
        "model_id": item["bank"].model_id,
        "layer": item["bank"].layer,
        "set_size": item["set_size"],
        "n_codes": int(z.shape[0]),
        "concept_dim": int(z.shape[1]),
        "mean_active_count": float(active.sum(dim=1).float().mean()),
        "concepts_ever_active": int(freq.gt(0).sum()),
        "frac_concepts_ever_active": float(freq.gt(0).float().mean()),
        "mean_active_frequency": float(freq.mean()),
        "max_active_frequency": float(freq.max()),
        "support_entropy": support_entropy(freq),
        "mean_code_norm": float(z.norm(dim=-1).mean()),
        "mean_dense_code_norm": float(z_dense.norm(dim=-1).mean()),
    }


def write_code_diagnostics(path: Path, trained: dict[str, dict[str, Any]]) -> None:
    rows = [code_diagnostics_for_item(key, item) for key, item in sorted(trained.items())]
    write_rows(path, rows)


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key, "") not in {"", None}]
    return float(sum(vals) / len(vals)) if vals else math.nan


def summarize_group(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(field, "") for field in group_fields), []).append(row)
    out = []
    for key in sorted(grouped):
        vals = grouped[key]
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
                "n": len(vals),
                "raw_topk": mean_float(vals, "topk_overlap"),
                "shuffled_topk": mean_float(vals, "shuffled_topk_overlap"),
                "real_minus_shuffled_topk": mean_float(vals, "real_minus_shuffled_topk"),
                "train_test_topk_gap": mean_float(vals, "train_test_topk_gap"),
                "transfer_mse": mean_float(vals, "transfer_mse"),
                "cosine": mean_float(vals, "cosine"),
            }
        )
        out.append(item)
    return out


def write_summary_artifacts(run_dir: Path, results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    alpha0 = [row for row in results if float(row["steering_alpha"]) == 0.0]
    summaries = {
        "bridge_method_summary": summarize_group(alpha0, ["method", "bridge"]),
        "method_relation_summary": summarize_group(alpha0, ["method", "relation", "bridge"]),
        "family_pair_summary": summarize_group(alpha0, ["method", "source_family", "target_family", "bridge"]),
        "depth_pair_summary": summarize_group(alpha0, ["method", "source_depth_pct", "target_depth_pct", "bridge"]),
        "set_size_summary": summarize_group(alpha0, ["method", "set_size", "bridge"]),
        "family_depth_pair_summary": summarize_group(
            alpha0,
            ["method", "source_family", "target_family", "source_depth_pct", "target_depth_pct", "bridge"],
        ),
    }
    summary_dir = run_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in summaries.items():
        write_rows(summary_dir / f"{name}.csv", rows)
        (summary_dir / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return summaries


def relation_for_row(row: dict[str, Any]) -> str:
    return "within_family" if row["source_family"] == row["target_family"] else "cross_family"


def pretty_method(method: str) -> str:
    return {
        "setconca": "SetConCA",
        "pointwise_topk": "Pointwise TopK",
    }.get(method, method.replace("_", " ").title())


def pretty_bridge(bridge: str) -> str:
    return {
        "identity": "Identity",
        "procrustes": "Procrustes",
        "ridge": "Ridge",
        "mlp": "MLP",
    }.get(bridge, bridge.replace("_", " ").title())


def pretty_relation(relation: str) -> str:
    return {
        "within_family": "Within family",
        "cross_family": "Cross family",
    }.get(relation, relation.replace("_", " ").title())


def pretty_family(family: str) -> str:
    return {
        "llama3": "Llama 3",
        "qwen3": "Qwen 3",
        "gemma3": "Gemma 3",
    }.get(family, family.replace("_", " ").title())


def controlled_label() -> str:
    return "Controlled TopK overlap\n(real minus shuffled)"


def add_bar_labels(ax: Any, bars: Any, *, y_pad: float = 0.004, fontsize: int = 9) -> None:
    for bar in bars:
        height = float(bar.get_height())
        va = "bottom" if height >= 0 else "top"
        offset = y_pad if height >= 0 else -y_pad
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.3f}",
            ha="center",
            va=va,
            fontsize=fontsize,
        )


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_results(run_dir: Path, trained: dict[str, dict[str, Any]], results: list[dict[str, Any]]) -> None:
    if not results:
        return
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    filtered = [r for r in results if r["steering_alpha"] == 0.0]
    bridge_scores: dict[str, list[float]] = {}
    for row in filtered:
        label = f"{row.get('method', 'setconca')}:{row['bridge']}"
        bridge_scores.setdefault(label, []).append(float(row["topk_overlap"]))
    labels = sorted(bridge_scores)
    means = [sum(bridge_scores[label]) / len(bridge_scores[label]) for label in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, means)
    plt.ylabel("Mean TopK overlap")
    plt.xlabel("Bridge")
    plt.title("Concept transfer by bridge")
    plt.tight_layout()
    plt.savefig(fig_dir / "bridge_topk_overlap.png", dpi=180)
    plt.close()

    alpha_scores: dict[float, list[float]] = {}
    for row in results:
        alpha_scores.setdefault(float(row["steering_alpha"]), []).append(float(row["steering_gain_vs_random"]))
    xs = sorted(alpha_scores)
    ys = [sum(alpha_scores[x]) / len(alpha_scores[x]) for x in xs]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Structured minus random similarity")
    plt.xlabel("Steering alpha")
    plt.title("Steering proxy gain")
    plt.tight_layout()
    plt.savefig(fig_dir / "steering_proxy_gain.png", dpi=180)
    plt.close()

    set_bridge: dict[tuple[int, str], list[float]] = {}
    for row in filtered:
        key = (int(row["set_size"]), f"{row.get('method', 'setconca')}:{row['bridge']}")
        set_bridge.setdefault(key, []).append(float(row["topk_overlap"]))
    set_sizes = sorted({key[0] for key in set_bridge})
    bridges = sorted({key[1] for key in set_bridge})
    plt.figure(figsize=(8, 4.5))
    for bridge in bridges:
        ys = []
        for set_size in set_sizes:
            vals = set_bridge.get((set_size, bridge), [])
            ys.append(sum(vals) / len(vals) if vals else float("nan"))
        plt.plot(set_sizes, ys, marker="o", label=bridge)
    plt.ylabel("Mean TopK overlap")
    plt.xlabel("Set size")
    plt.title("Concept transfer by set size and bridge")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "set_size_bridge_topk_overlap.png", dpi=180)
    plt.close()

    layer_rows = []
    for item in trained.values():
        history = item["training"]["history"]
        if not history:
            continue
        final = history[-1]
        layer_rows.append(
            (
                int(item["bank"].layer),
                int(item["set_size"]),
                float(final["test"].get("total", float("nan"))),
            )
        )
    by_layer_set: dict[tuple[int, int], list[float]] = {}
    for layer, set_size, value in layer_rows:
        by_layer_set.setdefault((layer, set_size), []).append(value)
    layers = sorted({key[0] for key in by_layer_set})
    train_set_sizes = sorted({key[1] for key in by_layer_set})
    plt.figure(figsize=(8, 4.5))
    for layer in layers:
        ys = []
        for set_size in train_set_sizes:
            vals = by_layer_set.get((layer, set_size), [])
            ys.append(sum(vals) / len(vals) if vals else float("nan"))
        plt.plot(train_set_sizes, ys, marker="o", label=f"layer {layer}")
    plt.ylabel("Final test total loss")
    plt.xlabel("Set size")
    plt.title("Training loss by layer and set size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "training_loss_by_layer_set_size.png", dpi=180)
    plt.close()

    control_scores: dict[str, list[float]] = {}
    for row in filtered:
        label = f"{row.get('method', 'setconca')}:{row['bridge']}"
        control_scores.setdefault(f"{label} real", []).append(float(row["topk_overlap"]))
        control_scores.setdefault(f"{label} shuffled", []).append(float(row["shuffled_topk_overlap"]))
    labels = sorted(control_scores)
    means = [sum(control_scores[label]) / len(control_scores[label]) for label in labels]
    colors = ["#4477aa" if label.endswith("real") else "#cc6677" for label in labels]
    plt.figure(figsize=(10, 4.8))
    plt.bar(range(len(labels)), means, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Mean TopK overlap")
    plt.title("Real vs shuffled-anchor bridge control")
    plt.tight_layout()
    plt.savefig(fig_dir / "bridge_real_vs_shuffled_control.png", dpi=180)
    plt.close()

    adjusted_scores: dict[str, list[float]] = {}
    for row in filtered:
        label = f"{row.get('method', 'setconca')}:{row['bridge']}"
        adjusted_scores.setdefault(label, []).append(float(row["real_minus_shuffled_topk"]))
    labels = sorted(adjusted_scores)
    means = [sum(adjusted_scores[label]) / len(adjusted_scores[label]) for label in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, means)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Mean real - shuffled TopK overlap")
    plt.xlabel("Bridge")
    plt.title("Shuffled-controlled bridge signal")
    plt.tight_layout()
    plt.savefig(fig_dir / "bridge_adjusted_topk_overlap.png", dpi=180)
    plt.close()

    set_adjusted: dict[tuple[int, str], list[float]] = {}
    for row in filtered:
        key = (int(row["set_size"]), f"{row.get('method', 'setconca')}:{row['bridge']}")
        set_adjusted.setdefault(key, []).append(float(row["real_minus_shuffled_topk"]))
    plt.figure(figsize=(8, 4.5))
    for bridge in sorted({key[1] for key in set_adjusted}):
        ys = []
        for set_size in set_sizes:
            vals = set_adjusted.get((set_size, bridge), [])
            ys.append(sum(vals) / len(vals) if vals else float("nan"))
        plt.plot(set_sizes, ys, marker="o", label=bridge)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Mean real - shuffled TopK overlap")
    plt.xlabel("Set size")
    plt.title("Shuffled-controlled transfer by set size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "set_size_adjusted_topk_overlap.png", dpi=180)
    plt.close()


def plot_heatmap(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    value_field: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    x_labels = sorted({str(row[x_field]) for row in rows})
    y_labels = sorted({str(row[y_field]) for row in rows})
    x_pos = {label: i for i, label in enumerate(x_labels)}
    y_pos = {label: i for i, label in enumerate(y_labels)}
    grid = [[float("nan") for _ in x_labels] for _ in y_labels]
    for row in rows:
        grid[y_pos[str(row[y_field])]][x_pos[str(row[x_field])]] = float(row[value_field])

    x_pretty = [pretty_family(label) if "family" in x_field else label for label in x_labels]
    y_pretty = [pretty_family(label) if "family" in y_field else label for label in y_labels]
    width = max(6.0, len(x_labels) * 1.7)
    height = max(4.8, len(y_labels) * 1.2)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(grid, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(image, ax=ax, shrink=0.88)
    cbar.set_label("Controlled TopK overlap", rotation=90)
    ax.set_xticks(range(len(x_labels)), x_pretty, rotation=0)
    ax.set_yticks(range(len(y_labels)), y_pretty)
    for yi, row_vals in enumerate(grid):
        for xi, value in enumerate(row_vals):
            if not math.isnan(value):
                text_color = "white" if value < 0.15 else "#222222"
                ax.text(xi, yi, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=10)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_summary_artifacts(run_dir: Path, summaries: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bridge_rows = [row for row in summaries.get("bridge_method_summary", []) if row["bridge"] != "identity"]
    methods = sorted({row["method"] for row in bridge_rows}, key=lambda x: (x != "pointwise_topk", x))
    bridges = sorted({row["bridge"] for row in bridge_rows}, key=lambda x: ["identity", "procrustes", "ridge", "mlp"].index(x) if x in {"identity", "procrustes", "ridge", "mlp"} else 99)
    lookup = {(row["method"], row["bridge"]): float(row["real_minus_shuffled_topk"]) for row in bridge_rows}
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    x = list(range(len(bridges)))
    width = 0.36
    colors = {"pointwise_topk": "#8da0cb", "setconca": "#66c2a5"}
    for i, method in enumerate(methods):
        offsets = [pos + (i - (len(methods) - 1) / 2) * width for pos in x]
        vals = [lookup.get((method, bridge), math.nan) for bridge in bridges]
        bars = ax.bar(offsets, vals, width=width, label=pretty_method(method), color=colors.get(method, "#4477aa"))
        add_bar_labels(ax, bars)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x, [pretty_bridge(bridge) for bridge in bridges])
    ax.set_ylabel(controlled_label())
    ax.set_title("SetConCA Gives The Stronger Controlled Linear Bridge Signal", fontsize=15, pad=12)
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.savefig(fig_dir / "summary_method_bridge_adjusted.png", dpi=220)
    plt.close(fig)

    relation_rows = summaries.get("method_relation_summary", [])
    relation_rows = [row for row in relation_rows if row["bridge"] in {"procrustes", "ridge"}]
    methods = sorted({row["method"] for row in relation_rows}, key=lambda x: (x != "pointwise_topk", x))
    relations = ["cross_family", "within_family"]
    bridges = ["procrustes", "ridge"]
    labels = [(relation, bridge) for relation in relations for bridge in bridges]
    lookup = {
        (row["method"], row["relation"], row["bridge"]): float(row["real_minus_shuffled_topk"])
        for row in relation_rows
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    x = list(range(len(labels)))
    width = 0.36
    for i, method in enumerate(methods):
        offsets = [pos + (i - (len(methods) - 1) / 2) * width for pos in x]
        vals = [lookup.get((method, relation, bridge), math.nan) for relation, bridge in labels]
        bars = ax.bar(offsets, vals, width=width, label=pretty_method(method), color=colors.get(method, "#228833"))
        add_bar_labels(ax, bars)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x, [f"{pretty_relation(relation)}\n{pretty_bridge(bridge)}" for relation, bridge in labels])
    ax.set_ylabel(controlled_label())
    ax.set_title("Controlled Transfer Holds Within And Across Llama/Qwen Families", fontsize=15, pad=12)
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.savefig(fig_dir / "summary_relation_bridge_adjusted.png", dpi=220)
    plt.close(fig)

    set_rows = summaries.get("set_size_summary", [])
    fig, ax = plt.subplots(figsize=(10.0, 5.6), constrained_layout=True)
    line_styles = {
        ("setconca", "procrustes"): ("#66c2a5", "o", "-"),
        ("setconca", "ridge"): ("#1b9e77", "o", "-"),
        ("pointwise_topk", "procrustes"): ("#8da0cb", "s", "--"),
        ("pointwise_topk", "ridge"): ("#4c78a8", "s", "--"),
    }
    plotted = []
    for label in sorted({f"{row['method']}:{row['bridge']}" for row in set_rows}):
        method, bridge = label.split(":", 1)
        if bridge == "identity":
            continue
        vals = [
            row
            for row in set_rows
            if row["method"] == method and row["bridge"] == bridge
        ]
        vals = sorted(vals, key=lambda row: int(row["set_size"]))
        color, marker, linestyle = line_styles.get((method, bridge), ("#555555", "o", "-"))
        plotted.append((method, bridge))
        ax.plot(
            [int(row["set_size"]) for row in vals],
            [float(row["real_minus_shuffled_topk"]) for row in vals],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.3,
            label=f"{pretty_method(method)} + {pretty_bridge(bridge)}",
        )
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Number of paraphrases per semantic set")
    ax.set_ylabel(controlled_label())
    ax.set_title("Larger Semantic Sets Improve SetConCA Transfer", fontsize=15, pad=12)
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.savefig(fig_dir / "summary_set_size_adjusted.png", dpi=220)
    plt.close(fig)

    for method in sorted({row["method"] for row in summaries.get("depth_pair_summary", [])}):
        for bridge in sorted({row["bridge"] for row in summaries.get("depth_pair_summary", [])}):
            rows = [
                row
                for row in summaries["depth_pair_summary"]
                if row["method"] == method and row["bridge"] == bridge
            ]
            if rows:
                plot_heatmap(
                    fig_dir / f"summary_depth_heatmap_{method}_{bridge}.png",
                    rows,
                    x_field="target_depth_pct",
                    y_field="source_depth_pct",
                    value_field="real_minus_shuffled_topk",
                    title=f"Depth Pair Signal: {pretty_method(method)} + {pretty_bridge(bridge)}",
                    xlabel="Target depth (%)",
                    ylabel="Source depth (%)",
                )

    for method in sorted({row["method"] for row in summaries.get("family_pair_summary", [])}):
        for bridge in sorted({row["bridge"] for row in summaries.get("family_pair_summary", [])}):
            rows = [
                row
                for row in summaries["family_pair_summary"]
                if row["method"] == method and row["bridge"] == bridge
            ]
            if rows:
                plot_heatmap(
                    fig_dir / f"summary_family_heatmap_{method}_{bridge}.png",
                    rows,
                    x_field="target_family",
                    y_field="source_family",
                    value_field="real_minus_shuffled_topk",
                    title=f"Family Pair Signal: {pretty_method(method)} + {pretty_bridge(bridge)}",
                    xlabel="Target family",
                    ylabel="Source family",
                )


def write_report(
    run_dir: Path,
    banks: list[BankSpec],
    trained: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    summaries: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    report = run_dir / "REPORT.md"
    bridge_summary: dict[tuple[str, str], list[dict[str, float]]] = {}
    alpha_summary: dict[float, list[float]] = {}
    for row in results:
        if row["steering_alpha"] == 0.0:
            bridge_summary.setdefault((row.get("method", "setconca"), row["bridge"]), []).append(
                {
                    "raw": float(row["topk_overlap"]),
                    "shuffled": float(row["shuffled_topk_overlap"]),
                    "adjusted": float(row["real_minus_shuffled_topk"]),
                    "train_test_gap": float(row["train_test_topk_gap"]),
                }
            )
        alpha_summary.setdefault(float(row["steering_alpha"]), []).append(float(row["steering_gain_vs_random"]))

    lines = [
        "# SetConCA V2 Transfer and Steering Run",
        "",
        "## Scope",
        "",
        f"- Activation banks selected: {len(banks)}",
        f"- Trained SetConCA models: {len(trained)}",
        f"- Pair/bridge/steering rows: {len(results)}",
        "",
        "## Bridge Transfer Summary",
        "",
        "Raw TopK overlap is reported with a shuffled-anchor control. The main controlled transfer score is `real - shuffled`.",
        "",
        "| Method | Bridge | Raw TopK | Shuffled TopK | Real - shuffled | Train-test gap | N pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, bridge in sorted(bridge_summary):
        vals = bridge_summary[(method, bridge)]
        raw = sum(row["raw"] for row in vals) / len(vals)
        shuffled = sum(row["shuffled"] for row in vals) / len(vals)
        adjusted = sum(row["adjusted"] for row in vals) / len(vals)
        train_test_gap = sum(row["train_test_gap"] for row in vals) / len(vals)
        lines.append(
            f"| `{method}` | `{bridge}` | {raw:.4f} | {shuffled:.4f} | {adjusted:.4f} | {train_test_gap:.4f} | {len(vals)} |"
        )
    lines.extend(
        [
            "",
            "## Steering Proxy Summary",
            "",
            "| Alpha | Mean structured-random similarity | N rows |",
            "| ---: | ---: | ---: |",
        ]
    )
    for alpha in sorted(alpha_summary):
        vals = alpha_summary[alpha]
        lines.append(f"| {alpha:.2f} | {sum(vals) / len(vals):.4f} | {len(vals)} |")
    if summaries:
        lines.extend(
            [
                "",
                "## Automatic Summaries",
                "",
                "These tables are generated from `transfer_steering_results.csv` at `steering_alpha=0`.",
                "",
                "### Within/Cross-Family Summary",
                "",
                "| Method | Relation | Bridge | Real - shuffled | Raw TopK | Shuffled TopK | N |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summaries.get("method_relation_summary", []):
            lines.append(
                f"| `{row['method']}` | `{row['relation']}` | `{row['bridge']}` | "
                f"{float(row['real_minus_shuffled_topk']):.4f} | {float(row['raw_topk']):.4f} | "
                f"{float(row['shuffled_topk']):.4f} | {row['n']} |"
            )
        lines.extend(
            [
                "",
                "### Depth-Pair Summary",
                "",
                "| Method | Source depth | Target depth | Bridge | Real - shuffled | N |",
                "| --- | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in summaries.get("depth_pair_summary", []):
            lines.append(
                f"| `{row['method']}` | {row['source_depth_pct']} | {row['target_depth_pct']} | "
                f"`{row['bridge']}` | {float(row['real_minus_shuffled_topk']):.4f} | {row['n']} |"
            )
    lines.extend(
        [
            "",
            "## Important Caveat",
            "",
            "The steering metric here is a concept-code steering proxy, not a full language-model behavioral intervention. It tests whether bridged concept directions move target concept codes more than random directions.",
            "",
            "Raw bridge overlap can be inflated if different examples use similar high-frequency concepts. Treat `real_minus_shuffled_topk` as the first-pass controlled bridge signal.",
            "",
            "## Artifacts",
            "",
            "- `training_summary.csv`",
            "- `transfer_steering_results.csv`",
            "- `transfer_steering_results.json`",
            "- `figures/bridge_topk_overlap.png`",
            "- `figures/steering_proxy_gain.png`",
            "- `figures/set_size_bridge_topk_overlap.png`",
            "- `figures/training_loss_by_layer_set_size.png`",
            "- `figures/bridge_real_vs_shuffled_control.png`",
            "- `figures/bridge_adjusted_topk_overlap.png`",
            "- `figures/set_size_adjusted_topk_overlap.png`",
            "- `summaries/bridge_method_summary.csv`",
            "- `summaries/method_relation_summary.csv`",
            "- `summaries/family_pair_summary.csv`",
            "- `summaries/depth_pair_summary.csv`",
            "- `summaries/set_size_summary.csv`",
            "- `figures/summary_method_bridge_adjusted.png`",
            "- `figures/summary_relation_bridge_adjusted.png`",
            "- `figures/summary_set_size_adjusted.png`",
            "- `figures/summary_depth_heatmap_<method>_<bridge>.png`",
            "- `figures/summary_family_heatmap_<method>_<bridge>.png`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_set_sizes(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate SetConCA V2 concept transfer and steering grid.")
    parser.add_argument("--activation-root", default="data/activations/model_grid_s16_min16_4A100")
    parser.add_argument("--out-dir", default="results/v2_transfer_steering")
    parser.add_argument("--set-sizes", default="2,4,6,8,10,12,14,16")
    parser.add_argument("--only-family", default=None, help="Comma-separated families to include, e.g. llama3,qwen3")
    parser.add_argument("--exclude-family", default=None, help="Comma-separated families to exclude, e.g. gemma3")
    parser.add_argument("--only-size", default=None)
    parser.add_argument("--only-layer-pct", default=None, help="Example: 20, 60, or 90")
    parser.add_argument("--max-banks", type=int, default=None)
    parser.add_argument("--max-sets", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--concept-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--residual-rank", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--bridges", default="identity,procrustes,ridge,mlp")
    parser.add_argument("--methods", default="setconca", help="Comma-separated methods: setconca,pointwise_topk")
    parser.add_argument("--include-cross-method-pairs", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse existing codes.pt and metrics.json model artifacts when present.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--steering-alphas", default="0,0.5,1,2,5")
    parser.add_argument("--include-self-pairs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shared-recon-weight", type=float, default=1.0)
    parser.add_argument("--full-recon-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--support-consistency-weight", type=float, default=0.05)
    parser.add_argument("--offdiag-weight", type=float, default=0.01)
    parser.add_argument("--residual-energy-weight", type=float, default=0.01)
    args = parser.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    activation_root = resolve_project_path(args.activation_root)
    run_dir = resolve_project_path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    banks = filter_banks(
        discover_banks(activation_root),
        only_family=args.only_family,
        exclude_family=args.exclude_family,
        only_size=args.only_size,
        only_layer_pct=args.only_layer_pct,
        max_banks=args.max_banks,
    )
    set_sizes = parse_set_sizes(args.set_sizes)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown_methods = sorted(set(methods) - {"setconca", "pointwise_topk"})
    if unknown_methods:
        raise ValueError(f"Unsupported method(s): {', '.join(unknown_methods)}")
    manifest = {
        "activation_root": str(activation_root),
        "out_dir": str(run_dir),
        "args": vars(args),
        "banks": [asdict(b) | {"path": str(b.path)} for b in banks],
        "set_sizes": set_sizes,
        "methods": methods,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Selected {len(banks)} activation bank(s); run_dir={run_dir}", flush=True)
    if args.dry_run:
        for bank in banks:
            for set_size in set_sizes:
                print(f"DRY {bank.bank_id} S={set_size} {bank.path}")
        return

    weights = LossWeights(
        shared_recon=args.shared_recon_weight,
        full_recon=args.full_recon_weight,
        contrastive=args.contrastive_weight,
        support_consistency=args.support_consistency_weight,
        offdiag=args.offdiag_weight,
        residual_energy=args.residual_energy_weight,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    trained: dict[str, dict[str, Any]] = {}
    total_jobs = len(banks) * len(set_sizes) * len(methods)
    job_idx = 0
    for bank in banks:
        for set_size in set_sizes:
            for method in methods:
                job_idx += 1
                key = f"{method}__{bank.bank_id}__s{set_size:02d}"
                out_dir = (
                    run_dir
                    / "models"
                    / method
                    / bank.family
                    / f"{bank.size}_{bank.size_label}"
                    / bank.model_slug
                    / bank.layer_name
                    / f"s{set_size:02d}"
                )
                print(f"[{job_idx}/{total_jobs}] train {key}", flush=True)
                train_spec = TrainSpec(bank, set_size, out_dir)
                sliced = slice_bank(load_activation_bank(bank.path), set_size=set_size, max_sets=args.max_sets)
                codes_path = out_dir / "codes.pt"
                metrics_path = out_dir / "metrics.json"
                if args.resume and codes_path.exists() and metrics_path.exists():
                    print(f"[{job_idx}/{total_jobs}] resume {key}", flush=True)
                    codes = torch.load(codes_path, map_location="cpu", weights_only=False)
                    training = json.loads(metrics_path.read_text(encoding="utf-8"))
                elif method == "setconca":
                    model, training = train_model(train_spec, args, device, weights)
                    codes = extract_codes(model, sliced, train_frac=args.train_frac, batch_size=args.batch_size, device=device)
                    torch.save(codes, codes_path)
                else:
                    model, training = train_pointwise_model(train_spec, args, device)
                    codes = extract_pointwise_codes(model, sliced, train_frac=args.train_frac, batch_size=args.batch_size, device=device)
                    torch.save(codes, codes_path)
                trained[key] = {
                    "method": method,
                    "bank": bank,
                    "set_size": set_size,
                    "training": training,
                    "codes": codes,
                    "out_dir": out_dir,
                }

    write_training_summary(run_dir / "training_summary.csv", trained)
    write_code_diagnostics(run_dir / "code_diagnostics.csv", trained)
    results = evaluate_pairs(trained, args, run_dir)
    summaries = write_summary_artifacts(run_dir, results)
    plot_results(run_dir, trained, results)
    plot_summary_artifacts(run_dir, summaries)
    write_report(run_dir, banks, trained, results, summaries)
    final = {
        "elapsed_s": time.time() - t0,
        "n_banks": len(banks),
        "n_trained": len(trained),
        "n_result_rows": len(results),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
