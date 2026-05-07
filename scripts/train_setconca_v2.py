from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from data.activation_sets import load_activation_bank
from model.setconca_v2 import SetConCAV2
from setconca_v2.paths import resolve_project_path
from training.losses import LossWeights, compute_v2_loss


def batch_indices(n: int, batch_size: int, *, shuffle: bool, device: torch.device) -> List[torch.Tensor]:
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    return [idx[start : start + batch_size] for start in range(0, n, batch_size)]


def average_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def evaluate(model: SetConCAV2, x: torch.Tensor, weights: LossWeights, batch_size: int) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for idx in batch_indices(len(x), batch_size, shuffle=False, device=x.device):
            xb = x[idx]
            _, parts = compute_v2_loss(model, xb, None, weights)
            rows.append({key: float(value.cpu()) for key, value in parts.items()})
    return average_metrics(rows)


def train(args: argparse.Namespace) -> Dict[str, object]:
    t0 = time.time()
    bank = load_activation_bank(resolve_project_path(args.activations))
    hidden = bank.hidden
    if args.max_sets is not None:
        hidden = hidden[: args.max_sets]
    if args.views is not None:
        hidden = hidden[:, : args.views]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hidden = hidden.to(device)
    train_x, test_x = bank.__class__(hidden, bank.texts[: len(hidden)], bank.meta).train_test(args.train_frac)

    model = SetConCAV2(
        hidden_dim=hidden.shape[-1],
        concept_dim=args.concept_dim,
        topk=args.topk,
        residual_scale=args.residual_scale,
        residual_rank=args.residual_rank,
        dropout=args.dropout,
    ).to(device)
    weights = LossWeights(
        shared_recon=args.shared_recon_weight,
        full_recon=args.full_recon_weight,
        contrastive=args.contrastive_weight,
        support_consistency=args.support_consistency_weight,
        offdiag=args.offdiag_weight,
        residual_energy=args.residual_energy_weight,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rows: List[Dict[str, float]] = []
        for idx in batch_indices(len(train_x), args.batch_size, shuffle=True, device=device):
            xb = train_x[idx]
            neg_idx = torch.randperm(len(train_x), device=device)[: len(idx)]
            xneg = train_x[neg_idx]
            loss, parts = compute_v2_loss(model, xb, xneg, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            rows.append({key: float(value.cpu()) for key, value in parts.items()})
        train_metrics = average_metrics(rows)
        test_metrics = evaluate(model, test_x, weights, args.batch_size) if len(test_x) else {}
        item = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(item)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} train_total={train_metrics.get('total', 0):.6f} "
                f"test_total={test_metrics.get('total', 0):.6f}",
                flush=True,
            )

    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "model.pt"
    metrics_path = out_dir / "metrics.json"
    manifest = {
        "activations": str(resolve_project_path(args.activations)),
        "activation_meta": bank.meta,
        "hidden_shape": list(hidden.shape),
        "train_shape": list(train_x.shape),
        "test_shape": list(test_x.shape),
        "device": str(device),
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
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SetConCA V2 on an activation bank.")
    parser.add_argument("--activations", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--views", type=int, default=None)
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
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--shared-recon-weight", type=float, default=1.0)
    parser.add_argument("--full-recon-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--support-consistency-weight", type=float, default=0.05)
    parser.add_argument("--offdiag-weight", type=float, default=0.01)
    parser.add_argument("--residual-energy-weight", type=float, default=0.01)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
