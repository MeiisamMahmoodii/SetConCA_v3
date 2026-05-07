from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.setconca_v2 import SetConCAV2


@dataclass
class LossWeights:
    shared_recon: float = 1.0
    full_recon: float = 0.05
    contrastive: float = 0.25
    support_consistency: float = 0.05
    offdiag: float = 0.01
    residual_energy: float = 0.01


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.T / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    )


def hard_negative_margin(
    z_a: torch.Tensor,
    z_p: torch.Tensor,
    z_n: torch.Tensor | None,
    margin: float = 0.2,
) -> torch.Tensor:
    if z_n is None:
        return z_a.new_tensor(0.0)
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_p, dim=-1)
    z_n = F.normalize(z_n, dim=-1)
    pos = (z_a * z_p).sum(dim=-1)
    neg = (z_a * z_n).sum(dim=-1)
    return F.relu(margin + neg - pos).mean()


def support_consistency_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    return ((torch.sigmoid(z_a.abs()) - torch.sigmoid(z_b.abs())) ** 2).mean()


def offdiag_decorrelation(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    zn = F.normalize(z, dim=0)
    gram = zn.T @ zn
    eye = torch.eye(gram.shape[0], device=z.device)
    return ((gram - eye) ** 2).mean()


def split_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = x.shape[1]
    if s == 1:
        return x, x
    mid = max(1, s // 2)
    return x[:, :mid, :], x[:, mid:, :] if mid < s else x[:, :mid, :]


def compute_v2_loss(
    model: SetConCAV2,
    x: torch.Tensor,
    x_negative: torch.Tensor | None,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = model(x)
    full_recon = ((out.recon - x) ** 2).mean()
    shared_recon = ((out.shared_recon - x) ** 2).mean()

    xa, xp = split_views(x)
    za = model(xa).z
    zp = model(xp).z
    zn = model(x_negative).z if x_negative is not None else None

    contrast = info_nce(za, zp) + hard_negative_margin(za, zp, zn)
    support = support_consistency_loss(za, zp)
    offdiag = offdiag_decorrelation(out.z)
    residual_energy = (out.recon - out.shared_recon).pow(2).mean()

    total = (
        weights.shared_recon * shared_recon +
        weights.full_recon * full_recon +
        weights.contrastive * contrast +
        weights.support_consistency * support +
        weights.offdiag * offdiag +
        weights.residual_energy * residual_energy
    )
    parts = {
        "total": total.detach(),
        "shared_recon": shared_recon.detach(),
        "full_recon": full_recon.detach(),
        "contrastive": contrast.detach(),
        "support_consistency": support.detach(),
        "offdiag": offdiag.detach(),
        "residual_energy": residual_energy.detach(),
    }
    return total, parts
