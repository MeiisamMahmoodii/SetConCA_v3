from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ForwardOutput:
    recon: torch.Tensor
    shared_recon: torch.Tensor
    z: torch.Tensor
    z_dense: torch.Tensor
    u: torch.Tensor


class SetConCAV2(nn.Module):
    """
    Set-ConCA V2.

    Key design changes from V1:
    - explicit normalized contrastive embedding path;
    - shared-only reconstruction is always exposed;
    - residual path is capacity/scale controlled;
    - TopK can select by absolute value to preserve signed concept evidence.
    """

    def __init__(
        self,
        hidden_dim: int,
        concept_dim: int = 128,
        topk: int = 32,
        residual_scale: float = 0.05,
        residual_rank: int | None = None,
        dropout: float = 0.0,
        topk_abs: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.concept_dim = concept_dim
        self.topk = min(topk, concept_dim)
        self.residual_scale = residual_scale
        self.topk_abs = topk_abs

        self.encoder = nn.Linear(hidden_dim, concept_dim)
        self.norm = nn.LayerNorm(concept_dim, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shared_decoder = nn.Linear(concept_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        if residual_rank is None:
            self.residual_down = None
            self.residual_up = nn.Linear(concept_dim, hidden_dim, bias=False)
        else:
            self.residual_down = nn.Linear(concept_dim, residual_rank, bias=False)
            self.residual_up = nn.Linear(residual_rank, hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.shared_decoder.weight)
        if self.residual_down is not None:
            nn.init.xavier_uniform_(self.residual_down.weight)
        nn.init.xavier_uniform_(self.residual_up.weight)

    def topk_sparse(self, z_dense: torch.Tensor) -> torch.Tensor:
        score = z_dense.abs() if self.topk_abs else z_dense
        _, idx = torch.topk(score, self.topk, dim=-1)
        mask = torch.zeros_like(z_dense)
        mask.scatter_(-1, idx, 1.0)
        return z_dense * mask

    def encode_views(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def aggregate(self, u: torch.Tensor) -> torch.Tensor:
        pooled = u.mean(dim=1)
        return self.dropout(self.norm(pooled))

    def encode_set(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.encode_views(x)
        z_dense = self.aggregate(u)
        z = self.topk_sparse(z_dense)
        return z, z_dense, u

    def decode_shared(self, z: torch.Tensor) -> torch.Tensor:
        return self.shared_decoder(z) + self.bias

    def decode_residual(self, u: torch.Tensor) -> torch.Tensor:
        if self.residual_down is None:
            out = self.residual_up(u)
        else:
            out = self.residual_up(self.residual_down(u))
        return out * self.residual_scale

    def forward(self, x: torch.Tensor) -> ForwardOutput:
        z, z_dense, u = self.encode_set(x)
        shared = self.decode_shared(z).unsqueeze(1)
        residual = self.decode_residual(u)
        recon = shared + residual
        return ForwardOutput(
            recon=recon,
            shared_recon=shared + torch.zeros_like(x),
            z=z,
            z_dense=z_dense,
            u=u,
        )

    @torch.no_grad()
    def code(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).z

    @torch.no_grad()
    def normalized_code(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.code(x), dim=-1)

