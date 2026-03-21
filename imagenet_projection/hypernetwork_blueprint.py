"""Spectral hypernetwork projection baseline for episode-wise adaptation.

This module converts support-set spectral statistics into a context vector,
then predicts a per-episode linear projection matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention_spectral_encoder import AttentionSpectralEncoder


def _init_generators_as_random_projection(
    *,
    weight_generator: nn.Linear,
    bias_generator: nn.Linear,
    input_dim: int,
    output_dim: int,
    tiny_weight_std: float = 1e-4,
) -> None:
    """Make generator output a fixed random projection at initialization.

    With this initialization, the generated projection matrix is independent of
    context at step 0 because generator weights are zeroed and only biases are
    active.
    """
    if tiny_weight_std <= 0:
        raise ValueError("tiny_weight_std must be > 0")

    scale = float(output_dim) ** -0.5
    W0 = torch.randn(input_dim, output_dim) * scale

    with torch.no_grad():
        weight_generator.weight.normal_(mean=0.0, std=tiny_weight_std)
        weight_generator.bias.copy_(W0.reshape(-1))
        bias_generator.weight.normal_(mean=0.0, std=tiny_weight_std)
        bias_generator.bias.zero_()


def _init_low_rank_generators_as_random_projection(
    *,
    a_generator: nn.Linear,
    b_generator: nn.Linear,
    bias_generator: nn.Linear,
    input_dim: int,
    output_dim: int,
    rank: int,
    tiny_weight_std: float = 1e-4,
) -> None:
    """Initialize low-rank generators to emit a fixed random projection.

    At step 0, context dependence is very small due to tiny random generator
    weights, while biases define a fixed low-rank projection A @ B.
    """
    if rank <= 0:
        raise ValueError("rank must be > 0")
    if tiny_weight_std <= 0:
        raise ValueError("tiny_weight_std must be > 0")

    A0 = torch.randn(input_dim, rank) * (float(rank) ** -0.5)
    B0 = torch.randn(rank, output_dim) * (float(output_dim) ** -0.5)

    with torch.no_grad():
        a_generator.weight.normal_(mean=0.0, std=tiny_weight_std)
        a_generator.bias.copy_(A0.reshape(-1))
        b_generator.weight.normal_(mean=0.0, std=tiny_weight_std)
        b_generator.bias.copy_(B0.reshape(-1))
        bias_generator.weight.normal_(mean=0.0, std=tiny_weight_std)
        bias_generator.bias.zero_()


class SpectralContextEncoder(nn.Module):
    """Encode top-k support singular values/vectors into a learned context."""

    def __init__(
        self,
        *,
        input_dim: int,
        top_k: int = 16,
        context_hidden_dim: int = 256,
        covariance_jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if context_hidden_dim <= 0:
            raise ValueError("context_hidden_dim must be > 0")
        if covariance_jitter <= 0:
            raise ValueError("covariance_jitter must be > 0")

        self.top_k = int(top_k)
        self.input_dim = int(input_dim)
        self.context_hidden_dim = int(context_hidden_dim)
        self.covariance_jitter = float(covariance_jitter)

        # Raw context packs top-k singular values and top-k right-singular vectors.
        self.context_dim = self.top_k + (self.top_k * self.input_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(self.context_dim, self.context_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.context_hidden_dim, self.context_hidden_dim),
        )

    def forward(self, x_support: torch.Tensor) -> torch.Tensor:
        """Encode one support matrix [Ns, D] into context [1, C]."""
        if x_support.ndim != 2:
            raise ValueError(f"Expected x_support shape [Ns, D], got {tuple(x_support.shape)}")
        if x_support.shape[0] == 0:
            raise ValueError("x_support must have at least one row")
        if int(x_support.shape[1]) != self.input_dim:
            raise ValueError(
                f"x_support has feature dim {int(x_support.shape[1])}, expected {self.input_dim}"
            )

        x_centered = x_support - x_support.mean(dim=0, keepdim=True)
        n_support = int(x_centered.shape[0])
        cov_denom = max(n_support - 1, 1)
        cov = (x_centered.T @ x_centered) / float(cov_denom)
        cov = cov + self.covariance_jitter * torch.eye(
            self.input_dim,
            device=x_support.device,
            dtype=x_support.dtype,
        )
        eigvals, eigvecs = torch.linalg.eigh(cov)

        k = min(self.top_k, int(eigvals.shape[0]))
        top_eigvals = torch.flip(eigvals[-k:], dims=[0]).clamp_min(0.0)
        top_s = torch.sqrt(top_eigvals * float(cov_denom))
        top_vh = torch.flip(eigvecs[:, -k:], dims=[1]).T

        # Deterministic sign convention: orient each vector so its largest-magnitude
        # element is non-negative, removing eigenvector sign ambiguity.
        max_abs_indices = torch.argmax(torch.abs(top_vh), dim=1)
        signs = torch.sign(top_vh[torch.arange(k, device=top_vh.device), max_abs_indices])
        signs[signs == 0] = 1.0
        top_vh = top_vh * signs.unsqueeze(1)

        if k < self.top_k:
            pad_s = torch.zeros(self.top_k - k, device=x_support.device, dtype=x_support.dtype)
            pad_vh = torch.zeros(
                self.top_k - k,
                self.input_dim,
                device=x_support.device,
                dtype=x_support.dtype,
            )
            top_s = torch.cat([top_s, pad_s], dim=0)
            top_vh = torch.cat([top_vh, pad_vh], dim=0)

        raw_context = torch.cat([top_s, top_vh.reshape(-1)], dim=0).unsqueeze(0)
        return self.context_mlp(raw_context)


class SpectralHypernetworkAdapter(nn.Module):
    """Predict an episode-specific projection from support spectral statistics."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        top_k_components: int = 16,
        context_hidden_dim: int = 256,
        encoder_type: str = "mlp",
        attention_num_heads: int = 4,
        attention_num_layers: int = 2,
        low_rank_dim: int | None = None,
        tiny_weight_std: float = 1e-4,
        covariance_jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if top_k_components <= 0:
            raise ValueError("top_k_components must be > 0")
        if context_hidden_dim <= 0:
            raise ValueError("context_hidden_dim must be > 0")
        if encoder_type not in {"mlp", "attention"}:
            raise ValueError("encoder_type must be one of ['mlp', 'attention']")
        if attention_num_heads <= 0:
            raise ValueError("attention_num_heads must be > 0")
        if attention_num_layers <= 0:
            raise ValueError("attention_num_layers must be > 0")
        if low_rank_dim is not None and low_rank_dim <= 0:
            raise ValueError("low_rank_dim must be > 0 when provided")
        if tiny_weight_std <= 0:
            raise ValueError("tiny_weight_std must be > 0")
        if covariance_jitter <= 0:
            raise ValueError("covariance_jitter must be > 0")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.top_k_components = int(top_k_components)
        self.encoder_type = str(encoder_type)
        self.low_rank_dim = int(low_rank_dim) if low_rank_dim is not None else None

        if self.encoder_type == "mlp":
            self.encoder = SpectralContextEncoder(
                input_dim=self.input_dim,
                top_k=self.top_k_components,
                context_hidden_dim=context_hidden_dim,
                covariance_jitter=covariance_jitter,
            )
        else:
            self.encoder = AttentionSpectralEncoder(
                input_dim=self.input_dim,
                top_k=self.top_k_components,
                embed_dim=int(context_hidden_dim),
                num_heads=int(attention_num_heads),
                num_layers=int(attention_num_layers),
            )

        context_dim = self.encoder.context_hidden_dim
        self.weight_generator: nn.Linear | None = None
        self.a_generator: nn.Linear | None = None
        self.b_generator: nn.Linear | None = None

        if self.low_rank_dim is None:
            self.weight_generator = nn.Linear(context_dim, self.input_dim * self.output_dim)
        else:
            self.a_generator = nn.Linear(context_dim, self.input_dim * self.low_rank_dim)
            self.b_generator = nn.Linear(context_dim, self.low_rank_dim * self.output_dim)
        self.bias_generator = nn.Linear(context_dim, self.output_dim)
        if self.low_rank_dim is None:
            _init_generators_as_random_projection(
                weight_generator=self.weight_generator,
                bias_generator=self.bias_generator,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                tiny_weight_std=tiny_weight_std,
            )
        else:
            _init_low_rank_generators_as_random_projection(
                a_generator=self.a_generator,
                b_generator=self.b_generator,
                bias_generator=self.bias_generator,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                rank=self.low_rank_dim,
                tiny_weight_std=tiny_weight_std,
            )

    def forward(self, *, features: torch.Tensor, support_indices: torch.Tensor) -> torch.Tensor:
        """Project all episode rows using support-conditioned generated weights.

        Args:
            features: Episode tensor of shape [N, D].
            support_indices: Row indices used to build support-conditioned context.
        """
        if features.ndim != 2:
            raise ValueError(f"Expected features shape [N, D], got {tuple(features.shape)}")
        if support_indices.ndim != 1:
            raise ValueError("support_indices must be a 1D tensor")
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"features has dim {int(features.shape[1])}, expected {self.input_dim}"
            )

        x_support = features.index_select(0, support_indices)
        if x_support.shape[0] == 0:
            raise ValueError("support_indices produced an empty support set")

        support_mean = x_support.mean(dim=0, keepdim=True)
        x_target = features - support_mean

        context = self.encoder(x_support)
        if self.low_rank_dim is None:
            weights_flat = self.weight_generator(context)
            W = weights_flat.view(self.input_dim, self.output_dim)
        else:
            A = self.a_generator(context).view(self.input_dim, self.low_rank_dim)
            B = self.b_generator(context).view(self.low_rank_dim, self.output_dim)
            W = A @ B
        b = self.bias_generator(context).squeeze(0)

        return x_target @ W + b


class VanillaStatsHypernetworkAdapter(nn.Module):
    """Predict an episode-specific projection from simple support statistics.

    This baseline avoids spectral decomposition and conditions on concatenated
    per-feature support mean and standard deviation.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        context_hidden_dim: int = 256,
        eps: float = 1e-6,
        low_rank_dim: int | None = None,
        tiny_weight_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if context_hidden_dim <= 0:
            raise ValueError("context_hidden_dim must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if low_rank_dim is not None and low_rank_dim <= 0:
            raise ValueError("low_rank_dim must be > 0 when provided")
        if tiny_weight_std <= 0:
            raise ValueError("tiny_weight_std must be > 0")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.eps = float(eps)
        self.low_rank_dim = int(low_rank_dim) if low_rank_dim is not None else None

        context_dim = 2 * self.input_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, int(context_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(context_hidden_dim), int(context_hidden_dim)),
            nn.ReLU(),
        )
        self.weight_generator: nn.Linear | None = None
        self.a_generator: nn.Linear | None = None
        self.b_generator: nn.Linear | None = None
        if self.low_rank_dim is None:
            self.weight_generator = nn.Linear(
                int(context_hidden_dim), self.input_dim * self.output_dim
            )
        else:
            self.a_generator = nn.Linear(int(context_hidden_dim), self.input_dim * self.low_rank_dim)
            self.b_generator = nn.Linear(int(context_hidden_dim), self.low_rank_dim * self.output_dim)
        self.bias_generator = nn.Linear(int(context_hidden_dim), self.output_dim)
        if self.low_rank_dim is None:
            _init_generators_as_random_projection(
                weight_generator=self.weight_generator,
                bias_generator=self.bias_generator,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                tiny_weight_std=tiny_weight_std,
            )
        else:
            _init_low_rank_generators_as_random_projection(
                a_generator=self.a_generator,
                b_generator=self.b_generator,
                bias_generator=self.bias_generator,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                rank=self.low_rank_dim,
                tiny_weight_std=tiny_weight_std,
            )

    def forward(self, *, features: torch.Tensor, support_indices: torch.Tensor) -> torch.Tensor:
        """Project all episode rows using support-conditioned generated weights."""
        if features.ndim != 2:
            raise ValueError(f"Expected features shape [N, D], got {tuple(features.shape)}")
        if support_indices.ndim != 1:
            raise ValueError("support_indices must be a 1D tensor")
        if int(features.shape[1]) != self.input_dim:
            raise ValueError(
                f"features has dim {int(features.shape[1])}, expected {self.input_dim}"
            )

        x_support = features.index_select(0, support_indices)
        if x_support.shape[0] == 0:
            raise ValueError("support_indices produced an empty support set")

        support_mean = x_support.mean(dim=0, keepdim=True)
        support_std = x_support.var(dim=0, unbiased=False, keepdim=True).add(self.eps).sqrt()

        stats = torch.cat([support_mean, support_std], dim=1)
        context = self.context_mlp(stats)

        if self.low_rank_dim is None:
            weights_flat = self.weight_generator(context)
            W = weights_flat.view(self.input_dim, self.output_dim)
        else:
            A = self.a_generator(context).view(self.input_dim, self.low_rank_dim)
            B = self.b_generator(context).view(self.low_rank_dim, self.output_dim)
            W = A @ B
        b = self.bias_generator(context).squeeze(0)

        x_target = features - support_mean
        return x_target @ W + b