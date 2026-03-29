import torch
import torch.nn as nn
import time


def _synchronized_perf_counter(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class AttentionSpectralEncoder(nn.Module):
    """Encode top-k support singular values/vectors using self-attention."""

    def __init__(
        self,
        *,
        input_dim: int,
        top_k: int = 64,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        use_positional_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.input_dim = int(input_dim)
        self.top_k = int(top_k)
        self.context_hidden_dim = int(embed_dim)
        self.use_positional_embeddings = bool(use_positional_embeddings)

        # 1. Project singular-value-weighted vectors to embedding dimension.
        self.token_projection = nn.Linear(self.input_dim, self.context_hidden_dim)
        self.pos_embed: nn.Parameter | None
        if self.use_positional_embeddings:
            self.pos_embed = nn.Parameter(torch.randn(1, self.top_k, self.context_hidden_dim) * 0.02)
        else:
            self.pos_embed = None

        # 2. Lightweight Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.context_hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.context_hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.last_profile: dict[str, float] = {}

    def forward(self, x_support: torch.Tensor) -> torch.Tensor:
        """Encode one support matrix [Ns, D] into token memory [1, top_k, embed_dim]."""
        if x_support.ndim != 2:
            raise ValueError(f"Expected x_support shape [Ns, D], got {tuple(x_support.shape)}")
        if x_support.shape[0] == 0:
            raise ValueError("x_support must have at least one row")
        if int(x_support.shape[1]) != self.input_dim:
            raise ValueError(
                f"x_support has feature dim {int(x_support.shape[1])}, expected {self.input_dim}"
            )

        x_centered = x_support - x_support.mean(dim=0, keepdim=True)

        # Calculate SVD (eigendecomposition-equivalent spectral step)
        t0 = _synchronized_perf_counter(x_support.device)
        _, svals, vh = torch.linalg.svd(x_centered, full_matrices=False)
        t1 = _synchronized_perf_counter(x_support.device)
        self.last_profile = {
            "eigendecomp_ms": float((t1 - t0) * 1000.0),
        }

        # --- DETERMINISTIC SIGN CORRECTION ---
        # Find the index of the max absolute value in each singular vector
        max_abs_indices = torch.argmax(torch.abs(vh), dim=1)
        # Extract the signs of those specific elements
        signs = torch.sign(vh[torch.arange(vh.shape[0]), max_abs_indices])
        # Force the signs to be positive (if 0, keep as 1)
        signs[signs == 0] = 1.0
        # Broadcast and multiply to correct the vectors
        vh = vh * signs.unsqueeze(1)
        # -------------------------------------

        # Truncate or Pad to top_k
        k_actual = min(self.top_k, int(svals.shape[0]))
        top_s = svals[:k_actual]
        top_vh = vh[:k_actual, :]

        # Singular value injection: normalize values and scale each component vector.
        s_max = torch.max(top_s).clamp_min(1e-8)
        s_normalized = top_s / s_max
        top_vh = top_vh * s_normalized.unsqueeze(1)

        if k_actual < self.top_k:
            pad_s = torch.zeros(self.top_k - k_actual, device=x_support.device, dtype=x_support.dtype)
            pad_vh = torch.zeros(
                self.top_k - k_actual,
                self.input_dim,
                device=x_support.device,
                dtype=x_support.dtype,
            )
            top_s = torch.cat([top_s, pad_s], dim=0)
            top_vh = torch.cat([top_vh, pad_vh], dim=0)

        # Add a batch dimension: [1, top_k, input_dim]
        tokens = top_vh.unsqueeze(0)

        # Pass through architecture
        x = self.token_projection(tokens)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.transformer(x)

        return x