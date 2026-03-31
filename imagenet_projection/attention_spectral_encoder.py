import torch
import torch.nn as nn
import time


def _synchronized_perf_counter(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _sign_correct_rows(vecs: torch.Tensor) -> torch.Tensor:
    """Flip each row so its largest-magnitude element is positive (deterministic)."""
    max_abs_indices = torch.argmax(torch.abs(vecs), dim=1)
    signs = torch.sign(vecs[torch.arange(vecs.shape[0]), max_abs_indices])
    signs[signs == 0] = 1.0
    return vecs * signs.unsqueeze(1)


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
        scale_svd_by_singular_values: bool = True,
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
        self.scale_svd_by_singular_values = bool(scale_svd_by_singular_values)

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

        vh = _sign_correct_rows(vh)

        # Truncate or Pad to top_k
        k_actual = min(self.top_k, int(svals.shape[0]))
        top_s = svals[:k_actual]
        top_vh = vh[:k_actual, :]

        # Singular value injection: normalize values and scale each component vector.
        if self.scale_svd_by_singular_values:
            s_max = torch.max(top_s).clamp_min(1e-8)
            top_vh = top_vh * (top_s / s_max).unsqueeze(1)

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


class AttentionSpectralLDAEncoder(nn.Module):
    """Encode SVD spectral tokens + LDA discriminant tokens using self-attention.

    The support set produces two kinds of tokens:
      - SVD tokens  (top_k): right-singular vectors, optionally scaled by
                             normalised singular values — same as
                             AttentionSpectralEncoder.
      - LDA tokens  (C-1):   discriminant directions from Linear Discriminant
                             Analysis, optionally scaled by normalised
                             discriminant eigenvalues.  A learnable
                             ``lda_type_embed`` vector is *added* to every LDA
                             token after the shared linear projection so the
                             transformer can distinguish the two token types.

    Output shape: [1, top_k + (C-1), embed_dim]
    where C is the number of distinct classes in y_support.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        top_k: int = 64,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        use_positional_embeddings: bool = False,
        scale_svd_by_singular_values: bool = True,
        scale_lda_by_eigenvalues: bool = True,
        lda_regularization: float = 1e-4,
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
        self.scale_svd_by_singular_values = bool(scale_svd_by_singular_values)
        self.scale_lda_by_eigenvalues = bool(scale_lda_by_eigenvalues)
        self.lda_regularization = float(lda_regularization)

        # Shared projection for both SVD and LDA tokens.
        self.token_projection = nn.Linear(self.input_dim, self.context_hidden_dim)

        # Positional embeddings only for the fixed-size SVD token block.
        self.pos_embed: nn.Parameter | None
        if self.use_positional_embeddings:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.top_k, self.context_hidden_dim) * 0.02
            )
        else:
            self.pos_embed = None

        # Learnable type embedding added to every LDA token after projection.
        self.lda_type_embed = nn.Parameter(
            torch.randn(1, 1, self.context_hidden_dim) * 0.02
        )

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

    # ------------------------------------------------------------------
    # LDA helper
    # ------------------------------------------------------------------

    def _compute_lda_vectors(
        self,
        x_centered: torch.Tensor,
        y_support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return LDA discriminant vectors and their eigenvalues.

        Uses a Cholesky-based generalised eigenvalue decomposition:
            Sb v = λ Sw v
        where Sw is the (regularised) within-class scatter and Sb is the
        between-class scatter, both normalised by N.

        Returns:
            lda_vecs:    [C-1, D]  L2-normalised discriminant directions,
                                   sorted by descending eigenvalue.
            eigenvalues: [C-1]     corresponding (non-negative) eigenvalues.
        """
        D = int(x_centered.shape[1])
        N = int(x_centered.shape[0])
        classes = torch.unique(y_support)
        C = int(classes.shape[0])

        if C < 2:
            empty = torch.zeros(0, D, device=x_centered.device, dtype=x_centered.dtype)
            return empty, empty

        n_lda = C - 1
        overall_mean = x_centered.mean(dim=0)  # [D]

        Sw = torch.zeros(D, D, device=x_centered.device, dtype=x_centered.dtype)
        Sb = torch.zeros(D, D, device=x_centered.device, dtype=x_centered.dtype)

        for c in classes:
            mask = y_support == c
            xc = x_centered[mask]
            nc = float(mask.sum())
            mc = xc.mean(dim=0)
            diff_w = xc - mc.unsqueeze(0)          # [Nc, D]
            Sw = Sw + diff_w.T @ diff_w
            diff_b = (mc - overall_mean).unsqueeze(1)  # [D, 1]
            Sb = Sb + nc * (diff_b @ diff_b.T)

        # Normalise by N and regularise Sw.
        Sw = Sw / N + self.lda_regularization * torch.eye(
            D, device=x_centered.device, dtype=x_centered.dtype
        )
        Sb = Sb / N

        try:
            # Cholesky decomposition of Sw: Sw = L L^T
            L = torch.linalg.cholesky(Sw)
            L_inv = torch.linalg.inv(L)
            # Symmetrised transformed problem: M = L^{-1} Sb L^{-T}
            M = L_inv @ Sb @ L_inv.T
            M = (M + M.T) * 0.5  # enforce symmetry for eigh

            # eigh returns eigenvalues in ascending order.
            eigenvalues, eigenvectors = torch.linalg.eigh(M)

            # Keep top C-1, flip to descending order.
            eigenvalues = eigenvalues[-n_lda:].flip(0).clamp_min(0.0)
            eigenvectors = eigenvectors[:, -n_lda:].flip(1)  # [D, C-1]

            # Back-transform to original feature space: v = L^{-T} u
            lda_vecs = (L_inv.T @ eigenvectors).T  # [C-1, D]

        except torch.linalg.LinAlgError:
            lda_vecs = torch.zeros(n_lda, D, device=x_centered.device, dtype=x_centered.dtype)
            eigenvalues = torch.zeros(n_lda, device=x_centered.device, dtype=x_centered.dtype)
            return lda_vecs, eigenvalues

        # L2-normalise each discriminant direction.
        norms = lda_vecs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        lda_vecs = lda_vecs / norms

        # Deterministic sign correction.
        lda_vecs = _sign_correct_rows(lda_vecs)

        return lda_vecs, eigenvalues

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_support: torch.Tensor, y_support: torch.Tensor) -> torch.Tensor:
        """Encode one labelled support set into token memory.

        Args:
            x_support: [Ns, D] support features.
            y_support: [Ns]    integer class labels.

        Returns:
            [1, top_k + (C-1), embed_dim] token sequence.
        """
        if x_support.ndim != 2:
            raise ValueError(f"Expected x_support shape [Ns, D], got {tuple(x_support.shape)}")
        if x_support.shape[0] == 0:
            raise ValueError("x_support must have at least one row")
        if int(x_support.shape[1]) != self.input_dim:
            raise ValueError(
                f"x_support has feature dim {int(x_support.shape[1])}, expected {self.input_dim}"
            )
        if y_support.shape[0] != x_support.shape[0]:
            raise ValueError("x_support and y_support must have the same number of rows")

        x_centered = x_support - x_support.mean(dim=0, keepdim=True)

        # ---- SVD tokens ------------------------------------------------
        t0 = _synchronized_perf_counter(x_support.device)
        _, svals, vh = torch.linalg.svd(x_centered, full_matrices=False)
        t1 = _synchronized_perf_counter(x_support.device)

        vh = _sign_correct_rows(vh)

        k_actual = min(self.top_k, int(svals.shape[0]))
        top_s = svals[:k_actual]
        top_vh = vh[:k_actual, :]

        if self.scale_svd_by_singular_values:
            s_max = top_s.max().clamp_min(1e-8)
            top_vh = top_vh * (top_s / s_max).unsqueeze(1)

        if k_actual < self.top_k:
            pad = torch.zeros(
                self.top_k - k_actual, self.input_dim,
                device=x_support.device, dtype=x_support.dtype,
            )
            top_vh = torch.cat([top_vh, pad], dim=0)

        svd_tokens = top_vh.unsqueeze(0)  # [1, top_k, D]

        # ---- LDA tokens ------------------------------------------------
        t2 = _synchronized_perf_counter(x_support.device)
        lda_vecs, lda_eigs = self._compute_lda_vectors(x_centered, y_support)
        t3 = _synchronized_perf_counter(x_support.device)

        self.last_profile = {
            "eigendecomp_ms": float((t1 - t0) * 1000.0),
            "lda_ms": float((t3 - t2) * 1000.0),
        }

        if lda_vecs.shape[0] > 0 and self.scale_lda_by_eigenvalues:
            eig_max = lda_eigs.max().clamp_min(1e-8)
            lda_vecs = lda_vecs * (lda_eigs / eig_max).unsqueeze(1)

        lda_tokens = lda_vecs.unsqueeze(0)  # [1, C-1, D]

        # ---- Project, embed, and combine --------------------------------
        svd_x = self.token_projection(svd_tokens)   # [1, top_k, embed_dim]
        if self.pos_embed is not None:
            svd_x = svd_x + self.pos_embed

        if lda_tokens.shape[1] > 0:
            lda_x = self.token_projection(lda_tokens)   # [1, C-1, embed_dim]
            lda_x = lda_x + self.lda_type_embed          # broadcast [1, 1, embed_dim]
            tokens = torch.cat([svd_x, lda_x], dim=1)   # [1, top_k + C-1, embed_dim]
        else:
            tokens = svd_x

        return self.transformer(tokens)