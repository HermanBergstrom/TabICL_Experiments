import torch
import torch.nn as nn


class ParallelQueryDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        target_rows: int = 768,
        output_cols: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if target_rows <= 0:
            raise ValueError("target_rows must be > 0")
        if output_cols <= 0:
            raise ValueError("output_cols must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.target_rows = target_rows

        # 1. Learned query tokens, one per output row in W.
        self.query_embeds = nn.Parameter(torch.randn(1, target_rows, embed_dim) * 0.02)

        # 2. Decoder lets query rows attend to spectral memory.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 3. Map each decoded row embedding to its output-dimension row values.
        self.row_projector = nn.Linear(embed_dim, output_cols)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: Encoder token sequence. Shape: [1, top_k, embed_dim]
        Returns:
            W: The generated projection matrix. Shape: [target_rows, output_cols]
        """

        #print("Decoder received memory of shape:", memory.shape)

        if memory.ndim != 3:
            raise ValueError(f"Expected memory shape [B, top_k, embed_dim], got {tuple(memory.shape)}")

        # Expand queries to match batch size (which is 1 for episodic hypernetworks)
        batch_size = memory.shape[0]
        tgt = self.query_embeds.expand(batch_size, -1, -1)

        # Pass through decoder (queries look at each other, and cross-attend to memory)
        out = self.decoder(tgt=tgt, memory=memory)

        # Project to final column dimension
        W_batch = self.row_projector(out)

        # Squeeze the batch dimension to return the actual matrix
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1 for episodic generation, got {batch_size}")
        W = W_batch.squeeze(0)

        return W