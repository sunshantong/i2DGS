import torch
from torch import nn


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """
    Apply sinusoidal positional encoding to the last dimension of tensor x.

    Args:
        x: Tensor of shape (..., C)
        num_freqs: Number of frequency bands

    Returns:
        Tensor of shape (..., 2 * num_freqs * C)
    """
    if num_freqs <= 0:
        # Nothing to encode
        return x.new_empty(*x.shape[:-1], 0)

    # Create frequency bands on the same device and dtype as x
    freq_bands = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype))
    # (..., C, num_freqs)
    x_expanded = x.unsqueeze(-1) * freq_bands
    # (..., C * num_freqs)
    x_enc = torch.cat([x_expanded.sin(), x_expanded.cos()], dim=-1)
    # Reshape to (..., 2 * num_freqs * C)
    C = x.shape[-1]
    return x_enc.view(*x.shape[:-1], 2 * num_freqs * C)


class ResidualMLPBlock(nn.Module):
    """
    A small MLP block with a residual (skip) connection and LayerNorm.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.norm1(out)

        out = self.fc2(out)
        out = self.drop2(out)
        out = self.norm2(out)

        # Residual connection
        out = out + identity
        out = self.act2(out)
        return out


class RayPreprocessor(nn.Module):
    """
    Preprocess ray samples by concatenating positional, view, and color encodings,
    followed by a deeper residual MLP for robust feature extraction.
    """
    def __init__(
        self,
        viewpe: int = 8,
        pospe: int = 8,
        rgbpe: int = 6,
        featureC: int = 128,
        fea_output: int = 128,
        num_hidden_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.viewpe = viewpe
        self.pospe = pospe
        self.rgbpe = rgbpe

        # Compute the dimensionality of the concatenated input
        base_dim = 3  # pts, viewdirs, rgb each have last-dim=3
        enc_dim = (
            base_dim + 2 * pospe * base_dim
            + base_dim + 2 * viewpe * base_dim
            + base_dim + 2 * rgbpe * base_dim
        )
        self.input_dim = enc_dim

        # First linear layer to project to featureC
        self.in_fc = nn.Linear(self.input_dim, featureC)
        self.in_act = nn.ReLU(inplace=True)
        self.in_drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Stack of residual MLP blocks
        self.hidden_blocks = nn.ModuleList([
            ResidualMLPBlock(featureC, featureC, dropout=dropout)
            for _ in range(num_hidden_blocks)
        ])

        # Final projection layer
        self.out_fc = nn.Linear(featureC + self.input_dim, fea_output)

    def forward(
        self,
        pts: torch.Tensor,
        viewdirs: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        # Validate inputs
        assert pts.dim() >= 2 and pts.shape[-1] == 3, \
            f"Expected pts (..., 3), got {tuple(pts.shape)}"
        assert viewdirs.shape == pts.shape, \
            f"viewdirs shape {tuple(viewdirs.shape)} must match pts shape {tuple(pts.shape)}"
        assert rgb.shape == pts.shape, \
            f"rgb shape {tuple(rgb.shape)} must match pts shape {tuple(pts.shape)}"

        # Build input list
        parts = [pts]
        parts.append(positional_encoding(pts, self.pospe))
        parts.append(viewdirs)
        parts.append(positional_encoding(viewdirs, self.viewpe))
        parts.append(rgb)
        parts.append(positional_encoding(rgb, self.rgbpe))

        # Concatenate along the last dimension
        mlp_input = torch.cat(parts, dim=-1)  # shape: (..., input_dim)

        # Project input to featureC
        x = self.in_fc(mlp_input)
        x = self.in_act(x)
        x = self.in_drop(x)

        # Pass through residual blocks
        for block in self.hidden_blocks:
            x = block(x)

        # Concatenate skip connection from input and project to fea_output
        out = self.out_fc(torch.cat([x, mlp_input], dim=-1))
        return out
