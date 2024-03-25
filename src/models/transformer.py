from typing import Any

import torch
from torch import nn
from einops import rearrange

from src.noises.base import do_not_add_noise


class PositionalEncoding(nn.Module):
    """This class implements Positional Encoding.

    It adds an additional embedding to the input embedding corresponding to the output
    parameters which can be learnt.

    Args:
        num_patches (int): The number of patches.
        dim (int): The embedding dimension.
    """

    def __init__(self, num_patches: int, dim: int) -> None:
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate zero embedding to the input
        x = torch.cat(
            (x, torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)), dim=1)
        x = x + self.pos_embedding
        return x


class PreNorm(nn.Module):
    """This class implements PreNorm layer.

    It applies layer normalization before the input is passed into the layer.

    Args:
        dim (int): The embedding dimension.
    """

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Sequential):
    """This class implements FeedForward layer.

    It applies a linear transformation, followed by a non-linearity,
    followed by another linear transformation.

    By default the dropout is removed.

    Args:
        dim (int): The embedding dimension.
        hidden_dim (int): The hidden dimension in the feedforward layer.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )


class Attention(nn.Module):
    """This class implements Attention layer.

    It applies scaled dot-product attention.

    By default the dropout is removed.

    Args:
        dim (int): The embedding dimension.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    """This class implements Transformer without the embedding and output layer.

    Args:
        dim (int): The embedding dimension.
        depth (int): The number of layers.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        mlp_dim (int): The hidden dimension in the feedforward layer.
    """

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(
                            dim, heads=heads, dim_head=dim_head)),
                        PreNorm(dim, FeedForward(dim, mlp_dim)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    """This class implements Transformer together with the embedding and output layer for sentiment classification.

    The dropout has been purposely removed from the transformer and we
    add it manually if we explore activation noise.

    Args:
        input_size (int): Number of input tokens.
        output_size (int): Number of output classes.
        dim (int): The embedding dimension.
        depth (int): The depth of the transformer.
        heads (int): The number of heads in the transformer.
        mlp_dim (int): The hidden dimension in the feedforward layer.
        dim_head (int): The dimension of the head.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding(input_size[0], dim)
        self.transformer = TransformerEncoder(
            dim, depth, heads, dim_head, mlp_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, output_size))
        do_not_add_noise(self.output[-1])
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output(x)
        return x
