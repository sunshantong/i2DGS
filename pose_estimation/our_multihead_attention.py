import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def scaled_attention_product(q, k, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    attention = torch.nn.functional.softmax(attn_logits, dim=-1)
    return attention

    # positive_attn_logits = torch.nn.functional.elu(attn_logits) + 1.0
    # attention = torch.multiply(
    #     positive_attn_logits,
    #     positive_attn_logits.shape[-2]
    #     / (
    #         torch.sum(
    #             positive_attn_logits,
    #             dim=(-1, -2),
    #         )
    #         + 1.0e-4
    #     ),
    # )
    # return attention

# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert (
        mask.ndim > 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, ray_fea_size, img_fea_size, embed_dim, num_heads=1, learnable_temperature=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.learnable_temperature = learnable_temperature

        self.q_proj = nn.Linear(img_fea_size, embed_dim)
        self.k_proj = nn.Linear(ray_fea_size, embed_dim)

        if self.learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(math.sqrt(self.head_dim)))
        else:
            self.register_buffer('temperature', torch.tensor(math.sqrt(self.head_dim)))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)

    def forward(self, img_features, ray_features, mask=None):
        if mask is not None:
            mask = expand_mask(mask)

        q = self.q_proj(img_features)  # shape: (N_img, D)
        k = self.k_proj(ray_features)  # shape: (N_ray, D)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))  # (N_img, N_ray)
        attn_logits = attn_logits / self.temperature

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        attention = F.softmax(attn_logits, dim=-1)
        return attention
