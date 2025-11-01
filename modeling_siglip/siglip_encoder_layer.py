import torch
import torch.nn as nn

from modeling_siglip.siglip_attention import SiglipAttention
from modeling_siglip.siglip_config import SiglipVisionConfig
from modeling_siglip.siglip_mlp import SiglipMLP

class SiglipEncoderLayer(nn.Module):
    # Defines a single transformer encoder layer used in the SigLIP Vision Transformer
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # embedding dimension for all operations
        self.self_attn = SiglipAttention(config)  # multi-head self-attention mechanism
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # normalization before attention
        self.mlp = SiglipMLP(config)  # feed-forward neural network block (MLP)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # normalization before MLP

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Performs one full transformer encoder pass: attention + MLP + residual connections

        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states  # add residual connection
        # [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states  # add residual connection from MLP block

        return hidden_states  # output encoded representation for this layer