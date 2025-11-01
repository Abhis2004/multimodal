import torch
import torch.nn as nn

from modeling_siglip.siglip_config import SiglipVisionConfig
from modeling_siglip.siglip_encoder import SiglipEncoder
from modeling_siglip.siglip_vision_embedding import SiglipVisionEmbeddings

class SiglipVisionTransformer(nn.Module):
    # Main Vision Transformer model using SigLIP config
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size  # embedding dimension for all layers
        
        self.embeddings = SiglipVisionEmbeddings(config)  # converts image to patch embeddings
        self.encoder = SiglipEncoder(config)               # stack of transformer blocks
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # final layer norm
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)        # split image into patches + embed
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)  # transformer blocks
        last_hidden_state = self.post_layernorm(last_hidden_state)    # layer norm at the end
        return last_hidden_state