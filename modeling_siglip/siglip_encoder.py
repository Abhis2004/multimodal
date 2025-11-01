import torch
import torch.nn as nn

from modeling_siglip.siglip_config import SiglipVisionConfig
from modeling_siglip.siglip_encoder_layer import SiglipEncoderLayer

class SiglipEncoder(nn.Module):
    # Stacks multiple transformer encoder layers to form the main encoder block
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config  # store model configuration for reference

        # Create a list of transformer encoder layers
        # Each SiglipEncoderLayer performs self-attention + MLP + residual connections
        # The number of layers is defined by config.num_hidden_layers (e.g., 12 for ViT-B)
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor):
        # Performs sequential forward pass through all transformer encoder layers
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds  # initialize hidden states with patch embeddings

        for encoder_layer in self.layers:
            # Pass through each transformer layer one by one
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        # Final encoded representation after all transformer layers
        return hidden_states