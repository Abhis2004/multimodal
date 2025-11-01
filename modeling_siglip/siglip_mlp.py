import torch
import torch.nn as nn

from modeling_siglip.siglip_config import SiglipVisionConfig

class SiglipMLP(nn.Module):
    # Implements the MLP (feed-forward) block used inside each transformer layer
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config  # store model configuration for reference
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # first fully-connected layer (expands features)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # second fully-connected layer (projects back)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Applies two linear transformations with GELU activation between them

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermidiate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden states: [Batch_Size, Num_Patches, Intermidiate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')  # non-linear activation for better expressiveness
        # [Batch_Size, Num_Patches, Intermidiate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states  # returns transformed embeddings after MLP block