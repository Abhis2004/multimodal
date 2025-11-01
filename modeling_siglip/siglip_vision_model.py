from typing import Tuple
import torch.nn as nn

from modeling_siglip.siglip_config import SiglipVisionConfig
from modeling_siglip.siglip_vision_transformer import SiglipVisionTransformer

class SiglipVisionModel(nn.Module):
    # Wrapper model that runs the Vision Transformer
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)  # core transformer backbone
        
    def forward(self, pixel_values) -> Tuple:
        # image tensor -> sequence of patch embeddings processed by transformer
        return self.vision_model(pixel_values=pixel_values)
