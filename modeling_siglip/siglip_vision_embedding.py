import torch
import torch.nn as nn

from modeling_siglip.siglip_config import SiglipVisionConfig

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # embedding dimension for patches
        self.image_size = config.image_size  # input image size
        self.patch_size = config.patch_size  # patch size for splitting image
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2  # total number of patches
        self.num_positions = self.num_patches  # positions = number of patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # learnable positional embeddings
        
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),  # position indices for each patch
            persistant=False,  # non-trainable buffer
        )
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # Where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)  # flatten height & width into patch dimension
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)  # switch dimensions to match transformer input
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)  # add positional info
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
