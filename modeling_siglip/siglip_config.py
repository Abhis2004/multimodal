class SiglipVisionConfig:
    # Holds all hyperparameters for the SigLIP Vision Transformer model
    def __init__(
        self,
        hidden_size=768,              # size of hidden embeddings
        intermediate_size=3072,       # size of MLP layer inside transformer block
        num_hidden_layers=12,         # number of transformer encoder layers
        num_attention_heads=12,       # number of attention heads
        num_channels=3,               # input channels (3 for RGB)
        image_size=224,               # input image resolution
        patch_size=16,                # patch size for splitting image
        layer_norm_eps=1e-6,          # epsilon for layer norm stability
        attention_dropout=0.0,        # dropout in attention
        num_image_tokens: int = None, # optional, total number of image tokens
        **kwargs                      # extra unused args
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens