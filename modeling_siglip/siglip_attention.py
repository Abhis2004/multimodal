from typing import Optional, Tuple
import torch
import torch.nn as nn

from modeling_siglip.siglip_config import SiglipVisionConfig

class SiglipAttention(nn.Module):
    # Implements multi-head self-attention mechanism for the SigLIP Vision Transformer
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config  # store model configuration for reference
        self.embed_dim = config.hidden_size  # total embedding dimension
        self.num_heads = config.num_attention_heads  # number of parallel attention heads
        self.head_dim = self.embed_dim // self.num_heads  # dimension per attention head
        self.scale = self.head_dim ** -0.5  # Equivalent to 1 / sqrt(self.head_dim), used for attention scaling
        self.dropout = config.attention_dropout  # dropout rate for attention weights

        # Linear projections for keys, values, and queries
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Output projection after attention is combined across heads
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Performs multi-head self-attention on input hidden states

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim] {Embed_Dim = 1024}
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose for multi-head attention computation
        # query_states: [Batch_Size, Num_Patches, Num_Heads, Head_Dim] {Num_Heads, Head_Dim = 8, 128}
        # After transpose: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        # We transpose to bring 'Num_Heads' before 'Num_Patches' so that each head can process
        # its attention independently and in parallel during matrix multiplication.
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate the attention using the formal Q * K^T/sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)
        # Scaling stabilizes gradients by keeping the dot product magnitudes small

        # Ensure tensor shapes are valid before proceeding
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training to regularize attention
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)
        # This produces a weighted sum of value vectors for each patch and head

        # Sanity check for shape consistency
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Transposing back merges head outputs per patch for final projection

        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # Flatten all heads into a single embedding per patch

        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)
        # Final linear projection to mix information across heads

        return attn_output, attn_weights  # return both output features and attention maps