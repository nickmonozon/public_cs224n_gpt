from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    # Apply the linear transformation to the sub-layer output
    transformed = dense_layer(output)  # W * ouput
    
    # Apply dropout to the transformed output
    transformed = dropout(transformed)
    
    # Add the residual connection (original input) to the transformed output
    return input + transformed


  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """

    # ---- Self-Attention Sub-Layer ----
    # Apply layer norm to the input
    normed_hidden_states = self.attention_layer_norm(hidden_states)
    
    # Compute multi-headed self-attention using normed hidden states
    attention_out = self.self_attention(normed_hidden_states, attention_mask)
    
    # Combine attention output with original input (residual connection)
    # Using helper, we apply a linear transformation (attention_dense) and dropout (attention_dropout)
    # Then add to original hidden_states
    hidden_states = self.add(
        input=hidden_states,
        output=attention_out,
        dense_layer=self.attention_dense,
        dropout=self.attention_dropout
    )
    
    # ---- Feed-Forward Sub-Layer ----
    # Apply layer norm to hidden states
    normed_hidden_states = self.out_layer_norm(hidden_states)
    
    # Pass through intermediate dense layer and apply activation
    interm_out = self.interm_dense(normed_hidden_states)
    interm_out = self.interm_af(interm_out)
    
    # Use add helper again as above
    hidden_states = self.add(
        input=hidden_states,
        output=interm_out,
        dense_layer=self.out_dense,
        dropout=self.out_dropout
    )
    
    return hidden_states

