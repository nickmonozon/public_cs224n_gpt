import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    key: tensor of shape [bs, num_heads, seq_len, head_dim]
    query: tensor of shape [bs, num_heads, seq_len, head_dim]
    value: tensor of shape[bs, num_heads, seq_len, head_dim]
    attention_mask: tensor of shape [bs, 1, 1, seq_len]
    """
    # Compute raw attention scores by taking the dot product between queries and keys
    # key.transpose(-1, -2) =  [bs, heads, seq_len, head_dim] -> [bs, heads, head_dim, seq_len]
    scores = torch.matmul(query, key.transpose(-1, -2))

    # Scale the attention scores by sqrt(head_dim)
    scaling_factor = pow(self.attention_head_size, 0.5)
    scores = scores / scaling_factor

    # Incorporate the provided attention mask (e.g., to mask padding tokens)
    scores = scores + attention_mask

    # Create a causal mask to prevent attending to future tokens
    # This mask is a boolean tensor with True in positions that should be masked
    seq_length = query.size(-2)
    causal_mask =  causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(scores.device)
    
    # Apply causal mask
    # set scores for future positions to -inf
    scores = scores.masked_fill(causal_mask, float('-inf'))

    # Apply softmax to get attn probabilities
    attn_probs = torch.nn.functional.softmax(scores, dim=-1)

    # Apply dropout to the attn probabilities
    attn_probs = self.dropout(attn_probs)

    # Multiply the attn probabilities with the value tensor to get the weighted sum
    attn_out = torch.matmul(attn_probs, value)

    # Rearrange the tensor from [bs, num_heads, seq_len, head_dim] -> to [batch_size, seq_len, hidden_size]
    # Here, hidden_size = num_heads * head_dim.
    attn_out = rearrange(attn_out, 'b h t d -> b t (h d)')
    
    return attn_out


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
