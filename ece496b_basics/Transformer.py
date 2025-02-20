import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import typing

# Define which device this LM is running on
DEVICE = torch.device("cpu")

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        """
        Root Mean Square Layer Normalization

        Args:
            d_model (int): Dimensionality of input embeddings.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # Learnable gain parameter

    def set_weights_from_dict(self, d):
        # manually assign the weights, if the keys match, we can use load_state_dict() instead
        self.weight.data = d['weight']

    def forward(self, x):
        """
        Apply RMS normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: RMS normalized output of the same shape as input.
        """
        # Compute RMS value across the last dimension (feature dimension)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize input and apply gain
        return self.weight * (x / rms)

# Implementing the GELU activation function
def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit (GELU) activation function."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# Implementing the Position-Wise Feed-Forward Network (FFN)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Position-wise feed-forward network as described in the assignment.
        
        Args:
            d_model (int): Input and output dimensionality.
            d_ff (int): Inner layer dimensionality, typically 4 * d_model.
        """
        super(PositionwiseFeedForward, self).__init__()
        # Linear layers without bias
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def load_state_dict(self, d):
        self.w1.weight.data = d['w1.weight']
        self.w2.weight.data = d['w2.weight']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        return self.w2(gelu(self.w1(x)))

# Implementing softmax function
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes the softmax of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int, optional): The dimension to apply the softmax. Defaults to the last dimension.

    Returns:
        torch.Tensor: Softmax normalized tensor.
    """
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    exp_val = torch.exp(x - max_val)
    return exp_val / exp_val.sum(dim=dim, keepdim=True)

# Implementing scaled dot product attention function
def scaled_dot_product_attention(
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    p_drop: float = 0.0,
) -> torch.Tensor:
    """
    Computes the scaled dot-product attention.

    Args:
        Q (torch.Tensor): Query matrix of shape (batch_size, seq_len, d_k).
        K (torch.Tensor): Key matrix of shape (batch_size, seq_len, d_k).
        V (torch.Tensor): Value matrix of shape (batch_size, seq_len, d_v).
        mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len). Defaults to None.
        pdrop

    Returns:
        torch.Tensor: Output of attention mechanism, shape (batch_size, seq_len, d_v).
    """
    # pre_scores: (B, ..., S, K) @ (B, ..., K, S) -> (B, ..., S, S)
    pre_scores = K @ Q.transpose(-1, -2) / math.sqrt(K.shape[-1])
    # apply mask
    if mask is not None:
        pre_scores = pre_scores.masked_fill(mask, -torch.inf)
    # scores: B x ... x S x S
    scores = softmax(pre_scores, -1)
    if p_drop > 0:
        scores = torch.nn.functional.dropout(scores, p_drop)
    return scores @ V

class CausalMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_k = d_model // num_heads
        
        # Linear layers for projecting queries, keys, and values
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, seq_len, _ = x.size()
        
        # Generate Q, K, V matrices
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Create a causal mask to prevent attention to future tokens
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Apply scaled dot-product attention with dropout
        attention_output = scaled_dot_product_attention(q, k, v, mask=causal_mask, p_drop=self.attn_pdrop)

        # Reassemble the multi-head attention outputs
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        return self.output_proj(attention_output)

    def load_weights_from_dict(self, state_dict: dict):
        """
        Load weights from a custom state dictionary into the attention module.
        
        Args:
            state_dict (dict): State dictionary with the following keys:
                - 'q_heads.{N}.weight', 'k_heads.{N}.weight', 'v_heads.{N}.weight':
                  Individual head weights for query, key, and value projections.
                  N is the head index (0 to num_heads - 1).
                  Shape: (d_k, d_model)
                - 'output_proj.weight':
                  Output projection weight matrix.
                  Shape: (d_model, d_model)
        """
        for i in range(self.num_heads):
            self.query_proj.weight.data[i * self.d_k:(i + 1) * self.d_k] = state_dict[f'q_heads.{i}.weight']
            self.key_proj.weight.data[i * self.d_k:(i + 1) * self.d_k] = state_dict[f'k_heads.{i}.weight']
            self.value_proj.weight.data[i * self.d_k:(i + 1) * self.d_k] = state_dict[f'v_heads.{i}.weight']
        self.output_proj.weight.data = state_dict['output_proj.weight']

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float = 0.0,
        residual_pdrop: float = 0.0,
        is_parallel: bool = False,
        norm_type: typing.Literal["post", "pre", "none"] = "pre"
    ) -> None:
        """
        Implements a pre-norm Transformer block.

        Args:
            d_model (int): Dimensionality of Transformer input/output.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the position-wise feed-forward layer.
            attn_pdrop (float): Dropout rate for attention probabilities.
            residual_pdrop (float): Dropout rate for residual connections.
            is_parallel (bool): If True, runs attention and FFN in parallel.
            norm_type (str): "pre" for pre-norm, "post" for post-norm, "none" for no normalization.
        """
        super().__init__()

        self.is_parallel = is_parallel
        self.norm_type = norm_type

        # Normalization layers
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Multi-head self-attention module
        self.attention = CausalMultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        
        # Position-wise feed-forward network
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        
        # Dropout layer for residual connections
        self.dropout = nn.Dropout(residual_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        if self.is_parallel:
            assert self.norm_type == "pre", "Parallel execution only supports pre-norm"
            return x + self.dropout(self.attention(self.norm1(x))) + self.dropout(self.ffn(self.norm2(x)))

        if self.norm_type == "post":
            y = self.norm1(x + self.dropout(self.attention(x)))
            return self.norm2(y + self.dropout(self.ffn(y)))

        if self.norm_type == "none":
            return x + self.dropout(self.attention(x)) + self.dropout(self.ffn(x))

        # Standard pre-norm execution
        y = x + self.dropout(self.attention(self.norm1(x)))
        return y + self.dropout(self.ffn(self.norm2(y)))

    def load_weights_from_dict(self, state_dict: dict):
        """
        Loads weights from a provided state dictionary.

        Args:
            state_dict (dict): Dictionary containing model weights.
        """
        if 'attention' in state_dict:
            self.attention.load_from_state_dict(state_dict['attention'])
        if 'norm1' in state_dict:
            self.norm1.load_from_state_dict(state_dict['norm1'])
        if 'ffn' in state_dict:
            self.ffn.load_from_state_dict(state_dict['ffn'])
        if 'norm2' in state_dict:
            self.norm2.load_from_state_dict(state_dict['norm2'])    

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
        is_parallel: bool = False,
        norm_type: typing.Literal["post", "pre", "none"] = "pre"
    ) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = torch.nn.Dropout(residual_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(
            self.token_embeddings(x)
            + self.position_embeddings(torch.arange(x.shape[1]).to(DEVICE)).unsqueeze(0)
        )
        for block in self.layers:
            x = block(x)
        return self.lm_head(self.ln_final(x))
    def load_weights_from_dict(self, state_dict: dict):
        """
        Loads model weights from a state dictionary.

        Args:
            state_dict (dict): Dictionary containing model weights.
        """
        # Dict weights provided in test is not properly named? Took forever to fix
        key_mapping = {
          "token_emb.weight": "token_embeddings.weight",
          "position_emb": "position_embeddings",
          "final_norm.weight": "ln_final.weight",
          "final_norm.bias": "ln_final.bias",
          "output_layer.weight": "lm_head.weight",
        }

        # Create a new state dictionary with mapped keys
        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key in key_mapping:
                mapped_state_dict[key_mapping[key]] = value
            else:
                mapped_state_dict[key] = value  # Keep the key unchanged if not in mapping

        # Load transformer block weights
        for i, layer in enumerate(self.layers):
            layer_state_dict = {k[len(f"layers.{i}."):] : v for k, v in state_dict.items() if k.startswith(f"layers.{i}.")}
            layer.load_state_dict(layer_state_dict, strict=False)

        # Load main model weights
        self.load_state_dict(mapped_state_dict, strict=False)   
        
         