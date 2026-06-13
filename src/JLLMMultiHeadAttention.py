"""
JLLM Multi-Head Attention with RoPE
====================================
Implements attention with Rotary Position Embedding (RoPE).
"""

import math
import torch
import torch.nn.functional as F
from JLLMTransformerLayer import *

def apply_causal_mask(scores, seq_len, total_kv_len):
    """
    Apply causal mask for autoregressive models.

    Args:
        scores: Attention scores [batch, heads, seq, kv_seq]
        seq_len: Current sequence length
        total_kv_len: Total KV cache length

    Returns:
        Masked scores with -inf for future positions
    """
    if seq_len <= 1:
        return scores

    # Create causal mask
    mask = torch.full((seq_len, total_kv_len), float('-inf'), device=scores.device)
    # Upper triangular part (excluding diagonal) should be masked
    mask = torch.triu(mask, diagonal=total_kv_len - seq_len + 1)
    return scores + mask.unsqueeze(0).unsqueeze(0)


def compute_rope_freqs(head_dim, rope_theta, device):
    """
    Compute RoPE frequency tensor.

    RoPE uses angles based on position and dimension:
        theta_i = rope_theta ^ (-2i/d)

    Args:
        head_dim: Dimension per head (typically 128)
        rope_theta: Base frequency (e.g., 10000 for Llama, 1000000 for Qwen)
        device: Device to create tensor on

    Returns:
        Frequency tensor [head_dim // 2]
    """
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    return freqs


def apply_rope_to_position(x, positions, freqs):
    """
    Apply Rotary Position Embedding to input tensor.

    Args:
        x: Input tensor [batch, seq, heads, head_dim]
        positions: Position indices [seq]
        freqs: RoPE frequencies [head_dim // 2]

    Returns:
        RoPE-rotated tensor
    """
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    # angles: [seq, head_dim//2]
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    cos = angles.cos()[None, :, None, :]  # [1, seq, 1, head_dim//2]
    sin = angles.sin()[None, :, None, :]  # [1, seq, 1, head_dim//2]

    # Split x into halves along last dimension
    x1, x2 = x[..., :half_dim], x[..., half_dim:]  # each [batch, seq, heads, head_dim//2]

    # Rotate BOTH halves
    # First half: x1 * cos + (-x2) * sin
    # Second half: x2 * cos + x1 * sin
    x1_out = x1 * cos + (-x2) * sin
    x2_out = x2 * cos + x1 * sin

    return torch.cat([x1_out, x2_out], dim=-1)



# =============================================================================
# JLLM Multi-Head Attention Class
# =============================================================================

class JLLMMultiHeadAttention:
    """Multi-Head Attention with GQA and RoPE."""

    def __init__(self, loader, cacher, rope_theta=10000.0):
        self.loader = loader
        self.cacher = cacher
        self.rope_theta = rope_theta
        self.hidden_size = 4096
        self.num_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128

    def set_architecture(self, hidden_size, num_heads, num_kv_heads, head_dim=128):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
    def forward(self, layer_idx, x, weights=None, raw_flat_kv=None, device="cuda", current_seq_len=1):
        """
        Forward pass for one layer. 
        current_seq_len: Total length of the sequence including the current input chunk.
        Returns (output, (k_cache, v_cache)).
        """
        batch_size, seq_len, _ = x.shape

        # Use pre-loaded weights if provided, otherwise load from layer cache file
        if weights is None:
            # Preload layer into dictTensor then get expanded GPU weights            
            weights = self.loader.tensorManager.get_layer_weights(layer_idx, device)

        # Extract individual weights from the weights dict
        q_proj = weights.get(f"layers.{layer_idx}.self_attn.q_proj.weight")
        k_proj = weights.get(f"layers.{layer_idx}.self_attn.k_proj.weight")
        v_proj = weights.get(f"layers.{layer_idx}.self_attn.v_proj.weight")
        o_proj = weights.get(f"layers.{layer_idx}.self_attn.o_proj.weight")
        q_bias = weights.get(f"layers.{layer_idx}.self_attn.q_proj.bias")
        k_bias = weights.get(f"layers.{layer_idx}.self_attn.k_proj.bias")
        v_bias = weights.get(f"layers.{layer_idx}.self_attn.v_proj.bias")

        # Ensure passed weights match target execution device
        q_proj = q_proj.to(device) if q_proj is not None else None
        k_proj = k_proj.to(device) if k_proj is not None else None
        v_proj = v_proj.to(device) if v_proj is not None else None
        o_proj = o_proj.to(device) if o_proj is not None else None
        if q_bias is not None: q_bias = q_bias.to(device)
        if k_bias is not None: k_bias = k_bias.to(device)
        if v_bias is not None: v_bias = v_bias.to(device)

        # Apply Linear Projection + Bias
        target_dtype = q_proj.dtype if q_proj is not None else torch.float16
        x_cast = x.to(target_dtype)
        
        q_flat = torch.matmul(x_cast, q_proj.T)
        if q_bias is not None: q_flat += q_bias
        k_flat = torch.matmul(x_cast, k_proj.T)
        if k_bias is not None: k_flat += k_bias
        v_flat = torch.matmul(x_cast, v_proj.T)
        if v_bias is not None: v_flat += v_bias

        # Reshape for attention heads and enforce memory contiguity
        q = q_flat.view(batch_size, seq_len, self.num_heads, self.head_dim).contiguous()
        k = k_flat.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).contiguous()
        v = v_flat.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).contiguous()

        del q_proj, k_proj, v_proj
        if q_bias is not None: del q_bias
        if k_bias is not None: del k_bias
        if v_bias is not None: del v_bias

        # Compute accurate absolute positions for RoPE mapping
        seq_start_pos = current_seq_len - seq_len
        positions = torch.arange(seq_start_pos, current_seq_len, dtype=torch.int64, device=device)

        # Compute RoPE frequencies and apply to Q and K
        freqs = compute_rope_freqs(self.head_dim, self.rope_theta, device)
        q = apply_rope_to_position(q, positions, freqs)
        k = apply_rope_to_position(k, positions, freqs)

        # Cast to uniform low-precision storage layout
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        k_cache = k.clone()
        v_cache = v.clone()

        # Concatenate with history KV if available
        if raw_flat_kv is not None and raw_flat_kv[0] is not None:
            k_history, v_history = raw_flat_kv
            k_full = torch.cat([k_history, k], dim=1)
            v_full = torch.cat([v_history, v], dim=1)
        else:
            k_full = k
            v_full = v

        # Compute multi-head context tensor
        context = compute_attention(q, k_full, v_full, self.num_heads, self.num_kv_heads, self.head_dim)

        # Map memory types back smoothly for final projection matrix
        context = context.to(o_proj.dtype)
        output = torch.matmul(context, o_proj.T)
        del o_proj

        return output, (k_cache, v_cache)


def swiglu_mlp(normed_t, loader, layer_idx, device):
    """
    SwiGLU MLP implementation.

    SwiGLU = SiLU(gate) * up
    where SiLU(x) = x * sigmoid(x)

    Args:
        normed_t: Normalized input [batch, seq, hidden]
        loader: JLLMLoader instance
        layer_idx: Layer index
        device: Device for computation

    Returns:
        MLP output [batch, seq, hidden]
    """
    # Load weights from layer cache file
    weights = loader.tensorManager.get_layer_weights(layer_idx, device)

    gate_proj = weights.get(f"layers.{layer_idx}.mlp.gate_proj.weight")
    up_proj = weights.get(f"layers.{layer_idx}.mlp.up_proj.weight")
    down_proj = weights.get(f"layers.{layer_idx}.mlp.down_proj.weight")

    hidden_dim = normed_t.shape[-1]

    # Weight shapes for Llama-style models:
    # gate_proj: [intermediate_size, hidden]
    # up_proj: [intermediate_size, hidden]
    # down_proj: [hidden, intermediate_size]
    inter_size = gate_proj.shape[0]

    gate_t = gate_proj.view(inter_size, hidden_dim).to(device) if gate_proj is not None else None
    up_t = up_proj.view(inter_size, hidden_dim).to(device) if up_proj is not None else None
    down_t = down_proj.view(hidden_dim, inter_size).to(device) if down_proj is not None else None

    del gate_proj, up_proj, down_proj

    # SwiGLU computation
    gate_out = torch.matmul(normed_t, gate_t.T) if gate_t is not None else None
    up_out = torch.matmul(normed_t, up_t.T) if up_t is not None else None

    # SiLU activation: x * sigmoid(x)
    activated = F.silu(gate_out) if gate_out is not None else None

    # Element-wise multiply: silu(gate) * up
    swiglu_out = activated * up_out if (activated is not None and up_out is not None) else None

    # Down projection
    mlp_out = torch.matmul(swiglu_out, down_t.T) if down_t is not None else None

    return mlp_out
