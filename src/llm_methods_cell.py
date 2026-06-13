"""
LLM Methods - Cell (Token) Level
================================
Transformer layer computation for single token processing.
Used in autoregressive generation where tokens are processed one at a time.
"""

import math
import torch
import torch.nn.functional as F


# =============================================================================
# Cell-Level RMSNorm
# =============================================================================

def rms_norm_cell(x_cell, weight_t, eps=1e-5):
    """
    RMSNorm for a single token (cell).

    Args:
        x_cell: Single token hidden state [hidden_dim] or [1, hidden_dim]
        weight_t: Layer norm weight [hidden_dim]
        eps: Numerical stability

    Returns:
        Normalized tensor in same dtype as x_cell
    """
    if weight_t is None:
        return x_cell

    input_dtype = x_cell.dtype
    x_f32 = x_cell.to(torch.float32)
    weight_f32 = weight_t.to(torch.float32)

    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normalized = x_f32 * torch.rsqrt(variance + eps)

    result = normalized * weight_f32
    return result.to(input_dtype)


# =============================================================================
# Cell-Level QKV Projection
# =============================================================================

def compute_qkv_projection_cell(x_cell, q_proj, k_proj, v_proj, q_bias=None, k_bias=None, v_bias=None):
    """
    QKV projection for a single token, including biases for models like Qwen.
    """
    if x_cell.dim() == 1:
        x_cell = x_cell.unsqueeze(0)

    target_dtype = q_proj.dtype
    x_cell_f16 = x_cell.to(target_dtype)

    # Project and apply biases
    q = torch.matmul(x_cell_f16, q_proj.T) if q_proj is not None else x_cell_f16
    if q_bias is not None: q += q_bias

    k = torch.matmul(x_cell_f16, k_proj.T) if k_proj is not None else x_cell_f16
    if k_bias is not None: k += k_bias

    v = torch.matmul(x_cell_f16, v_proj.T) if v_proj is not None else x_cell_f16
    if v_bias is not None: v += v_bias

    return q, k, v


# =============================================================================
# Cell-Level Attention
# =============================================================================

def compute_attention_scores_cell(q_cell, k_cache, scale=None):
    """
    Compute attention scores for single query against cached keys.

    Args:
        q_cell: Query for current token [num_heads, head_dim]
        k_cache: Cached keys [num_kv_heads, seq_len, head_dim]
        scale: Optional scale factor

    Returns:
        Attention scores [num_heads, seq_len]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q_cell.shape[-1])

    num_heads = q_cell.shape[0]
    num_kv_heads = k_cache.shape[0]
    heads_per_kv = num_heads // num_kv_heads

    # Expand k for GQA
    k_expanded = k_cache.repeat_interleave(heads_per_kv, dim=0)

    # q: [num_heads, 1, head_dim]
    q = q_cell.unsqueeze(1)

    # scores: [num_heads, 1, seq] -> [num_heads, seq]
    scores = torch.matmul(q, k_expanded.transpose(-2, -1)).squeeze(1) * scale

    return scores


def apply_attention_softmax_cell(scores):
    """
    Softmax for single query.

    Args:
        scores: Attention scores [num_heads, seq_len]

    Returns:
        Attention weights [num_heads, seq_len]
    """
    scores_max = scores.amax(dim=-1, keepdim=True)
    attn_weights = (scores - scores_max).exp()
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
    return attn_weights


def compute_attention_output_cell(attn_weights, v_cache):
    """
    Compute attention output for single token.

    Args:
        attn_weights: Attention weights [num_heads, seq_len]
        v_cache: Cached values [num_kv_heads, seq_len, head_dim]

    Returns:
        Attention output [num_heads, head_dim]
    """
    num_heads = attn_weights.shape[0]
    num_kv_heads = v_cache.shape[0]
    heads_per_kv = num_heads // num_kv_heads

    # Expand v for GQA
    v_expanded = v_cache.repeat_interleave(heads_per_kv, dim=0)

    # attn: [num_heads, 1, seq]
    attn = attn_weights.unsqueeze(1)

    # output: [num_heads, 1, seq] @ [num_heads, seq, head_dim] -> [num_heads, 1, head_dim]
    output = torch.matmul(attn, v_expanded)

    return output.squeeze(1)


# =============================================================================
# Cell-Level MLP
# =============================================================================

def compute_mlp_projections_cell(x_cell, gate_proj, up_proj):
    target_dtype = gate_proj.dtype
    x_cell_f16 = x_cell.to(target_dtype)
    gate_out = torch.matmul(x_cell_f16, gate_proj.T) if gate_proj is not None else x_cell_f16
    up_out = torch.matmul(x_cell_f16, up_proj.T) if up_proj is not None else x_cell_f16
    return gate_out, up_out

def compute_mlp_down_projection_cell(swiglu_out, down_proj):
    return torch.matmul(swiglu_out, down_proj.T)


def silu_cell(x):
    """SiLU activation for cell."""
    return x * torch.sigmoid(x)


# =============================================================================
# Cell-Level RoPE (Rotary Position Embedding)
# =============================================================================

def compute_rope_freqs_cell(head_dim, rope_theta, device):
    """Compute RoPE frequency tensor for cell-level processing."""
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    return freqs


def apply_rope_to_position_cell_non_cell(q_cell, k_cell, position, freqs):
    """
    Apply RoPE to Q and K for a single position.

    Args:
        q_cell: Query tensor [num_heads, head_dim]
        k_cell: Key tensor [num_kv_heads, head_dim]
        position: Scalar position index
        freqs: RoPE frequencies [head_dim // 2]

    Returns:
        Tuple of (q_rotated, k_rotated) with RoPE applied
    """
    head_dim = q_cell.shape[-1]
    half_dim = head_dim // 2

    # position as tensor for broadcasting
    pos_t = torch.tensor([position], dtype=torch.float32, device=q_cell.device)

    # angles: [head_dim//2]
    angles = pos_t * freqs
    cos = angles.cos()
    sin = angles.sin()

    # Apply rotation to q
    q1, q2 = q_cell[..., :half_dim], q_cell[..., half_dim:]
    q1_out = q1 * cos + (-q2) * sin
    q2_out = q2 * cos + q1 * sin
    q_rotated = torch.cat([q1_out, q2_out], dim=-1)

    # Apply rotation to k
    k1, k2 = k_cell[..., :half_dim], k_cell[..., half_dim:]
    k1_out = k1 * cos + (-k2) * sin
    k2_out = k2 * cos + k1 * sin
    k_rotated = torch.cat([k1_out, k2_out], dim=-1)

    return q_rotated, k_rotated

def apply_rope_to_position_cell_cell(q_cell, k_cell, position, freqs):
    head_dim = q_cell.shape[-1]
    half_dim = head_dim // 2

    # Ensure position is a float tensor on the correct device
    pos_t = torch.tensor([position], dtype=torch.float32, device=q_cell.device)

    # Compute angles: shape [half_dim]
    angles = pos_t * freqs
    
    # Reshape to 4D to broadcast perfectly with [batch, 1, heads, half_dim]
    cos = angles.cos().view(1, 1, 1, half_dim)
    sin = angles.sin().view(1, 1, 1, half_dim)

    # Split Q and K along the last dimension
    q1, q2 = q_cell[..., :half_dim], q_cell[..., half_dim:]
    q1_out = q1 * cos - q2 * sin
    q2_out = q2 * cos + q1 * sin
    q_rotated = torch.cat([q1_out, q2_out], dim=-1)

    k1, k2 = k_cell[..., :half_dim], k_cell[..., half_dim:]
    k1_out = k1 * cos - k2 * sin
    k2_out = k2 * cos + k1 * sin
    k_rotated = torch.cat([k1_out, k2_out], dim=-1)

    return q_rotated, k_rotated

def compute_mlp_down_projection_cell(swiglu_out, down_proj):
    """
    MLP down projection for single token.

    Args:
        swiglu_out: SwiGLU output [intermediate_dim]
        down_proj: Down projection matrix [hidden, intermediate]

    Returns:
        Output [hidden_dim]
    """
    if swiglu_out.dim() == 1:
        swiglu_out = swiglu_out.unsqueeze(0)

    output = torch.matmul(swiglu_out, down_proj.T).squeeze(0)

    return output


# =============================================================================
# Cell-Level Residual
# =============================================================================

def residual_add_cell(x_cell, residual_cell):
    """Add residual for single cell."""
    return x_cell + residual_cell


def final_residual_add_cell(x_cell, mlp_out_cell):
    """Final residual add for single cell."""
    return x_cell + mlp_out_cell


# =============================================================================
# Complete Cell-Level Layer Forward
# =============================================================================

def transformer_layer_forward_cell(
    hidden_cell,
    residual,
    q_proj, k_proj, v_proj, o_proj,
    gate_proj, up_proj, down_proj,
    input_layernorm_weight, post_attention_layernorm_weight,
    num_heads, num_kv_heads, head_dim,
    k_cache_cell=None, v_cache_cell=None,
    rope_freqs=None, current_position=None,
    q_bias=None, k_bias=None, v_bias=None
):
    # 1. Input RMSNorm
    normed_cell = rms_norm_cell(hidden_cell, input_layernorm_weight)

    # 2. QKV Projection 
    q_cell, k_cell, v_cell = compute_qkv_projection_cell(
        normed_cell, q_proj, k_proj, v_proj, q_bias, k_bias, v_bias
    )

    # Maintain batch size to align with Prefill Cache (usually batch_size = 1)
    batch_size = hidden_cell.shape[0] if hidden_cell.dim() == 2 else 1

    # Reshape to 4D: [batch, seq_len=1, heads, head_dim]
    q_cell = q_cell.view(batch_size, 1, num_heads, head_dim)
    k_cell = k_cell.view(batch_size, 1, num_kv_heads, head_dim)
    v_cell = v_cell.view(batch_size, 1, num_kv_heads, head_dim)

    # 2b. Apply RoPE (works natively on 4D)
    if rope_freqs is not None and current_position is not None:
        q_cell, k_cell = apply_rope_to_position_cell_cell(q_cell, k_cell, current_position, rope_freqs)

    # 3. Cache Append (Both tensors are now 4D, concatenating on seq_len dim=1)
    if k_cache_cell is not None and v_cache_cell is not None:
        k_full = torch.cat([k_cache_cell, k_cell], dim=1)
        v_full = torch.cat([v_cache_cell, v_cell], dim=1)
    else:
        k_full = k_cell
        v_full = v_cell

    # 4. Attention Using Native PyTorch SDPA 
    q_attn = q_cell.transpose(1, 2)  # [batch, heads, 1, dim]
    k_attn = k_full.transpose(1, 2)  # [batch, kv_heads, seq, dim]
    v_attn = v_full.transpose(1, 2)

    # ========================================================
    # CRITICAL DTYPE ALIGNMENT FIX
    # ========================================================
    # Force query and key to match the value cache's float16 precision
    q_attn = q_attn.to(dtype=v_attn.dtype)
    k_attn = k_attn.to(dtype=v_attn.dtype)

    # Expand KV heads for GQA (Qwen uses Grouped-Query Attention)
    num_queries_per_kv = num_heads // num_kv_heads
    if num_queries_per_kv > 1:
        k_attn = k_attn.repeat_interleave(num_queries_per_kv, dim=1)
        v_attn = v_attn.repeat_interleave(num_queries_per_kv, dim=1)

    # Compute context (Now safe from dtype divergence crashes)
    context = torch.nn.functional.scaled_dot_product_attention(
        q_attn, k_attn, v_attn, is_causal=False
    )
    
    # Reshape back to [batch, hidden_dim]
    context = context.transpose(1, 2).contiguous().reshape(batch_size, num_heads * head_dim)

    # 5. Output projection + Residual Add
    attn_output_proj = torch.matmul(context, o_proj.T) if o_proj is not None else context
    
    # Ensure residual matches dimensionality before adding
    if residual.dim() == 1 and attn_output_proj.dim() == 2:
        residual = residual.unsqueeze(0)
        
    hidden_cell = residual_add_cell(attn_output_proj, residual)
    residual = hidden_cell

    # 6. Post-attention RMSNorm
    normed_cell = rms_norm_cell(hidden_cell, post_attention_layernorm_weight)

    # 7. MLP
    gate_out, up_out = compute_mlp_projections_cell(normed_cell, gate_proj, up_proj)
    gate_activated = silu_cell(gate_out)
    swiglu_out = gate_activated * up_out
    mlp_output = compute_mlp_down_projection_cell(swiglu_out, down_proj)

    # 11. Final Residual Add
    output_cell = final_residual_add_cell(hidden_cell, mlp_output)

    # Ensure precision consistency across step calls
    output_cell = output_cell.to(dtype=hidden_cell.dtype)

    # Return safe 2D tensor for the next layer if batch=1
    if hidden_cell.dim() == 1:
        output_cell = output_cell.squeeze(0)

    return output_cell, k_full, v_full


# =============================================================================
# Utility
# =============================================================================

def print_cell_computation_order():
    """Print cell-level computation stages."""
    stages = [
        "1. RMSNorm(input_cell)",
        "2. QKV Projection (single token)",
        "3. Attention scores vs cached K",
        "4. Softmax + Attention output",
        "5. Residual Add 1",
        "6. Post-attention RMSNorm",
        "7. MLP Projections (gate + up)",
        "8. SiLU activation",
        "9. SwiGLU",
        "10. MLP Down Projection",
        "11. Residual Add 2 (output)"
    ]
    print("\nCell-Level Transformer Computation Order:")
    print("-" * 50)
    for stage in stages:
        print(f"  {stage}")


if __name__ == "__main__":
    print_cell_computation_order()
