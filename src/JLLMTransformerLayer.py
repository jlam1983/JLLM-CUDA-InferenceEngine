"""
JLLM Transformer Layer - Complete Implementation
=================================================
All-in-one transformer layer with:
- RMSNorm
- QKV Projection with biases
- RoPE (Rotary Position Embedding)
- Multi-Head Attention with GQA
- SwiGLU MLP
- Sequence-level and Cell-level forward passes
"""

import math
import torch
import torch.nn.functional as F


# =============================================================================
# RMSNorm
# =============================================================================

def rms_norm(x_t, weight_t, eps=1e-5):
    """RMSNorm for sequence input [batch, seq, hidden]."""
    if weight_t is None:
        return x_t
    x_f32 = x_t.to(torch.float32)
    weight_f32 = weight_t.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normalized = x_f32 * torch.rsqrt(variance + eps)
    return normalized * weight_f32


def rms_norm_cell(x_cell, weight_t, eps=1e-5):
    """RMSNorm for single token [hidden_dim]."""
    if weight_t is None:
        return x_cell
    input_dtype = x_cell.dtype
    x_f32 = x_cell.to(torch.float32)
    weight_f32 = weight_t.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normalized = x_f32 * torch.rsqrt(variance + eps)
    return (normalized * weight_f32).to(input_dtype)


# =============================================================================
# QKV Projection
# =============================================================================

def compute_qkv_projection(x, q_proj, k_proj, v_proj, q_bias=None, k_bias=None, v_bias=None):
    """QKV projection with optional biases (for Qwen models)."""
    q = torch.matmul(x, q_proj.T) if q_proj is not None else x
    if q_bias is not None: q = q + q_bias
    k = torch.matmul(x, k_proj.T) if k_proj is not None else x
    if k_bias is not None: k = k + k_bias
    v = torch.matmul(x, v_proj.T) if v_proj is not None else x
    if v_bias is not None: v = v + v_bias
    return q, k, v


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================

def compute_rope_freqs(head_dim, rope_theta, device):
    """Compute RoPE frequency tensor [head_dim // 2]."""
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    return freqs


def apply_rope_to_position(x, positions, freqs):
    """Apply RoPE to tensor [batch, seq, heads, head_dim]."""
    head_dim = x.shape[-1]
    half_dim = head_dim // 2
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x1_out = x1 * cos + (-x2) * sin
    x2_out = x2 * cos + x1 * sin
    return torch.cat([x1_out, x2_out], dim=-1)


def apply_rope_to_position_cell(q_cell, k_cell, position, freqs):
    """Apply RoPE to single token Q and K [batch, 1, heads, head_dim]."""
    head_dim = q_cell.shape[-1]
    half_dim = head_dim // 2
    pos_t = torch.tensor([position], dtype=torch.float32, device=q_cell.device)
    angles = pos_t * freqs
    cos = angles.cos().view(1, 1, 1, half_dim)
    sin = angles.sin().view(1, 1, 1, half_dim)
    q1, q2 = q_cell[..., :half_dim], q_cell[..., half_dim:]
    q1_out = q1 * cos - q2 * sin
    q2_out = q2 * cos + q1 * sin
    q_rotated = torch.cat([q1_out, q2_out], dim=-1)
    k1, k2 = k_cell[..., :half_dim], k_cell[..., half_dim:]
    k1_out = k1 * cos - k2 * sin
    k2_out = k2 * cos + k1 * sin
    k_rotated = torch.cat([k1_out, k2_out], dim=-1)
    return q_rotated, k_rotated


# =============================================================================
# Attention
# =============================================================================

def apply_causal_mask(scores, seq_len, total_kv_len):
    """Apply causal mask for autoregressive models."""
    if seq_len <= 1:
        return scores
    mask = torch.full((seq_len, total_kv_len), float('-inf'), device=scores.device)
    mask = torch.triu(mask, diagonal=total_kv_len - seq_len + 1)
    return scores + mask.unsqueeze(0).unsqueeze(0)


def compute_attention(q, k, v, num_heads, num_kv_heads, head_dim, scale=None):
    """Multi-head attention with GQA support. Q,K,V: [batch, seq, heads, head_dim]."""
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    num_queries_per_kv = num_heads // num_kv_heads
    k_expanded = k.repeat_interleave(num_queries_per_kv, dim=2)
    v_expanded = v.repeat_interleave(num_queries_per_kv, dim=2)
    q_attn = q.transpose(1, 2).to(torch.float16)
    k_attn = k_expanded.transpose(1, 2).to(torch.float16)
    v_attn = v_expanded.transpose(1, 2).to(torch.float16)
    context = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
    return context.transpose(1, 2).contiguous().reshape(q.size(0), q.size(1), num_heads * head_dim)


def compute_attention_with_cache(q_cell, k_cache, v_cache, num_heads, num_kv_heads, head_dim, scale=None):
    """Attention for single token using cached K/V. Q: [num_heads, head_dim], K,V: [num_kv_heads, seq, head_dim]."""
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    num_queries_per_kv = num_heads // num_kv_heads
    k_expanded = k_cache.repeat_interleave(num_queries_per_kv, dim=0)
    q = q_cell.unsqueeze(1)
    scores = torch.matmul(q, k_expanded.transpose(-2, -1)).squeeze(1) * scale
    scores_max = scores.amax(dim=-1, keepdim=True)
    attn_weights = (scores - scores_max).exp()
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
    v_expanded = v_cache.repeat_interleave(num_queries_per_kv, dim=0)
    attn = attn_weights.unsqueeze(1)
    output = torch.matmul(attn, v_expanded)
    return output.squeeze(1)


# =============================================================================
# SwiGLU MLP
# =============================================================================

def silu(x):
    """SiLU activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def compute_mlp(normed_t, loader, layer_idx, device):
    """SwiGLU MLP: SiLU(gate) * up -> down. Load from layer cache file."""
    # Preload layer into dictTensor then get expanded GPU weights
    #loader.tensorManager.preload_layer(layer_idx, loader)
    weights = loader.tensorManager.get_layer_weights(layer_idx, device)

    gate_proj = weights.get(f"layers.{layer_idx}.mlp.gate_proj.weight")
    up_proj = weights.get(f"layers.{layer_idx}.mlp.up_proj.weight")
    down_proj = weights.get(f"layers.{layer_idx}.mlp.down_proj.weight")

    hidden_dim = normed_t.shape[-1]
    inter_size = gate_proj.shape[0] if gate_proj is not None else 0
    gate_t = gate_proj.view(inter_size, hidden_dim).to(device) if gate_proj is not None else None
    up_t = up_proj.view(inter_size, hidden_dim).to(device) if up_proj is not None else None
    down_t = down_proj.view(hidden_dim, inter_size).to(device) if down_proj is not None else None
    del gate_proj, up_proj, down_proj, weights
    # Cast input to match weight dtype
    target_dtype = gate_t.dtype if gate_t is not None else normed_t.dtype
    normed_cast = normed_t.to(target_dtype)
    gate_out = torch.matmul(normed_cast, gate_t.T) if gate_t is not None else None
    up_out = torch.matmul(normed_cast, up_t.T) if up_t is not None else None
    activated = silu(gate_out) if gate_out is not None else None
    swiglu_out = activated * up_out if (activated is not None and up_out is not None) else None
    mlp_out = torch.matmul(swiglu_out, down_t.T) if down_t is not None else None
    return mlp_out



# =============================================================================
# Transformer Layer Forward - Sequence Level
# =============================================================================

def apply_transformer_layer(
    layer_idx,
    hidden_t,
    loader,
    attention_layer,
    device,
    current_seq_len,
    raw_flat_kv=None
):
    """Complete transformer layer forward (sequence-level)."""
    residual = hidden_t

    # Preload layer into dictTensor then get expanded GPU weights
    weights = loader.tensorManager.get_layer_weights(layer_idx, device)

    input_ln_w = weights.get(f"layers.{layer_idx}.input_layernorm.weight")
    post_ln_w = weights.get(f"layers.{layer_idx}.post_attention_layernorm.weight")

    if input_ln_w is not None:
        norm_w = input_ln_w.to(device) if input_ln_w.device != device else input_ln_w
        attn_input = rms_norm(hidden_t, norm_w)
    else:
        attn_input = hidden_t

    attn_output_t, new_kv = attention_layer.forward(
        layer_idx=layer_idx,
        x=attn_input,
        weights=weights,
        raw_flat_kv=raw_flat_kv,
        device=device,
        current_seq_len=current_seq_len
    )

    hidden_t = residual + attn_output_t

    # MLP block
    residual = hidden_t

    if post_ln_w is not None:
        norm_w = post_ln_w.to(device) if post_ln_w.device != device else post_ln_w
        mlp_input = rms_norm(hidden_t, norm_w)
    else:
        mlp_input = hidden_t

    mlp_out = compute_mlp(mlp_input, loader, layer_idx, device)

    if mlp_out is not None:
        hidden_t = residual + mlp_out
    else:
        hidden_t = residual

    return hidden_t, new_kv


# =============================================================================
# Transformer Layer Forward - Cell Level (single token)
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
    """Complete transformer layer forward for single token (cell-level)."""
    # 1. Input RMSNorm
    normed_cell = rms_norm_cell(hidden_cell, input_layernorm_weight)

    # 2. QKV Projection
    q_cell, k_cell, v_cell = compute_qkv_projection(
        normed_cell, q_proj, k_proj, v_proj, q_bias, k_bias, v_bias
    )

    batch_size = hidden_cell.shape[0] if hidden_cell.dim() == 2 else 1

    q_cell = q_cell.view(batch_size, 1, num_heads, head_dim)
    k_cell = k_cell.view(batch_size, 1, num_kv_heads, head_dim)
    v_cell = v_cell.view(batch_size, 1, num_kv_heads, head_dim)

    # 2b. Apply RoPE
    if rope_freqs is not None and current_position is not None:
        q_cell, k_cell = apply_rope_to_position_cell(q_cell, k_cell, current_position, rope_freqs)

    # 3. Cache Append
    if k_cache_cell is not None and v_cache_cell is not None:
        k_full = torch.cat([k_cache_cell, k_cell], dim=1)
        v_full = torch.cat([v_cache_cell, v_cell], dim=1)
    else:
        k_full = k_cell
        v_full = v_cell

    # 4. Attention Using PyTorch SDPA
    q_attn = q_cell.transpose(1, 2)
    k_attn = k_full.transpose(1, 2)
    v_attn = v_full.transpose(1, 2)

    q_attn = q_attn.to(dtype=v_attn.dtype)
    k_attn = k_attn.to(dtype=v_attn.dtype)

    num_queries_per_kv = num_heads // num_kv_heads
    if num_queries_per_kv > 1:
        k_attn = k_attn.repeat_interleave(num_queries_per_kv, dim=1)
        v_attn = v_attn.repeat_interleave(num_queries_per_kv, dim=1)

    context = torch.nn.functional.scaled_dot_product_attention(
        q_attn, k_attn, v_attn, is_causal=False
    )

    context = context.transpose(1, 2).contiguous().reshape(batch_size, num_heads * head_dim)

    # 5. Output projection + Residual Add
    attn_output_proj = torch.matmul(context, o_proj.T) if o_proj is not None else context

    if residual.dim() == 1 and attn_output_proj.dim() == 2:
        residual = residual.unsqueeze(0)

    hidden_cell = attn_output_proj + residual
    residual = hidden_cell

    # 6. Post-attention RMSNorm
    normed_cell = rms_norm_cell(hidden_cell, post_attention_layernorm_weight)

    # 7. MLP
    if gate_proj is not None and up_proj is not None and down_proj is not None:
        gate_out = torch.matmul(normed_cell, gate_proj.T)
        up_out = torch.matmul(normed_cell, up_proj.T)
        gate_activated = silu(gate_out)
        swiglu_out = gate_activated * up_out
        mlp_output = torch.matmul(swiglu_out, down_proj.T)
    else:
        mlp_output = None

    # 8. Final Residual Add
    output_cell = hidden_cell + mlp_output if mlp_output is not None else hidden_cell
    output_cell = output_cell.to(dtype=hidden_cell.dtype)

    if hidden_cell.dim() == 1:
        output_cell = output_cell.squeeze(0)

    return output_cell, k_full, v_full


# =============================================================================
# Utility
# =============================================================================

def print_computation_order():
    """Print sequence-level computation stages."""
    stages = [
        "1. RMSNorm(input)",
        "2. QKV Projection",
        "3. RoPE",
        "4. Attention + GQA",
        "5. Output Projection",
        "6. Residual Add 1",
        "7. Post-attention RMSNorm",
        "8. SwiGLU (gate * up)",
        "9. MLP Down Projection",
        "10. Residual Add 2 (output)"
    ]
    print("\nTransformer Layer Computation Order (Sequence-Level):")
    print("-" * 50)
    for stage in stages:
        print(f"  {stage}")


def print_cell_computation_order():
    """Print cell-level computation stages."""
    stages = [
        "1. RMSNorm(input_cell)",
        "2. QKV Projection (single token)",
        "3. RoPE (single position)",
        "4. Cache Append",
        "5. Attention vs cached K/V",
        "6. Output Projection + Residual Add",
        "7. Post-attention RMSNorm",
        "8. SwiGLU",
        "9. Residual Add 2 (output)"
    ]
    print("\nTransformer Layer Computation Order (Cell-Level):")
    print("-" * 50)
    for stage in stages:
        print(f"  {stage}")


if __name__ == "__main__":
    print_computation_order()
    print()
    print_cell_computation_order()