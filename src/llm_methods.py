"""
LLM Methods - Sequence Level
==========================
Transformer layer computation functions for sequence processing.
"""

import math
import torch
import torch.nn.functional as F


# =============================================================================
# RMSNorm
# =============================================================================

def rms_norm(x_t, weight_t, eps=1e-5):
    """
    RMSNorm for sequence input.

    Args:
        x_t: Input tensor [batch, seq, hidden]
        weight_t: Layer norm weight [hidden]
        eps: Numerical stability

    Returns:
        Normalized tensor [batch, seq, hidden]
    """
    if weight_t is None:
        return x_t

    x_f32 = x_t.to(torch.float32)
    weight_f32 = weight_t.to(torch.float32)

    variance = x_f32.pow(2).mean(-1, keepdim=True)
    normalized = x_f32 * torch.rsqrt(variance + eps)

    return normalized * weight_f32


# =============================================================================
# QKV Projection
# =============================================================================

def compute_qkv_projection(x, q_proj, k_proj, v_proj):
    """
    QKV projection for sequence.

    Args:
        x: Input tensor [batch, seq, hidden]
        q_proj, k_proj, v_proj: Projection matrices

    Returns:
        Tuple of (q, k, v) tensors
    """
    q = torch.matmul(x, q_proj.T) if q_proj is not None else x
    k = torch.matmul(x, k_proj.T) if k_proj is not None else x
    v = torch.matmul(x, v_proj.T) if v_proj is not None else x

    return q, k, v


# =============================================================================
# Attention
# =============================================================================

def compute_attention_scores(q, k, scale=None):
    """
    Compute attention scores: Q @ K^T / sqrt(d)

    Args:
        q: Query tensor [batch, seq, heads, head_dim]
        k: Key tensor [batch, seq, heads, head_dim]
        scale: Optional scale factor

    Returns:
        Attention scores [batch, heads, seq, seq]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return scores


def apply_attention_softmax(scores):
    """
    Apply softmax to attention scores.

    Args:
        scores: Attention scores [batch, heads, seq, seq]

    Returns:
        Attention weights (normalized)
    """
    scores_max = scores.amax(dim=-1, keepdim=True)
    attn_weights = (scores - scores_max).exp()
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
    return attn_weights


def compute_attention_output(attn_weights, v):
    """
    Compute attention output: attn_weights @ V

    Args:
        attn_weights: Attention weights [batch, heads, seq, seq]
        v: Value tensor [batch, seq, heads, head_dim]

    Returns:
        Context tensor [batch, seq, heads, head_dim]
    """
    return torch.matmul(attn_weights, v)


# =============================================================================
# Residual and MLP
# =============================================================================

def residual_add(x, residual):
    """Add residual connection."""
    return x + residual


def compute_mlp_projections(x, gate_proj, up_proj):
    """
    Compute MLP gate and up projections.

    Args:
        x: Input tensor [batch, seq, hidden]
        gate_proj, up_proj: Projection matrices

    Returns:
        Tuple of (gate_out, up_out)
    """
    gate_out = torch.matmul(x, gate_proj.T) if gate_proj is not None else x
    up_out = torch.matmul(x, up_proj.T) if up_proj is not None else x
    return gate_out, up_out


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def compute_mlp_down_projection(swiglu_out, down_proj):
    """
    MLP down projection.

    Args:
        swiglu_out: SwiGLU output [batch, seq, intermediate]
        down_proj: Down projection matrix

    Returns:
        Output tensor [batch, seq, hidden]
    """
    return torch.matmul(swiglu_out, down_proj.T) if down_proj is not None else swiglu_out


def final_residual_add(x, mlp_out):
    """Final residual add."""
    return x + mlp_out


# =============================================================================
# Complete Layer Forward
# =============================================================================

def transformer_layer_forward(
    hidden_states,
    residual,
    q_proj, k_proj, v_proj, o_proj,
    gate_proj, up_proj, down_proj,
    input_layernorm_weight, post_attention_layernorm_weight,
    num_heads, num_kv_heads, head_dim
):
    """
    Complete transformer layer forward pass.

    Args:
        hidden_states: Input [batch, seq, hidden]
        residual: Skip connection input
        q_proj, k_proj, v_proj, o_proj: Attention projections
        gate_proj, up_proj, down_proj: MLP projections
        input_layernorm_weight, post_attention_layernorm_weight: Layer norms
        num_heads: Number of Q heads
        num_kv_heads: Number of K,V heads
        head_dim: Dimension per head

    Returns:
        Tuple of (output, (k, v) for cache)
    """
    # 1. Input RMSNorm
    normed = rms_norm(hidden_states, input_layernorm_weight)

    # 2. QKV Projection
    q, k, v = compute_qkv_projection(normed, q_proj, k_proj, v_proj)

    # Reshape for multi-head
    batch_size, seq_len, _ = hidden_states.shape
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim)

    # 3. Attention scores
    scale = 1.0 / math.sqrt(head_dim)
    scores = compute_attention_scores(q, k, scale)

    # 4. Softmax
    attn_weights = apply_attention_softmax(scores)

    # 5. Attention output
    attn_output = compute_attention_output(attn_weights, v)

    # Reshape and project
    attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)
    attn_output = torch.matmul(attn_output, o_proj.T) if o_proj is not None else attn_output

    # 6. Residual Add 1
    hidden_states = residual_add(attn_output, residual)
    residual = hidden_states

    # 7. Post-attention RMSNorm
    normed = rms_norm(hidden_states, post_attention_layernorm_weight)

    # 8. MLP Projections
    gate_out, up_out = compute_mlp_projections(normed, gate_proj, up_proj)

    # 9. SiLU activation
    gate_activated = silu(gate_out)

    # 10. SwiGLU
    swiglu_out = gate_activated * up_out

    # 11. MLP Down
    mlp_output = compute_mlp_down_projection(swiglu_out, down_proj)

    # 12. Final Residual
    output = final_residual_add(residual, mlp_output)

    return output, (k, v)


def print_computation_order():
    """Print the computation stages."""
    stages = [
        "1. RMSNorm(input)",
        "2. QKV Projection",
        "3. Attention scores",
        "4. Softmax",
        "5. Attention output",
        "6. Residual Add 1",
        "7. Post-attention RMSNorm",
        "8. MLP Projections (gate + up)",
        "9. SiLU activation",
        "10. SwiGLU",
        "11. MLP Down Projection",
        "12. Residual Add 2 (output)"
    ]
    print("\nTransformer Layer Computation Order:")
    print("-" * 50)
    for stage in stages:
        print(f"  {stage}")


if __name__ == "__main__":
    print_computation_order()
