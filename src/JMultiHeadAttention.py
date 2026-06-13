import math
import torch
import torch.nn.functional as F

def apply_causal_mask(scores, seq_len, total_kv_len):
    """Apply causal mask accounting for historical tokens in the KV cache."""
    if seq_len <= 1:
        return scores

    mask = torch.full((seq_len, total_kv_len), float('-inf'), device=scores.device)
    mask = torch.triu(mask, diagonal=total_kv_len - seq_len + 1)
    return scores + mask.unsqueeze(0).unsqueeze(0)

def swiglu_mlp(normed_t, loader, layer_idx, device):
    """🌟 Fused SwiGLU math stream"""
    gate_flat = loader.get_matrix(f"layers.{layer_idx}.mlp.gate_proj.weight", target_device=device)
    up_flat = loader.get_matrix(f"layers.{layer_idx}.mlp.up_proj.weight", target_device=device)
    down_flat = loader.get_matrix(f"layers.{layer_idx}.mlp.down_proj.weight", target_device=device)

    hidden_dim = normed_t.shape[-1]
    gate_t = gate_flat.reshape(-1, hidden_dim)
    up_t = up_flat.reshape(-1, hidden_dim)
    down_t = down_flat.reshape(hidden_dim, -1)

    gate_linear = torch.matmul(normed_t, gate_t.T)
    up_out = torch.matmul(normed_t, up_t.T)
    
    # Use native optimized SiLU
    mlp_out = torch.matmul(F.silu(gate_linear) * up_out, down_t.T)
    
    del gate_flat, up_flat, down_flat
    return mlp_out

class JMultiHeadAttention:
    def __init__(self, loader, cacher):
        self.loader = loader
        self.cacher = cacher
        self.hidden_size = 4096
        self.num_heads = 32
        self.head_dim = 128
        self.num_kv_heads = 8

    def forward(self, layer_idx, x, raw_flat_kv=None, extractor=None, device="cuda", current_seq_len=1):
        batch_size, seq_len, _ = x.shape

        q_proj = self.loader.get_matrix(f"layers.{layer_idx}.self_attn.q_proj.weight", target_device=device)
        k_proj = self.loader.get_matrix(f"layers.{layer_idx}.self_attn.k_proj.weight", target_device=device)
        v_proj = self.loader.get_matrix(f"layers.{layer_idx}.self_attn.v_proj.weight", target_device=device)

        q_current = torch.matmul(x, q_proj.T).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k_current = torch.matmul(x, k_proj.T).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v_current = torch.matmul(x, v_proj.T).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Cache V immediately (no rotation)
        v_out_torch = v_current.clone()
        del q_proj, k_proj, v_proj

        # --- RoPE ---
        seq_start_pos = current_seq_len - seq_len if current_seq_len > seq_len else 0
        positions = torch.arange(seq_start_pos, seq_start_pos + seq_len, dtype=torch.float32, device=device)
        # RoPE with correct theta for Qwen2.5-7B (theta=1000000.0)
        rope_theta = 1000000.0
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))
        
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        cos = angles.cos()[:, None, :]
        sin = angles.sin()[:, None, :]
        
        cos_full = torch.cat((cos, cos), dim=-1)
        sin_full = torch.cat((sin, sin), dim=-1)
        
        def rotate_half(tensor):
            t1, t2 = tensor[..., :self.head_dim // 2], tensor[..., self.head_dim // 2:]
            return torch.cat((-t2, t1), dim=-1)

        q_current = (q_current * cos_full) + (rotate_half(q_current) * sin_full)
        k_current = (k_current * cos_full) + (rotate_half(k_current) * sin_full)

        # Cache K AFTER rotation
        k_out_torch = k_current.clone()

        # --- KV Management ---
        if raw_flat_kv and raw_flat_kv[0] is not None:
            history_k, history_v = raw_flat_kv
            k_full = torch.cat((history_k, k_current.to(torch.float16)), dim=1)
            v_full = torch.cat((history_v, v_current.to(torch.float16)), dim=1)
        else:
            k_full, v_full = k_current.to(torch.float16), v_current.to(torch.float16)

        # --- GQA Alignment ---
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        k_scaled = k_full.repeat_interleave(num_queries_per_kv, dim=2)
        v_scaled = v_full.repeat_interleave(num_queries_per_kv, dim=2)

        # --- Attention ---
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q_current.to(torch.float16).transpose(1, 2), k_scaled.transpose(1, 2).transpose(2, 3)) * scale

        if seq_len > 1:
            total_kv_len = k_full.shape[1]
            scores = apply_causal_mask(scores, seq_len, total_kv_len)

        scores_max = scores.amax(dim=-1, keepdim=True)
        attn_weights = (scores - scores_max).exp()
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        context = torch.matmul(attn_weights, v_scaled.transpose(1, 2)).transpose(1, 2)
        context_flat = context.reshape(batch_size * seq_len, self.num_heads * self.head_dim)

        # --- Output Projection ---
        o_proj = self.loader.get_matrix(f"layers.{layer_idx}.self_attn.o_proj.weight", target_device=device)
        output_states = torch.matmul(context_flat, o_proj.T).reshape(batch_size, seq_len, self.hidden_size)
        
        del o_proj
        return output_states, (k_out_torch, v_out_torch)