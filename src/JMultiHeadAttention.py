import math
import cupy as cp
import torch

class JMultiHeadAttention:
    """
    純 GPU 原生加速版多頭注意力機制 (100% CUDA-Native 閉環)
    """
    def __init__(self, loader, cacher, layer_idx, num_heads=32, head_dim=128, max_seq_len=8192):
        self.loader = loader
        self.cacher = cacher
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.max_seq_len = max_seq_len

    def to_cupy(self, torch_tensor):
        """標準高能橋樑：將 Torch Tensor 0 複製轉換為 CuPy Array"""
        if torch_tensor is None:
            return None
        if isinstance(torch_tensor, cp.ndarray):
            return torch_tensor
        return cp.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor.contiguous()))
    
    def to_torch(self, cupy_array, device="cuda", dtype=torch.float16):
        """標準高能橋樑：將 CuPy Array 0 複製轉換為 Torch Tensor"""
        if cupy_array is None:
            return None
        return torch.utils.dlpack.from_dlpack(cupy_array.toDlpack()).to(device=device, dtype=dtype)
    
    def forward(self, loader, hidden_states, past_kv=None, device="cuda"):
        """
        hidden_states: [batch, seq_len, hidden_size]
        """
        uk_str = f"{self.layer_idx}"
        
        # ====================== 1. 載入權重 (純 GPU 流式滑動視窗) ======================
        q_w_flat = self.cacher.get(f"layers.{self.layer_idx}.self_attn.q_proj.weight", uk_str, target_device=device)
        k_w_flat = self.cacher.get(f"layers.{self.layer_idx}.self_attn.k_proj.weight", uk_str, target_device=device)
        v_w_flat = self.cacher.get(f"layers.{self.layer_idx}.self_attn.v_proj.weight", uk_str, target_device=device)
        o_w_flat = self.cacher.get(f"layers.{self.layer_idx}.self_attn.o_proj.weight", uk_str, target_device=device)

        # 🎯【核心大修正：幾何形狀自適應推導防線】
        # 拋棄死板的硬編碼 (4096, 4096)，改用模型的真實物理體積動態反推形狀
        q_w_cp = q_w_flat.reshape(-1, self.hidden_size)
        o_w_cp = o_w_flat.reshape(self.hidden_size, -1)
        
        # 🚀 這裡就是消滅 ValueError 的黑科技：
        # 不論模型是 MHA 還是 8 頭的 GQA、甚至是 1 頭的 MQA，-1 會命令 CUDA 自動填入正確的縱軸高度！
        # k_w_cp 完美的自適應成 [1024, 4096]，與真實硬體完全相容
        k_w_cp = k_w_flat.reshape(-1, self.hidden_size)
        v_w_cp = v_w_flat.reshape(-1, self.hidden_size)

        # 🎯【核心優化：動態識別並紀錄 K/V 的真實頭數（kv_heads）】
        # 這樣等一下 reshape QKV 格式時，維度才能完美齒輪嚙合
        real_kv_heads = k_w_cp.shape[0] // self.head_dim

        # ====================== 2. Q、K、V 投影 ======================
        hidden_cp = self.to_cupy(hidden_states)

        # 執行標準矩陣相乘，此時 q 的形狀是 [B, S, 4096]，而 k 和 v 會自動且正確地變成 [B, S, 1024]
        q = cp.matmul(hidden_cp, q_w_cp.T)
        k = cp.matmul(hidden_cp, k_w_cp.T)
        v = cp.matmul(hidden_cp, v_w_cp.T)

        del q_w_flat, k_w_flat, v_w_flat, q_w_cp, k_w_cp, v_w_cp

        # ====================== 3. 重塑為多頭格式 ======================
        batch, seq_len, _ = q.shape

        # 🎯【核心修正】：將原本死板的 self.num_heads 改為動態推導出來的 real_kv_heads！
        # 這樣不論是 Q 還是 K/V，在橫跨自迴圈時的多頭幾何維度通通對齊
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, real_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, real_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # ====================== 4. KV Cache (純 GPU 內高能閉環) ======================
        if past_kv is not None:
            past_k_cp, past_v_cp = past_kv
            k = cp.concatenate([past_k_cp, k], axis=2)
            v = cp.concatenate([past_v_cp, v], axis=2)
        
        new_past_kv = (k, v)

        # ====================== 5. 縮放點積注意力 + 智慧因果 Mask ======================
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # 🎯【GQA 核心廣播安全對齊】：
        # 如果是 GQA (Q頭數 32 > KV頭數 8)，我們必須在計算注意力分數前，
        # 讓 K 和 V 順著頭維度（axis=1）進行自適應複製擴展，與 Q 頭完全共振對齊！
        if self.num_heads != real_kv_heads:
            num_queries_per_kv = self.num_heads // real_kv_heads
            # 將 k, v 從 [B, 8, S, D] 瞬間廣播拉伸為 [B, 32, S, D]
            k_scaled = cp.repeat(k, num_queries_per_kv, axis=1)
            v_scaled = cp.repeat(v, num_queries_per_kv, axis=1)
        else:
            k_scaled = k
            v_scaled = v

        # 使用對齊擴展後的 k_scaled 計算 scores [B, 32, seq_len, kv_len]
        scores = cp.matmul(q, k_scaled.transpose(0, 1, 3, 2)) * scale
        kv_len = k_scaled.shape[2]
        
        if seq_len > 1:
            mask = cp.triu(cp.ones((seq_len, kv_len), dtype=cp.bool_), k=1)
            mask = mask[None, None, :, :]
            scores = cp.where(mask, cp.array(float('-inf'), dtype=scores.dtype), scores)

        # Softmax
        scores_max = cp.max(scores, axis=-1, keepdims=True)
        attn_weights = cp.exp(scores.astype(cp.float32) - scores_max.astype(cp.float32))
        attn_weights = attn_weights / cp.sum(attn_weights, axis=-1, keepdims=True)
        attn_weights = attn_weights.astype(scores.dtype)

        # ====================== 6. Attention Output ======================
        # 使用對齊擴展後的 v_scaled 計算輸出
        attn_output = cp.matmul(attn_weights, v_scaled)   # [batch, 32, seq_len, head_dim]

        # 合併多頭幾何
        attn_output = cp.ascontiguousarray(attn_output.transpose(0, 2, 1, 3)).reshape(batch, seq_len, self.hidden_size)

        # ====================== 7. Output Projection ======================
        attn_output = cp.matmul(attn_output, o_w_cp.T)
        del o_w_flat, o_w_cp

        # 0 複製轉回 Torch Tensor
        attn_output_torch = self.to_torch(attn_output, device)

        return attn_output_torch, new_past_kv