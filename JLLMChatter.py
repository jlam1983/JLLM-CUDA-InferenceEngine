"""
JLLM Chatter — layer-by-layer LLM inference using .jllm weights.
JLLM Chatter — 使用 .jllm 權重的逐層 LLM 推論引擎。

Memory management (Windows-safe):
記憶體管理（Windows 相容）：
  • uint8 raw data: load from mmap → CPU RAM numpy arrays (~0.5GB total)
  • uint8 原始資料：從記憶體對應載入至 CPU RAM numpy 陣列（約 0.5GB 總計）
  • float16 GPU reconstruction: one layer at a time
  • float16 GPU 重建：每次處理一個層
  • LRU cache: last 2 layers kept as float16 on GPU (avoid re-reconstruction)
  • LRU 快取：最後 2 層以 float16 保留在 GPU 上（避免重建）
  • Every del followed by gc.collect() + cp.cuda.Stream.null.synchronize()
  • 每個 del 後接 gc.collect() + cp.cuda.Stream.null.synchronize()
  • References set to None explicitly before GC
  • 明確將參考設為 None 後再進行 GC
  • Prefill: rebuild cache each step (layers reused across prefill)
  • Prefill：每步重建快取（層在 Prefill 期間跨步驟重用）
  • Decode: keep 2 cached layers, rebuild rest per step
  • Decode：保留 2 個快取層，每步重建其餘層
"""

# 理論說明：JLLM 是一種量化格式，將模型權重以 uint8 原始資料儲存，配合 outlier 值重建為 float16。
# 步驟說明：從 mmap 載入 uint8 資料至 CPU RAM，後續在 GPU 上即時重建為 float16，以節省記憶體。

from __future__ import annotations

import gc
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Generator

import numpy as np

try:
    import cupy as cp
    _GPU_AVAILABLE = cp.is_available()
except ImportError:
    _GPU_AVAILABLE = False
    cp = None

from JLLMReader import JLLMReader

_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_JLLM = _SCRIPT_DIR / "jqwen.jllm"
_DEFAULT_MODEL_DIR = _SCRIPT_DIR / "Models" / "qwen"

_GEN = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "max_new_tokens": 512,
    "eos_token_ids": [151645, 151643],
}


# ══════════════════════════════════════════════════════════════════════════════
#  JLLMChatter — Pipeline Inference Engine
# ══════════════════════════════════════════════════════════════════════════════

class JLLMChatter:
    # ─────────────────────────────────────────────────────────────────────────
    #  __init__ — initialise reader, tokenizer, memory buffers
    # __init__ — 初始化讀取器、分詞器、記憶體緩衝區
    # ─────────────────────────────────────────────────────────────────────────
    # 理論說明：__init__ 為管線引擎的建構函式，負責初始化 JLLM 讀取器、分詞器、記憶體緩衝區及快取結構。
    # 步驟說明：依序執行讀取器初始化、架構參數提取、分詞器載入、LRU 快取配置、KV 快取初始化。

    def __init__(
        self,
        jllm_path: str | Path | None = None,
        *,
        use_cuda: bool = True,
        tokenizer_dir: str | Path | None = None,
        max_seq_len: int = 8192,
    ):
        self.jllm_path = Path(jllm_path or _DEFAULT_JLLM)
        self.tokenizer_dir = Path(tokenizer_dir or _DEFAULT_MODEL_DIR)
        self.use_cuda = use_cuda and _GPU_AVAILABLE
        self.max_seq_len = max_seq_len
        self._xp = cp if self.use_cuda else np

        # ── JLLM reader ───────────────────────────────────────────────────
        # JLLM 讀取器
        # 理論說明：JLLMReader 負責讀取 .jllm 檔案格式，解析架構資訊與張量資料。
        # 步驟說明：開啟 JLLM 檔案，提取模型類型、hidden_size、num_layers 等架構參數。
        self._reader = JLLMReader(self.jllm_path)
        self.model_type: str = self._reader.model_type()
        arch = self._reader.architecture()

        self.hidden_size = arch["hidden_size"]
        self.num_layers = arch["num_hidden_layers"]
        self.num_heads = arch["num_attention_heads"]
        self.num_kv_heads = arch["num_key_value_heads"]
        self.vocab_size = arch["vocab_size"]
        self.rope_theta = arch["rope_theta"]
        self.intermediate_size = arch.get("intermediate_size", self.hidden_size * 4)
        self.head_dim = arch.get("head_dim", self.hidden_size // self.num_heads)

        # ── Tokenizer ─────────────────────────────────────────────────────
        # 分詞器
        # 理論說明：分詞器將文字輸入轉換為 token ID 序列，並將輸出 token 序列轉換回文字。
        # 步驟說明：使用 Hugging Face AutoTokenizer 從指定目錄載入分詞器模型。
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir), trust_remote_code=True
        )
        self.eos_token_ids: list[int] = _GEN["eos_token_ids"]
        self._rope_inv_freq = self._compute_rope_inv_freq()

        # ── uint8 raw weights (CPU RAM) ───────────────────────────────────
        # uint8 原始權重（CPU RAM）
        # 理論說明：uint8 原始權重包含 mapping、normal_indices、outliers、outlier_mask，用於重建 float16 張量。
        # 步驟說明：建立 layer_raw、embed_raw、lm_head_raw、final_norm_raw 字典，儲存各層權重的 uint8 原始資料。
        self._layer_raw: dict[int, dict] = {}
        self._embed_raw: dict | None = None
        self._lm_head_raw: dict | None = None
        self._final_norm_raw: dict | None = None

        # ── float16 GPU cache (LRU, max 2 layers) ────────────────────────
        # float16 GPU 快取（LRU，最多 2 層）
        # 理論說明：LRU（Least Recently Used）快取保留最近使用的 2 層 float16 權重，避免重複重建。
        # 步驟說明：使用 OrderedDict 實作 LRU，decode 階段每步驟驅逐最舊層並載入新層。
        self._layer_f16: OrderedDict = OrderedDict()
        self._layer_f16_max = 2

        # ── float16 embed / lm_head / norm GPU cache ─────────────────────
        # float16 embed / lm_head / norm GPU 快取
        # 理論說明：embed、lm_head、final_norm 為共享權重，在 prefill 階段重建後快取供後續步驟重複使用。
        # 步驟說明：各權重首次使用時觸發重建，之後直接取用快取，不重複重建。
        self._embed_f16_cache = None
        self._lm_head_f16_cache = None
        self._final_norm_f16_cache = None

        # ── KV cache ──────────────────────────────────────────────────────
        # KV 快取
        # 理論說明：KV 快取儲存已計算的 Key 和 Value 向量，避免自迴歸解碼時重複計算歷史 token 的注意力。
        # 步驟說明：每層維護獨立的 k/v 張量，新 token 產生時附加至現有 KV 序列後方。
        self._kv_cache: dict = {}
        self._kv_len = 0

        self._gen = dict(_GEN)

        # Load uint8 raw data to CPU RAM (one-time, ~0.5GB)
        # 將 uint8 原始資料載入 CPU RAM（一次性，約 0.5GB）
        # 理論說明：JLLM 格式的 uint8 資料從磁碟 mmap 載入至 CPU RAM，GPU 重建時再上傳，以減少 GPU 記憶體佔用。
        # 步驟說明：依序呼叫 _step_00_load_layer_raw 載入各層權重及 embed/lm_head/norm 權重至 CPU RAM。
        self._step_00_load_raw_to_ram()

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 00 — load raw uint8 weights from mmap into CPU RAM
    # ══════════════════════════════════════════════════════════════════════

    def _step_00_load_raw_to_ram(self):
        """Load all tensor raw data (uint8) from mmap into CPU numpy arrays."""
        print("  Loading uint8 raw to CPU RAM...", end="", flush=True)
        for li in range(self.num_layers):
            self._layer_raw[li] = self._step_00_load_layer_raw(li)
            print(f" L{li+1}", end="", flush=True)
        self._embed_raw = self._step_00_load_raw("model.embed_tokens.weight")
        self._lm_head_raw = self._step_00_load_raw("lm_head.weight")
        self._final_norm_raw = self._step_00_load_raw("model.norm.weight")
        print(" done")
        gc.collect()

    def _step_00_load_layer_raw(self, layer_idx: int) -> dict:
        """Load all weight tensors for one transformer layer."""
        # 載入單一 transformer 層的所有權重張量
        # 理論說明：每層包含 q_proj、k_proj、v_proj、o_proj（注意力）、gate_proj、up_proj、down_proj（MLP）及兩個 layernorm。
        # 步驟說明：針對指定 layer_idx，依序查詢並載入該層的 9 個權重張量原始資料。
        prefix = f"model.layers.{layer_idx}."
        names = [
            ("q_proj", "self_attn.q_proj.weight"),
            ("k_proj", "self_attn.k_proj.weight"),
            ("v_proj", "self_attn.v_proj.weight"),
            ("o_proj", "self_attn.o_proj.weight"),
            ("gate_proj", "mlp.gate_proj.weight"),
            ("up_proj", "mlp.up_proj.weight"),
            ("down_proj", "mlp.down_proj.weight"),
            ("input_layernorm", "input_layernorm.weight"),
            ("post_attention_layernorm", "post_attention_layernorm.weight"),
        ]
        return {key: self._step_00_load_raw(prefix + full_name) for key, full_name in names}

    def _step_00_load_raw(self, name: str) -> dict:
        """Fetch raw uint8 components for one tensor from the mmap reader."""
        # 從 mmap 讀取器取得單一張量的 uint8 原始元件
        # 理論說明：JLLM 格式將權重分為 normal 值（uint8 mapping + indices）和 outlier 值，支援高效壓縮。
        # 步驟說明：呼叫 reader.read_tensor_raw 取得 outliers、mapping、normal_indices、outlier_mask 等元件。
        try:
            raw = self._reader.read_tensor_raw(name)
        except KeyError:
            available = self._reader.tensor_names()
            raise KeyError(
                f"Tensor '{name}' not found in JLLM. Available: {available[:10]}"
            ) from None
        return {
            "outliers":       np.asarray(raw["outliers"]),
            "mapping":        np.asarray(raw["mapping"]),
            "normal_indices": np.asarray(raw["normal_indices"]),
            "outlier_mask":   np.asarray(raw["outlier_mask"]),
            "shape":          raw["shape"],
            "numel":          int(np.prod(raw["shape"])),
            "outlier_count":  len(raw["outliers"]),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 01 — embed token IDs to hidden states
    # ══════════════════════════════════════════════════════════════════════

    def _step_01_embed(self, input_ids):
        """Look up embedding vectors for token IDs and return hidden states."""
        # 查詢 token ID 的嵌入向量並回傳隱藏狀態
        # 理論說明：嵌入層將 token ID 查詢為對應的 hidden_dim 維度向量，是 Transformer 的第一層輸入。
        # 步驟說明：若快取未建立則先重建 embed 權重，再以 input_ids 查詢嵌入表，回傳隱藏狀態矩陣。
        if self._embed_f16_cache is None:
            self._embed_f16_cache = self._step_02_reconstruct(self._embed_raw)
        return self._xp.asarray(self._embed_f16_cache[input_ids])

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 02 — reconstruct float16 from uint8 raw (CPU→GPU)
    # ══════════════════════════════════════════════════════════════════════

    def _step_02_reconstruct(self, raw: dict):
        """Reconstruct a float16 tensor on GPU/CPU from its uint8 components."""
        # 從 uint8 元件重建 float16 張量（GPU 或 CPU）
        # 理論說明：利用 outlier_mask 排序區分 normal/outlier 位置，依據 mapping 和 indices 重建完整 float16 向量。
        # 步驟說明：排序 outlier_mask 分離位置，將 mapping[normal_indices] 填入 normal_pos，outliers 填入 outlier_pos。
        numel = raw["numel"]
        outlier_count = raw["outlier_count"]

        sort_idx = self._xp.asarray(raw["outlier_mask"]).argsort(kind="stable")
        normal_pos  = sort_idx[:numel - outlier_count]
        outlier_pos = sort_idx[numel - outlier_count:]

        result = self._xp.empty(numel, dtype=np.float16)
        result[normal_pos]  = self._xp.asarray(raw["mapping"])[self._xp.asarray(raw["normal_indices"])]
        result[outlier_pos] = self._xp.asarray(raw["outliers"])
        return result.reshape(raw["shape"])


    def _step_02_expand_f16(self, raw: dict) :
        """Expand a uint8 raw dict to float16 on GPU/CPU (cached, idempotent)."""
        # 將 uint8 原始字典擴展為 GPU/CPU 上的 float16（已快取，冪等）
        # 理論說明：expand_f16 與 reconstruct 原理相同，但支援就地快取（raw["_f16"]），避免重複重建相同權重。
        # 步驟說明：檢查快取是否存在，若存在直接回傳；否則重建後存入 raw["_f16"] 再回傳。
        f16 = raw.get("_f16")
        if f16 is not None:
            return f16

        numel = raw["numel"]
        outlier_count = raw["outlier_count"]

        outlier_mask = self._xp.asarray(raw["outlier_mask"])
        sort_idx = outlier_mask.argsort(kind="stable")
        normal_pos  = sort_idx[:numel - outlier_count]
        outlier_pos = sort_idx[numel - outlier_count:]

        mapping       = self._xp.asarray(raw["mapping"])
        normal_indices = self._xp.asarray(raw["normal_indices"])

        result = self._xp.empty(numel, dtype=np.float16)
        result[normal_pos]  = mapping[normal_indices]
        result[outlier_pos] = self._xp.asarray(raw["outliers"])
        f16_result = result.reshape(raw["shape"])

        raw["_f16"] = f16_result
        return f16_result

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 03 — forward a single transformer layer
    # ══════════════════════════════════════════════════════════════════════

    def _step_03_forward_layer(self, layer_idx: int, hidden_states, w: dict) :
        """One transformer layer: RMSNorm → Attention (Q/K/V + RoPE + GQA + mask) → SwiGLU MLP."""
        # 單一 transformer 層：RMSNorm → Attention (Q/K/V + RoPE + GQA + mask) → SwiGLU MLP
        # 理論說明：每層執行：輸入正規化 → QKV 投影 + RoPE 旋轉位置編碼 → 分組查詢注意力 → 殘差連接 → SwiGLU MLP → 輸出。
        # 步驟說明：依序執行 RMSNorm、矩陣投影、RoPE、KV 快取查詢、GQA、注意力分數計算、Softmax、MLP，最後回傳隱藏狀態。
        xp = self._xp

        # Expand uint8 raw weights to float16 on-the-fly (cached after first call)
        # 即時將 uint8 原始權重擴展為 float16（首次呼叫後快取）
        # 理論說明：首次使用權重時自動觸發 expand_f16 重建，並快取於 raw["_f16"]，後續呼叫直接取用。
        # 步驟說明：對 w 字典中每個原始權重呼叫 expand_f16，確保所有權重已轉換為 float16 格式。
        w = {k: self._step_02_expand_f16(v) for k, v in w.items()}

        # Pre-norm
        # 預備正規化
        # 理論說明：Transformer 使用 Pre-Norm 結構，每層輸入前先經 RMSNorm 正規化，穩定梯度流動。
        # 步驟說明：呼叫 rms_norm 對 hidden_states 以 input_layernorm 權重進行正規化。
        h = self._step_05_rms_norm(hidden_states, w["input_layernorm"])

        # QKV projections
        # QKV 投影
        # 理論說明：QKV 投影將隱藏狀態透過三個獨立權重矩陣轉換為 Query、Key、Value 向量，維度為 [seq, hidden_dim]。
        # 步驟說明：分別執行 h @ q_proj.T、h @ k_proj.T、h @ v_proj.T 矩陣乘法產生 Q、K、V。
        q = xp.asarray(h @ w["q_proj"].T)
        k = xp.asarray(h @ w["k_proj"].T)
        v = xp.asarray(h @ w["v_proj"].T)

        seq_l = q.shape[0]
        q = xp.transpose(xp.reshape(q, (seq_l, self.num_heads, self.head_dim)),    (1, 0, 2))
        k = xp.transpose(xp.reshape(k, (seq_l, self.num_kv_heads, self.head_dim)), (1, 0, 2))
        v = xp.transpose(xp.reshape(v, (seq_l, self.num_kv_heads, self.head_dim)), (1, 0, 2))

        q = self._step_06_apply_rope(q)
        k = self._step_06_apply_rope(k)

        # KV cache
        # KV 快取
        # 理論說明：KV 快取避免自迴歸生成時重複計算歷史 token 的注意力，每層維護獨立的 K/V 序列。
        # 步驟說明：將新產生的 K/V 向量附加至該層的 KV 快取，並回傳完整序列供後續注意力計算使用。
        k_full, v_full = self._step_07_cache_kv(layer_idx, k, v)

        # Grouped-query attention: repeat KV heads to match Q heads
        # 分組查詢注意力：重複 KV heads 以匹配 Q heads
        # 理論說明：GQA 允許 KV head 數量少於 Q head 數量，透過重複 KV 向量匹配 Q 維度以節省計算量。
        # 步驟說明：若 num_kv_heads < num_heads，則沿 axis=0 重複 K/V 向量 repeat = num_heads // num_kv_heads 次。
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_full = xp.repeat(k_full, repeat, axis=0)
            v_full = xp.repeat(v_full, repeat, axis=0)

        # Attention scores + causal mask
        # 注意力分數 + 因果遮罩
        # 理論說明：因果遮罩確保每個位置只能注意到自身及其之前的 token，防止未來資訊洩漏。
        # 步驟說明：計算 Q @ K^T 加上縮放因子，若序列長度 > 1則套用下三角因果遮罩，將未來位置設為 -1e4。
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = xp.matmul(q * scale, xp.transpose(k_full, (0, 2, 1)))
        if seq_l > 1:
            causal = xp.tril(xp.ones((seq_l, k_full.shape[1]), dtype=xp.bool_))
            scores = xp.where(causal, scores, xp.asarray(-1e4, dtype=scores.dtype))

        attn = self._step_08_softmax(scores, axis=-1)
        attn_out = xp.matmul(attn, v_full)
        attn_out = xp.transpose(attn_out, (1, 0, 2)).reshape((seq_l, self.hidden_size))
        h = xp.asarray(h + (attn_out @ w["o_proj"].T))

        # MLP (SwiGLU) — float32 for stability
        # MLP (SwiGLU) — 使用 float32 以確保穩定性
        # 理論說明：SwiGLU 是一種門控線性單元，結合 SiLU 啟動函式與門控機制，提升模型表達能力；使用 float32 確保數值穩定。
        # 步驟說明：計算 gate = h_norm @ gate_proj.T、up = h_norm @ up_proj.T，將 gate 通過 SiLU 後與 up 相乘，再通過 down_proj。
        h_norm = self._step_05_rms_norm(h, w["post_attention_layernorm"])
        gate = h_norm @ w["gate_proj"].T
        up   = h_norm @ w["up_proj"].T
        gate_f = gate.astype(xp.float32)
        up_f   = up.astype(xp.float32)
        silu   = up_f / (1.0 + xp.exp(-up_f))
        h_mlp  = xp.asarray(gate_f * silu) @ w["down_proj"].T

        return hidden_states + h_mlp

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 04 — final RMS norm after all layers
    # ══════════════════════════════════════════════════════════════════════

    def _step_04_final_norm(self, hidden_states) :
        """Apply final RMS normalization before the LM head."""
        # 在 LM head 之前套用最終 RMS 正規化
        # 理論說明：最終正規化確保所有層輸出在進入 LM head 前已標準化，穩定輸出分佈。
        # 步驟說明：若快取未建立則先重建 final_norm 權重，再以 rms_norm 對 hidden_states 進行正規化。
        if self._final_norm_f16_cache is None:
            self._final_norm_f16_cache = self._step_02_reconstruct(self._final_norm_raw)
        return self._step_05_rms_norm(hidden_states, self._final_norm_f16_cache)

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 05 — RMS norm (shared across prefill and decode)
    # ══════════════════════════════════════════════════════════════════════

    def _step_05_rms_norm(self, x, weight, eps=1e-6) :
        xp = self._xp
        x_f = x.astype(xp.float32)
        variance = xp.mean(x_f ** 2, axis=-1, keepdims=True)
        norm = x_f * (variance + eps) ** (-0.5)
        return (norm * weight).astype(x.dtype)
        # 理論說明：RMSNorm 僅計算均方根而非均值，移除均值中央化以簡化計算並保持效能。
        # 步驟說明：計算 x_f^2 的均值，取倒數開方後乘以 x_f，再乘以 weight 權重，回傳與輸入同 dtype 的正規化結果。

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 06 — rotary positional embeddings (RoPE)
    # ══════════════════════════════════════════════════════════════════════

    def _step_06_apply_rope(self, x) :
        xp = self._xp
        seq = x.shape[1]
        n_pairs = self.head_dim // 2
        positions = xp.arange(seq, dtype=xp.float32)
        inv_freq = xp.asarray(self._rope_inv_freq, dtype=xp.float32)
        angles = xp.outer(positions, inv_freq)
        cos = xp.cos(angles).astype(x.dtype)
        sin = xp.sin(angles).astype(x.dtype)
        cos = cos[None, :, :]
        sin = sin[None, :, :]
        x1, x2 = x[:, :, :n_pairs], x[:, :, n_pairs:]
        return xp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).astype(x.dtype)
        # 理論說明：RoPE 旋轉位置編碼透過對 query/key 向量施加旋轉矩陣，將位置資訊編碼至向量維度中。
        # 步驟說明：計算位置角度，產生 cos/sin 矩陣，將向量分為前半/後半維度，執行旋轉融合後拼接回原有維度。

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 07 — KV cache management
    # ══════════════════════════════════════════════════════════════════════

    def _step_07_cache_kv(self, layer_idx: int, k, v):
        xp = self._xp
        if f"k{layer_idx}" not in self._kv_cache:
            self._kv_cache[f"k{layer_idx}"] = xp.zeros(
                (self.num_kv_heads, self.max_seq_len, self.head_dim), dtype=xp.float16)
            self._kv_cache[f"v{layer_idx}"] = xp.zeros(
                (self.num_kv_heads, self.max_seq_len, self.head_dim), dtype=xp.float16)
            self._kv_len = 0

        pos = self._kv_len
        seq = k.shape[1]
        self._kv_cache[f"k{layer_idx}"][:, pos:pos+seq, :] = k
        self._kv_cache[f"v{layer_idx}"][:, pos:pos+seq, :] = v
        self._kv_len += seq
        return (
            self._kv_cache[f"k{layer_idx}"][:, :pos+seq, :],
            self._kv_cache[f"v{layer_idx}"][:, :pos+seq, :],
        )
        # 理論說明：KV 快取維護已計算的 Key/Value 向量，使自迴歸解碼時無需重新計算歷史 token 的注意力。
        # 步驟說明：首次呼叫時初始化該層的 K/V 緩衝區，后續呼叫將新 K/V 向量附加至現有序列後方並回傳完整序列。

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 08 — softmax (shared)
    # ══════════════════════════════════════════════════════════════════════

    def _step_08_softmax(self, x, axis=-1) :
        x_max = self._xp.max(x, axis=axis, keepdims=True)
        exp_x = self._xp.exp(x - x_max)
        return exp_x / self._xp.sum(exp_x, axis=axis, keepdims=True)
        # 理論說明：Softmax 將注意力分數轉換為機率分佈，指數運算配合最大值減法確保數值穩定。
        # 步驟說明：計算最大值進行數值穩定化，執行指數運算後除以總和得到機率分佈。

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 09 — LM head: project hidden → logits
    # ══════════════════════════════════════════════════════════════════════

    def _step_09_lm_head(self, hidden_states) :
        """Project hidden states to vocabulary-sized logits via the LM head."""
        # 透過 LM head 將隱藏狀態投影至詞彙大小的 logits
        # 理論說明：LM head 將正規化後的隱藏狀態投影至詞彙維度，產生每個 token 的未標準化分數（logits）。
        # 步驟說明：若快取未建立則先重建 lm_head 權重，執行 hidden_states @ lm_head.T 矩陣乘法產生 logits。
        if self._lm_head_f16_cache is None:
            self._lm_head_f16_cache = self._step_02_reconstruct(self._lm_head_raw)
        return self._xp.asarray(hidden_states @ self._lm_head_f16_cache.T)

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 10 — sample next token from logits
    # ══════════════════════════════════════════════════════════════════════

    def _step_10_sample_token(
        self, logits, temperature=0.7, top_p=0.8, top_k=20,
        repetition_penalty=1.05, prev_tokens=None,
    ) -> int:
        xp = self._xp
        if logits.ndim == 2:
            logits = logits[0]
        logits = xp.asarray(logits, dtype=xp.float32)

        # Repetition penalty
        # 理論說明：重複懲罰降低已生成 token 的分數，鼓勵模型產生更多樣化的回應。
        # 步驟說明：對 prev_tokens 中每個唯一 token，若其 logits > 0 則除以 penalty，否則乘以 penalty。
        if repetition_penalty != 1.0 and prev_tokens:
            for tok in set(prev_tokens):
                if 0 <= tok < len(logits):
                    logits[tok] /= repetition_penalty if logits[tok] > 0 else repetition_penalty

        # Temperature scaling
        # 理論說明：Temperature 控制機率分佈的平滑度，高溫增加隨機性，低溫使分佈更尖峰。
        # 步驟說明：若 temperature 設值且不等於 1.0，則將 logits 除以 temperature 以調整分佈。
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        # 理論說明：Top-k 過濾只保留分數最高的 k 個 token，其餘設為負無窮，避免低分 token 被選中。
        # 步驟說明：找出第 k 高的分數閾值，將低於閾值的 logits 設為 -inf。
        if top_k > 0:
            top_k = min(top_k, len(logits))
            thr = xp.sort(logits)[-top_k]
            logits = xp.where(logits < thr, -xp.inf, logits)

        # Nucleus (top-p) filtering
        # 理論說明：Nucleus (top-p) 過濾從最高分開始累加機率，保留總和達到 top_p 的 token，動態調整候選集合大小。
        # 步驟說明：排序 logits，計算 softmax 後的累加和，找到達到 top_p 的切點，將其餘位置設為 -inf。
        if top_p < 1.0:
            idx = xp.argsort(logits)[::-1]
            p = self._step_08_softmax(logits[idx])
            cs = xp.cumsum(p)
            cut = int(xp.searchsorted(cs, top_p).item())
            mask = xp.ones_like(logits, dtype=xp.bool_)
            mask[idx[:cut + 1]] = False
            logits = xp.where(mask, -xp.inf, logits)

        probs = self._step_08_softmax(logits)
        probs_cpu = cp.asnumpy(probs) if self.use_cuda else np.asarray(probs)
        probs_cpu /= probs_cpu.sum()
        return int(np.random.choice(len(probs_cpu), p=probs_cpu))

    # ══════════════════════════════════════════════════════════════════════
    #  PIPELINE STEP 11 — LRU layer cache management
    # ══════════════════════════════════════════════════════════════════════

    def _step_11_get_layer_f16(self, layer_idx: int) -> dict:
        """Return layer weights (from cache or reconstruct from CPU uint8)."""
        # 回傳層權重（從快取或從 CPU uint8 重建）
        # 理論說明：LRU 快取策略確保常用層保留在記憶體中，減少重建頻率以提升效能。
        # 步驟說明：檢查快取是否已有所需層，若有則移至末尾表示最近使用；否則在快取已滿時驅逐最舊層後新增。
        if layer_idx in self._layer_f16:
            self._layer_f16.move_to_end(layer_idx)
            return self._layer_f16[layer_idx]

        # Evict oldest entry if cache is full
        # 若快取已滿，驅逐最舊的項目
        # 理論說明：當快取達到容量上限時，LRU 策略驅逐最久未使用的項目以容納新項目。
        # 步驟說明：呼叫 popitem(last=False) 移除最舊項目，釋放其關聯的 GPU 陣列記憶體。
        if len(self._layer_f16) >= self._layer_f16_max:
            _, evict_weights = self._layer_f16.popitem(last=False)
            for arr in evict_weights.values():
                del arr
            if "_f16" in evict_weights:
                del evict_weights["_f16"]
            del evict_weights
            gc.collect()
            if self.use_cuda:
                cp.cuda.Stream.null.synchronize()

        raw = self._layer_raw[layer_idx]
        self._layer_f16[layer_idx] = raw
        return raw

    def _step_11_evict_layer(self, layer_idx: int):
        """Evict a layer from the GPU cache and free its GPU memory."""
        # 從 GPU 快取驅逐層並釋放其 GPU 記憶體
        # 理論說明：驅逐操作釋放特定層的 GPU 記憶體，避免記憶體不足，並確保資源正確釋放。
        # 步驟說明：從快取移除該層，逐一刪除其權重陣列，呼叫 gc.collect() 並同步 CUDA 串流。
        if layer_idx not in self._layer_f16:
            return
        arr = self._layer_f16.pop(layer_idx)
        for v in arr.values():
            del v
        if "_f16" in arr:
            del arr["_f16"]
        del arr
        gc.collect()
        if self.use_cuda:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()

    def _step_11_dispose_all_f16(self):
        """Free all float16 GPU caches."""
        # 釋放所有 float16 GPU 快取
        # 理論說明：結束生成任務時徹底釋放所有快取記憶體，避免記憶體洩漏。
        # 步驟說明：清空所有層快取及 embed/lm_head/final_norm 快取，呼叫 gc.collect() 並釋放 GPU 記憶體池。
        for w in self._layer_f16.values():
            for v in w.values():
                del v
            w.clear()
        self._layer_f16.clear()
        for ref in (self._embed_f16_cache, self._lm_head_f16_cache, self._final_norm_f16_cache):
            if ref is not None:
                del ref
        self._embed_f16_cache = None
        self._lm_head_f16_cache = None
        self._final_norm_f16_cache = None
        gc.collect()
        if self.use_cuda:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()

    # ══════════════════════════════════════════════════════════════════════
    #  PUBLIC API — chat()
    # ══════════════════════════════════════════════════════════════════════

    def chat(
        self,
        prompt: str | list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        **_: object,
    ) -> Generator[str, None, str]:
        T   = temperature        if temperature        is not None else self._gen["temperature"]
        tp  = top_p              if top_p              is not None else self._gen["top_p"]
        tk  = top_k              if top_k              is not None else self._gen["top_k"]
        rp  = repetition_penalty if repetition_penalty is not None else self._gen["repetition_penalty"]
        max_new = max_new_tokens if max_new_tokens    is not None else self._gen["max_new_tokens"]

        # ── Parse prompt ─────────────────────────────────────────────────
        # 解析 prompt
        # 理論說明：支援字串或聊天格式列表輸入，分詞器將其轉換為 token ID 序列作為模型輸入。
        # 步驟說明：若為列表則使用 apply_chat_template 格式化，否則直接使用輸入字串，最後轉換為 input_ids。
        if isinstance(prompt, list):
            prompt_str = self._tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = prompt
        input_ids = self._tokenizer(prompt_str, return_tensors="np").input_ids[0]

        # ══════════════════════════════════════════════════════════════
        #  PIPELINE — PREFILL PHASE
        #  (all layers cached; no eviction during prefill)
        # ══════════════════════════════════════════════════════════════
        # 理論說明：Prefill 階段一次性處理完整輸入序列，建立 KV 快取，之後解碼階段可快速取用。
        # 步驟說明：預熱所有層快取、嵌入輸入、前向所有層、執行最終正規化。

        # STEP 00: warm up layer cache (no eviction while all layers are loaded)
        # STEP 00：預熱層快取（所有層載入時不驅逐）
        # 理論說明：Prefill 期間將所有層載入快取，避免中途驅逐導致重複重建，確保推論效能。
        # 步驟說明：暫存 LRU 上限後設為總層數，依序呼叫 get_layer_f16 載入所有層。
        saved_max = self._layer_f16_max
        self._layer_f16_max = self.num_layers

        for li in range(self.num_layers):
            self._step_11_get_layer_f16(li)

        # STEP 01: embed input tokens
        # STEP 01：嵌入輸入 tokens
        # 理論說明：將輸入 token ID 序列查詢為嵌入向量，產生初始隱藏狀態。
        # 步驟說明：呼叫 embed 函式查詢嵌入表，產生 [seq_len, hidden_dim] 維度的隱藏狀態矩陣。
        hidden = self._step_01_embed(input_ids)

        # Cache final norm before clearing CPU RAM
        # 在清除 CPU RAM 前快取最終正規化權重
        # 理論說明：final_norm 為共享權重，在 prefill 階段使用後解碼階段仍需使用，須先行快取。
        # 步驟說明：若快取未建立則重建 final_norm 權重至 GPU，確保解碼階段無需 CPU RAM 仍可使用。
        if self._final_norm_f16_cache is None:
            self._final_norm_f16_cache = self._step_02_reconstruct(self._final_norm_raw)

        # Cache lm_head before clearing CPU RAM
        # 在清除 CPU RAM 前快取 LM head 權重
        # 理論說明：lm_head 為共享權重，解碼階段每步都需使用，須先行快取避免之後從 None 重建。
        # 步驟說明：若快取未建立則重建 lm_head 權重至 GPU，確保解碼階段無需 CPU RAM 仍可使用。
        if self._lm_head_f16_cache is None:
            self._lm_head_f16_cache = self._step_02_reconstruct(self._lm_head_raw)

        # Free CPU RAM — all weights are now on GPU
        # 釋放 CPU RAM — 所有權重現在都在 GPU 上
        # 理論說明：Prefill 完成後 CPU RAM 中的原始權重已不再需要，可釋放以節省記憶體。
        # 步驟說明：清除 layer_raw、embed_raw、lm_head_raw、final_norm_raw 並執行垃圾回收。
        self._layer_raw.clear()
        self._embed_raw = None
        self._lm_head_raw = None
        self._final_norm_raw = None
        gc.collect()

        # STEP 02–03: forward all transformer layers
        # STEP 02–03：前向所有 transformer 層
        # 理論說明：依序通過所有 transformer 層，每層執行注意力與 MLP 運算，逐步提取特徵。
        # 步驟說明：對每層取得 float16 權重，呼叫 forward_layer 執行該層的前向計算。
        for li in range(self.num_layers):
            w = self._step_11_get_layer_f16(li)
            hidden = self._step_03_forward_layer(li, hidden, w)

        # STEP 04: final RMS norm
        # STEP 04：最終 RMS 正規化
        # 理論說明：在進入 LM head 前對隱藏狀態進行最終正規化，確保輸出分佈穩定。
        # 步驟說明：呼叫 final_norm 函式對 hidden_states 進行正規化。
        hidden = self._step_04_final_norm(hidden)

        # Restore LRU cap for decode phase
        # 恢復 LRU 上限以進入解碼階段
        # 理論說明：Prefill 完成後恢復 LRU 上限至 2，進入解碼階段的層級快取管理模式。
        # 步驟說明：將 _layer_f16_max 恢復為 saved_max 值。
        self._layer_f16_max = saved_max

        # ══════════════════════════════════════════════════════════════
        #  PIPELINE — DECODE PHASE  (autoregressive, one token at a time)
        # ══════════════════════════════════════════════════════════════
        # 理論說明：解碼階段為自迴歸生成，每次產生一個新 token，透過 KV 快取加速生成。
        # 步驟說明：迴圈執行取樣、嵌入、前向、驅逐，直至產生 EOS token 或達到最大長度。

        generated = []
        prev_ids  = []

        for _ in range(max_new):
            # STEP 09: project to logits
            # STEP 09：投影至 logits
            # 理論說明：將隱藏狀態透過 LM head 投影為詞彙維度的 logits，作為取樣的依據。
            # 步驟說明：呼叫 lm_head 函式執行矩陣乘法，產生每個 token 的未標準化分數。
            logits = self._step_09_lm_head(hidden)

            # STEP 10: sample next token
            # STEP 10：取樣下一個 token
            # 理論說明：根據 logits 進行重複懲罰、溫度縮放、Top-k/p 過濾後取樣，產生下一個 token。
            # 步驟說明：依序套用各過濾機制，最後以加權隨機取樣選擇 token。
            token = self._step_10_sample_token(
                logits, temperature=T, top_p=tp, top_k=tk,
                repetition_penalty=rp, prev_tokens=prev_ids,
            )

            if token in self.eos_token_ids:
                break

            token_str = self._tokenizer.decode(token, skip_special_tokens=False)
            generated.append(token_str)
            prev_ids.append(token)
            yield token_str

            # STEP 01: embed new token
            # STEP 01：嵌入新 token
            # 理論說明：將新產生的 token 查詢其嵌入向量，作為下一輪前向計算的輸入。
            # 步驟說明：將 token 轉為陣列後呼叫 embed 函式，取得該 token 的隱藏狀態。
            hidden = self._step_01_embed(self._xp.array([token]))

            # STEP 02–03: forward all layers (evict after each)
            # STEP 02–03：前向所有層（每層後驅逐）
            # 理論說明：解碼階段採用滑動視窗策略，每步驟只保留 2 層快取以節省記憶體。
            # 步驟說明：對每層執行前向計算後立即驅逐，釋放 GPU 記憶體供下一步使用。
            for li in range(self.num_layers):
                w = self._step_11_get_layer_f16(li)
                hidden = self._step_03_forward_layer(li, hidden, w)
                self._step_11_evict_layer(li)

            # STEP 04: final norm
            # STEP 04：最終正規化
            # 理論說明：每步解碼後執行最終正規化，確保進入 LM head 的隱藏狀態已標準化。
            # 步驟說明：呼叫 final_norm 函式對 hidden_states 進行正規化。
            hidden = self._step_04_final_norm(hidden)

            del logits
            if self.use_cuda:
                cp.cuda.Stream.null.synchronize()

        self._step_11_dispose_all_f16()
        gc.collect()
        return "".join(generated)

    # ─────────────────────────────────────────────────────────────────────────
    #  Config / lifecycle
    # 設定 / 生命週期
    # ─────────────────────────────────────────────────────────────────────────
    # 理論說明：生命週期管理涵蓋生成參數配置、資源釋放及上下文管理器實作，確保資源正確初始化與釋放。
    # 步驟說明：提供 generation_config、configure_generation、close 方法及 __enter__/__exit__ 上下文管理。

    @property
    def generation_config(self) -> dict:
        return dict(self._gen)

    def configure_generation(self, **kwargs):
        self._gen.update(kwargs)

    def close(self):
        self._layer_raw.clear()
        self._embed_raw = None
        self._lm_head_raw = None
        self._final_norm_raw = None
        self._step_11_dispose_all_f16()
        self._kv_cache.clear()
        self._reader.close()
        gc.collect()
        if self.use_cuda:
            cp.cuda.Stream.null.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, *_: object):
        self.close()

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal helpers (not part of pipeline)
    # 內部輔助函式（非管線的一部分）
    # ─────────────────────────────────────────────────────────────────────────
    # 理論說明：內部輔助函式支援管線運算，提供 RoPE 頻率預計算等基礎功能，不屬於主推論管線。
    # 步驟說明：_compute_rope_inv_freq 預計算 RoPE 的逆頻率陣列，供後續 apply_rope 使用。

    def _compute_rope_inv_freq(self) -> np.ndarray:
        inv = np.zeros(self.head_dim // 2, dtype=np.float32)
        for i in range(self.head_dim // 2):
            inv[i] = 1.0 / (self.rope_theta ** (2 * i / self.head_dim))
        return inv


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="JLLM Chatter")
    p.add_argument("--jllm", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    jllm_path = args.jllm or _DEFAULT_JLLM
    if not Path(jllm_path).exists():
        print(f"[Error] Not found: {jllm_path}")
        sys.exit(1)

    use_cuda = not args.cpu and _GPU_AVAILABLE
    print(f"[JLLM Chatter] Loading {jllm_path} on {'cuda' if use_cuda else 'cpu'} ...")

    with JLLMChatter(jllm_path, use_cuda=use_cuda) as c:
        print(f"[Model]  {c.model_type}")
        print(f"[Layers] {c.num_layers}  hidden={c.hidden_size}  "
              f"heads={c.num_heads}  kv_heads={c.num_kv_heads}")
        print(f"[Vocab]  {c.vocab_size}")

        gen = c.chat(
            args.prompt or "Explain the key differences between Python and JavaScript.",
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n[Output] ", end="", flush=True)
        for tok in gen:
            print(tok, end="", flush=True)
        print()


if __name__ == "__main__":
    main()