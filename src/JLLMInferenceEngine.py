import cupy as cp
import numpy as np
from transformers import AutoTokenizer
import torch
import math
from src.JMultiHeadAttention import *

class JLLMInferenceEngine:
    def __init__(self, loader, cacher, tokenizer_path, run_device="cuda", max_new_tokens=20):
        self.loader = loader
        self.device = run_device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.cacher = cacher
        self.num_layers = self.cacher.num_layers
        self.max_new_tokens = max_new_tokens
        
        # 預建注意力機制算子
        self.attn_layers = [
            JMultiHeadAttention(self.loader, self.cacher, layer_idx, num_heads=32, head_dim=128)
            for layer_idx in range(self.num_layers)
        ]
        self.kv_caches = [None] * self.num_layers
        
        # 🎯【核心大修正】：開機時不再一口氣 record_weight()！保持顯存完美淨空！
        print(f"🎬 JLLM 智能動態推理引擎已啟動！開機顯存佔用：0 MB。")

    def to_cupy(self, torch_tensor):
        """標準高能橋樑：將 Torch Tensor 0 複製轉換為 CuPy Array"""
        if torch_tensor is None:
            return None
        # 使用 dlpack 實現完美的 GPU 記憶體共享，不發生 PCIe 搬運
        return cp.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor.contiguous()))

    def to_torch(self, cupy_array, dtype=torch.float16):
        """標準高能橋樑：將 CuPy Array 0 複製轉換為 Torch Tensor"""
        if cupy_array is None:
            return None
        return torch.utils.dlpack.from_dlpack(cupy_array.toDlpack()).to(device=self.device, dtype=dtype)

    def rms_norm(self, x_cp, weight_cp, eps=1e-5):
        """
        究極穩定版：混合精度 RMSNorm，防範 FP16 平方溢出
        修正：將 PyTorch 專屬的 cp.rsqrt 改為相容的 CuPy 倒數平方根計算
        """
        # 1. 在計算平方和均值時，強行升級到 FP32 確保絕不溢出
        variance = cp.mean(x_cp.astype(cp.float32) ** 2, axis=-1, keepdims=True)
        
        # 2. 🎯【核心修正】：消滅 cp.rsqrt，改用 1 / cp.sqrt() 完美替代
        # 算完開根號倒數後，再降回 float16 與原矩陣相乘
        rsqrt_cp = 1.0 / cp.sqrt(variance + eps)
        x_cp = x_cp * rsqrt_cp.astype(cp.float16)
        
        # 3. 顯式對齊廣播維度，防止底層 CUDA 核心發生 Strides 錯位導致亂碼
        if x_cp.ndim == 3 and weight_cp.ndim == 1:
            return x_cp * weight_cp[None, None, :]
        elif x_cp.ndim == 2 and weight_cp.ndim == 1:
            return x_cp * weight_cp[None, :]
            
        return x_cp * weight_cp

    def generate_stream(self, prompt: str):
        # 1. Tokenize 入口
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_input_ids_cp = cp.asarray(input_ids.cpu().numpy(), dtype=cp.int32)

        # 🎯【智慧型名稱對齊】：自動從 Loader 搜出真實的全名，徹底解決 "model not found"
        all_tensor_names = list(self.loader.header["tensors"].keys())
        
        # 模糊匹配搜尋入口 Embedding 真實名稱
        real_embed_name = next((k for k in all_tensor_names if "embed_tokens" in k), "embed_tokens.weight")
        # 模糊匹配搜尋出口 Final Norm 真實名稱
        real_norm_name = next((k for k in all_tensor_names if "norm.weight" in k or ("model.norm" in k)), "norm.weight")
        # 模糊匹配搜尋出口 LM Head 真實名稱
        real_lm_head_name = next((k for k in all_tensor_names if "lm_head" in k), "lm_head.weight")

        print(f"🔍 智慧型硬體名稱對齊成功：")
        print(f" ┠─ 入口 Embedding: {real_embed_name}")
        print(f" ┠─ 出口 Final Norm: {real_norm_name}")
        print(f" ┠─ 出口 LM Head: {real_lm_head_name}")
        print(f"\n💬 使用者輸入: {prompt}")
        print("🧠 JLLM 智能動態推理引擎流式生成中...\n" + "─" * 80)

        for step in range(self.max_new_tokens):
            
            # ====================== 1. Embedding 動態加載 ======================
            # 🎯【核心修正】：使用模糊搜索出來的 real_embed_name 進行加載
            # ====================== 1. Embedding 動態加載 ======================
            embed_weight = self.loader.get_matrix(real_embed_name, self.device)
            if embed_weight is not None:
                # 智慧型防禦橋樑：自動識別 Embedding 型態，0 複製安全轉換
                if hasattr(embed_weight, "contiguous"):  # PyTorch
                    embed_weight_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(embed_weight.contiguous()))
                elif isinstance(embed_weight, cp.ndarray): # CuPy
                    embed_weight_cp = cp.ascontiguousarray(embed_weight)
                else:  # NumPy
                    embed_weight_cp = cp.ascontiguousarray(cp.asarray(embed_weight))

                # 壓縮雜湊註冊
                weights_bucket, keys = self.cacher._process_and_hash_weight(embed_weight_cp)
                
                # 🎯【核心修正】：成對登記！同時將 1D Keys 和 硬體資料安全桶（Bucket）塞進快取導航庫中
                self.cacher.weight_caches["embed_tokens.weight"] = {"0": keys}
                self.cacher.bucket_caches["embed_tokens.weight"] = {"0": weights_bucket} # 👈 補上這行，徹底摧毀 KeyError！
                
                # 此時 Cacher.get 就能完美從 bucket_caches 撈到反向地圖，發動 0 毫秒物理膨脹
                embed_flat_cp = self.cacher.get("embed_tokens.weight", "0", target_device=self.device)
                
                # 重塑回模型原本的 Embedding 矩陣形狀
                vocab_size = embed_weight.shape[0] if hasattr(embed_weight, "shape") else 128256
                embed_matrix_cp = embed_flat_cp.reshape(vocab_size, -1)
                
                # 全 GPU 內部高效查表
                hidden_cp = embed_matrix_cp[current_input_ids_cp]
                
                # 計算完畢立刻就地清空 Embedding 顯存，保持 0 MB 閒置
                self.cacher.clear_single_layer("0")
                del embed_weight, embed_weight_cp, embed_flat_cp, embed_matrix_cp
            else:
                raise RuntimeError(f"❌ 致命錯誤：在權重庫中完全找不到任何包含 'embed_tokens' 的矩陣！")

            # ====================== 2. Transformer Layers ======================
            for layer_idx in range(self.num_layers):
                residual = hidden_cp
                layer_key = f"{layer_idx}"

                # 【流式滑動視窗預載】：此時顯存中只有當前這 1 層的快取權重！
                self.cacher.cache_single_layer(layer_idx)

                # ---------- Input Layernorm ----------
                ln_w = self.cacher.get(f"layers.{layer_idx}.input_layernorm.weight", layer_key, target_device=self.device)
                hidden_cp = self.rms_norm(hidden_cp, ln_w)

                # ---------- Attention (極速複用版) ----------
                attn = self.attn_layers[layer_idx]
                hidden_torch = self.to_torch(hidden_cp)
                attn_output_torch, new_kv = attn.forward(
                    self.loader, hidden_torch, 
                    past_kv=self.kv_caches[layer_idx], 
                    device=self.device
                )
                self.kv_caches[layer_idx] = new_kv
                hidden_cp = residual + self.to_cupy(attn_output_torch)

                # ---------- Post Attention Norm + MLP ----------
                residual = hidden_cp
                post_ln_w = self.cacher.get(f"layers.{layer_idx}.post_attention_layernorm.weight", layer_key, target_device=self.device)
                normed_cp = self.rms_norm(hidden_cp, post_ln_w)

                # === SwiGLU MLP (gate + up + down) ===
                gate_flat = self.cacher.get(f"layers.{layer_idx}.mlp.gate_proj.weight", layer_key, target_device=self.device)
                up_flat   = self.cacher.get(f"layers.{layer_idx}.mlp.up_proj.weight", layer_key, target_device=self.device)
                down_flat = self.cacher.get(f"layers.{layer_idx}.mlp.down_proj.weight", layer_key, target_device=self.device)

                # MLP 二維幾何自適應重塑
                gate_cp = gate_flat.reshape(-1, normed_cp.shape[-1])
                up_cp   = up_flat.reshape(-1, normed_cp.shape[-1])
                down_cp = down_flat.reshape(-1, gate_cp.shape[0]) 

                # 🚀 執行 GPU Tensor Cores 高速二維矩陣相乘
                gate_linear = cp.matmul(normed_cp, gate_cp.T)
                up_out      = cp.matmul(normed_cp, up_cp.T)
                
                # 🎯【終極修正】：拋棄複雜的 scipy 子模組依賴，直接用純原生 cp 算子手動熔煉數值穩定的 Sigmoid！
                # 透過 cp.clip 防止 fp16 發生指數極端溢出，保證數值鋼鐵般的穩定度
                # 數學本質完美的 SwiGLU 門控融合：SiLU(x) = x * Sigmoid(x)
                sigmoid_gate = 1.0 / (1.0 + cp.exp(-cp.clip(gate_linear, -15.0, 15.0)))
                gate_out = gate_linear * sigmoid_gate
                
                # 門控信號與升維信號在 GPU 內萬箭齊發、點對點交織 [Batch, SeqLen, 14336]
                mlp_inter = gate_out * up_out
                
                # 單路降維壓回大腦初始維度 [Batch, SeqLen, 4096]
                mlp_out = cp.matmul(mlp_inter, down_cp.T)
                hidden_cp = residual + mlp_out

                # 徹底蒸發當前層的臨時大顯存，保持滑動視窗絕對淨空
                self.cacher.clear_single_layer(layer_idx)
                del gate_flat, up_flat, down_flat, gate_cp, up_cp, down_cp
                del gate_linear, up_out, sigmoid_gate, gate_out, mlp_inter, mlp_out

            # ====================== 3. Final Norm 動態加載 ======================
            # 🎯【核心修正】：使用模糊搜索出來的 real_norm_name 進行加載
            final_w = self.loader.get_matrix(real_norm_name, self.device)
            if final_w is not None:
                if hasattr(final_w, "contiguous"):
                    final_w_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(final_w.contiguous()))
                elif isinstance(final_w, cp.ndarray):
                    final_w_cp = cp.ascontiguousarray(final_w)
                else:
                    final_w_cp = cp.ascontiguousarray(cp.asarray(final_w))
                
                hidden_cp = self.rms_norm(hidden_cp, final_w_cp)
                del final_w, final_w_cp

            # ====================== 4. LM Head 動態加載 ======================
            # 🎯【核心修正】：使用模糊搜索出來的 real_lm_head_name 進行加載
            lm_head = self.loader.get_matrix(real_lm_head_name, self.device)
            if lm_head is not None:
                if hasattr(lm_head, "contiguous"):
                    lm_head_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(lm_head.contiguous()))
                elif isinstance(lm_head, cp.ndarray):
                    lm_head_cp = cp.ascontiguousarray(lm_head)
                else:
                    lm_head_cp = cp.ascontiguousarray(cp.asarray(lm_head))
                
                # 只對最後一個 token 計算預測概率
                logits_cp = cp.matmul(hidden_cp[:, -1:, :], lm_head_cp.T)
                next_token_id_cp = cp.argmax(logits_cp, axis=-1)
                
                # 輸出解碼文字
                print(self.tokenizer.decode([int(next_token_id_cp[0, 0])]), end="", flush=True)

                # 更新下一輪的輸入 ID
                current_input_ids_cp = next_token_id_cp
                del lm_head, lm_head_cp, logits_cp

        print("\n" + "─" * 80)
        print("✅ JLLM 智能動態推理引擎解碼結束！")