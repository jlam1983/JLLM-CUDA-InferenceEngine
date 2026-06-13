"""
JLLM Inference Engine - Core Logic
===================================
Main inference engine shared by GPU and CPU modes.
"""

import gc
import torch
import os
import shutil

from JLLMLoader import JLLMLoader
from JLLMExtractor import JLLMExtractor
from JLLMFileCacheConverter import JLLMFileCacheConverter
from JLLMTransformerLayer import (    
    apply_transformer_layer,
    transformer_layer_forward_cell,
    rms_norm_cell,
    compute_rope_freqs as compute_rope_freqs_cell,
)
from JLLMMultiHeadAttention import *
from JLLMStreamPrefetcher import *

# Alias for backward compatibility
rms_norm = rms_norm_cell



def mask_logits(logits_t, vocab_size, special_token_ids):
    """Mask OOV tokens and special tokens in logits before sampling.

    Args:
        logits_t: Logits tensor [..., vocab]
        vocab_size: Tokenizer vocabulary size (tokens >= this are OOV)
        special_token_ids: List of special token IDs to mask
    """
    mask = torch.zeros_like(logits_t)
    # Mask OOV tokens (beyond tokenizer vocab)
    mask[..., vocab_size:] = float('-inf')
    # Mask special tokens
    for sid in special_token_ids:
        if sid < logits_t.shape[-1]:
            mask[..., sid] = float('-inf')
    return logits_t + mask


def sample_token(logits_t, vocab_size, special_token_ids):
    """Sample next token after masking OOV and special tokens."""
    masked_logits = mask_logits(logits_t, vocab_size, special_token_ids)
    return masked_logits.argmax(dim=-1)

# =============================================================================
# Transformer Layer Application
# =============================================================================
# Tensor Alignment Utilities
# =============================================================================

def find_tensor_name(tensor_names, pattern):
    return next((k for k in tensor_names if pattern in k), None)

def align_tensor_names(all_tensor_names):
    return {
        "embed": find_tensor_name(all_tensor_names, "embed_tokens") or "embed_tokens.weight",
        "norm": find_tensor_name(all_tensor_names, "model.norm") or find_tensor_name(all_tensor_names, "ln_f") or "norm.weight",
        "lm_head": find_tensor_name(all_tensor_names, "lm_head") or "lm_head.weight"
    }

def clear_storage_dir(storage_dir):
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
        print(f"[Cache] Cleared {storage_dir}.")
    os.makedirs(storage_dir, exist_ok=True)

# =============================================================================
# Main Inference Engine Class
# =============================================================================

class JLLMInferenceEngine:
    def __init__(self, model_path, tokenizer, storage_dir="./cache_store", device="cuda", cache_mode="full_gpu", determinism_mode="fast"):
        self.device = device
        self.model_path = model_path
        self.storage_dir = storage_dir
        self.cache_mode = cache_mode
        self.determinism_mode = determinism_mode

        # Apply determinism settings before any computation
        self._apply_determinism(determinism_mode)

        self.loader = JLLMLoader(model_path, tokenizer=tokenizer)
        for layer_idx in range(28):
            self.loader.tensorManager.preload_layer(layer_idx, self.loader)
        self.tokenizer = self.loader.tokenizer

        self.extractor = JLLMExtractor()
        self.cacher = JLLMFileCacheConverter(storage_dir=storage_dir, run_device=self.device)
        self.prefetcher = JLLMStreamPrefetcher(run_device=self.device)
        self.attention_layer = JLLMMultiHeadAttention(self.loader, self.cacher)

        self._auto_detect_architecture()

        self.window_size = 256
        self.max_new_tokens = 100

        # High-Speed Weight Caching
        self.weight_cache = {}
        self.gpu_kv_cache = {}
        self.kv_cache = {}

        print(f"[JLLM] Engine initialized. Mode: {cache_mode}, Determinism: {determinism_mode}")
        print(f"   Architecture: hidden={self.hidden_size}, heads={self.num_heads}, layers={self.num_layers}, kv_heads={self.num_kv_heads}")

    def _apply_determinism(self, mode):
        """Apply determinism settings based on mode.

        Modes:
            fast - No constraints, fastest execution (default)
            deterministic - Force CUDA deterministic algorithms
            strict        - CPU fallback for full reproducibility
        """
        if mode == "fast":
            # Default behavior — fastest, may be non-deterministic
            if self.device.startswith("cuda"):
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
            print(f"[Determinism] Mode: fast (no constraints)")

        elif mode == "deterministic":
            # Force deterministic algorithms where supported
            if self.device.startswith("cuda"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
            print(f"[Determinism] Mode: deterministic (forced deterministic CUDA algorithms)")

        elif mode == "strict":
            # Full CPU fallback for reproducibility
            self.device = "cpu"
            print(f"[Determinism] Mode: strict (CPU fallback for full reproducibility)")
            print(f"[Warning] Running on CPU — significantly slower")

        else:
            raise ValueError(f"Unknown determinism_mode: {mode}. Must be 'fast', 'deterministic', or 'strict'.")

    def _get_special_token_ids(self):
        """Get list of special token IDs to mask during generation."""
        special_ids = set()
        tok = self.tokenizer
        # Collect all defined special token IDs
        if tok.pad_token_id is not None:
            special_ids.add(tok.pad_token_id)
        if tok.eos_token_id is not None:
            special_ids.add(tok.eos_token_id)
        if tok.bos_token_id is not None:
            special_ids.add(tok.bos_token_id)
        if tok.unk_token_id is not None:
            special_ids.add(tok.unk_token_id)
        # Also mask the extra special tokens (151643-151645) that exist in model but not tokenizer vocab
        vocab_size = tok.vocab_size
        for sid in range(vocab_size, vocab_size + 500):
            if sid >= self.loader.header.get("architecture", {}).get("vocab_size", vocab_size + 500):
                break
            # Mask if it looks like a special token (decode test)
            try:
                decoded = tok.decode([sid])
                if decoded.startswith("<|") and decoded.endswith("|>"):
                    special_ids.add(sid)
            except Exception:
                pass
        return list(special_ids)

    def _auto_detect_architecture(self):
        arch = self.loader.header.get("architecture", {})
        if arch and all(k in arch for k in ['hidden_size', 'num_hidden_layers', 'num_attention_heads']):
            self.hidden_size = arch['hidden_size']
            self.num_layers = arch['num_hidden_layers']
            self.num_heads = arch['num_attention_heads']
            self.num_kv_heads = arch['num_key_value_heads']
            self.rope_theta = arch.get('rope_theta', 10000.0)
        else:
            self._detect_from_tensors()

        self.attention_layer.hidden_size = self.hidden_size
        self.attention_layer.num_heads = self.num_heads
        self.attention_layer.num_kv_heads = self.num_kv_heads
        self.attention_layer.rope_theta = getattr(self, 'rope_theta', 10000.0)

    def _detect_from_tensors(self):
        tensors = self.loader.header["tensors"]
        embed_shape = next((tensors[n]["shape"] for n in tensors if "embed_tokens" in n), None)
        self.hidden_size = embed_shape[1] if (embed_shape and "shape" in embed_shape) else 4096

        layer_nums = {int(name.split(".")[1]) for name in tensors if name.startswith("layers.") and name.split(".")[1].isdigit()}
        self.num_layers = len(layer_nums) if layer_nums else 28

        q_proj_shape = next((tensors[n]["shape"] for n in tensors if "self_attn.q_proj.weight" in n), None)
        k_proj_shape = next((tensors[n]["shape"] for n in tensors if "self_attn.k_proj.weight" in n), None)

        if q_proj_shape and k_proj_shape:
            self.num_heads = q_proj_shape[0] // 128
            self.num_kv_heads = k_proj_shape[0] // 128
        else:
            self.num_heads, self.num_kv_heads = 32, 8
        self.rope_theta = 10000.0

    def _get_cached_weight(self, name, target_dtype=torch.float16):
        if name not in self.weight_cache:
            weight = self.loader.get_matrix(name, target_device=self.device)
            if weight is None:
                raise RuntimeError(f"[Engine] Error: Weight matrix {name} not found!")
            self.weight_cache[name] = weight.to(target_dtype)
        return self.weight_cache[name]

    def prefill_and_freeze_context(self, prompt_tokens, system_kv=None):
        clear_storage_dir(self.storage_dir)
        self.gpu_kv_cache.clear()

        if system_kv is not None:
            self.cacher.save_global_root_cache(system_kv["k"], system_kv["v"])

        current_seq_len = len(prompt_tokens)
        all_tensor_names = list(self.loader.header["tensors"].keys())
        tensor_names = align_tensor_names(all_tensor_names)

        # Pre-save all layer caches (one file per layer) before transformer loop
        for layer_idx in range(self.num_layers):
            cache_path = self.loader.tensorManager._layer_cache_path(layer_idx)
            if not (os.path.exists(cache_path + ".bin") and os.path.exists(cache_path + ".meta.pt")):
                print(f"[JLLM] Saving layer {layer_idx} cache...")
                self.loader.tensorManager.save_layer_cache(layer_idx, self.loader)

        input_ids_t = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)[None, :]

        embed_w = self._get_cached_weight(tensor_names["embed"])
        hidden_t = embed_w[input_ids_t]

        for layer_idx in range(self.num_layers):
            hidden_t, new_kv = apply_transformer_layer(
                layer_idx, hidden_t, self.loader, self.attention_layer, self.device, current_seq_len
            )
            new_k, new_v = new_kv

            if self.cache_mode == "full_gpu":
                self.gpu_kv_cache[layer_idx] = (new_k.contiguous(), new_v.contiguous())
            else:
                self.cacher.save_layer_compressed_cache(layer_idx, new_k, new_v)

        torch.cuda.empty_cache()
        print("[JLLM] Prefill complete. System context pinned.")

    def generate_stream(self, prompt: str, target_context_window=None):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = input_ids[0].tolist()
        current_input_ids_t = input_ids
        current_seq_len = current_input_ids_t.shape[1]

        all_tensor_names = list(self.loader.header["tensors"].keys())
        tensor_names = align_tensor_names(all_tensor_names)

        if self.cache_mode != "full_gpu":
            self.prefill_and_freeze_context(prompt_tokens=prompt_tokens)

        print(f"\n[JLLM] Input: {prompt}")
        print(f"[JLLM] Starting engine phase [{self.cache_mode}]...\n" + "-" * 60)

        with torch.no_grad():
            if self.cache_mode == "full_gpu":
                self._generate_full_gpu(current_input_ids_t, current_seq_len, tensor_names)
            elif self.cache_mode == "1_3_gpu" or self.cache_mode == "1/3_gpu":
                self._generate_1on3_gpu(current_input_ids_t, current_seq_len, tensor_names)
            elif self.cache_mode == "no_cache":
                self._generate_no_cache(current_input_ids_t, current_seq_len, tensor_names)
            elif self.cache_mode == "cell":
                self._generate_cell_level(current_input_ids_t, current_seq_len, tensor_names)

        print("\n" + "-" * 60 + "\n[JLLM] Generation complete!")

    def _generate_full_gpu(self, current_input_ids_t, current_seq_len, tensor_names):
        embed_w = self._get_cached_weight(tensor_names["embed"])
        hidden_t = embed_w[current_input_ids_t]

        # Prefill pass logic
        for layer_idx in range(self.num_layers):
            hidden_t, new_kv = apply_transformer_layer(
                layer_idx, hidden_t, self.loader, self.attention_layer, self.device, current_seq_len
            )
            self.kv_cache[layer_idx] = (new_kv[0].contiguous(), new_kv[1].contiguous())

        final_norm_w = self._get_cached_weight(tensor_names["norm"])
        hidden_norm = rms_norm(hidden_t, final_norm_w)

        lm_head_w = self._get_cached_weight(tensor_names["lm_head"])
        logits_t = torch.matmul(hidden_norm[:, -1:, :], lm_head_w.T)

        vocab_size = self.tokenizer.vocab_size
        special_ids = self._get_special_token_ids()
        next_token_id_t = sample_token(logits_t, vocab_size, special_ids)

        token_id = int(next_token_id_t[0, 0])
        print(self.tokenizer.decode([token_id]), end="", flush=True)

        # Autoregressive Decode Pass
        for step in range(self.max_new_tokens - 1):
            current_seq_len += 1
            hidden_t = embed_w[next_token_id_t]

            for layer_idx in range(self.num_layers):
                history_k, history_v = self.kv_cache[layer_idx]
                
                hidden_t, new_kv = apply_transformer_layer(
                    layer_idx, hidden_t, self.loader, self.attention_layer, self.device, current_seq_len,
                    raw_flat_kv=(history_k, history_v)
                )
                
                # ====== SMART APPEND FIX (Amnesia Fix) ======
                new_k, new_v = new_kv[0], new_kv[1]
                if new_k.shape[1] == 1:
                    k_full = torch.cat([history_k, new_k], dim=1)
                    v_full = torch.cat([history_v, new_v], dim=1)
                else:
                    k_full, v_full = new_k, new_v
                    
                self.kv_cache[layer_idx] = (k_full.contiguous(), v_full.contiguous())

            hidden_norm = rms_norm(hidden_t, final_norm_w)
            logits_t = torch.matmul(hidden_norm, lm_head_w.T)

            next_token_id_t = sample_token(logits_t, vocab_size, special_ids)
            token_id = int(next_token_id_t[0, 0])

            if token_id == self.tokenizer.eos_token_id:
                break

            print(self.tokenizer.decode([token_id]), end="", flush=True)

    def _generate_1on3_gpu(self, current_input_ids_t, current_seq_len, tensor_names):
        self.cacher.read_only = True
        embed_w = self._get_cached_weight(tensor_names["embed"])
        final_norm_w = self._get_cached_weight(tensor_names["norm"])
        lm_head_w = self._get_cached_weight(tensor_names["lm_head"])

        n = self.num_layers
        group_size = max(1, n // 3)
        groups = [list(range(0, group_size)), list(range(group_size, group_size * 2)), list(range(group_size * 2, n))]
        
        kv_queue = []
        next_token_id_t = current_input_ids_t[:, -1:] if current_input_ids_t.shape[1] > 1 else current_input_ids_t
        vocab_size = self.tokenizer.vocab_size
        special_ids = self._get_special_token_ids()

        for step in range(self.max_new_tokens):
            end_window = current_seq_len // self.cacher.interval
            start_window = max(0, end_window - self.window_size)

            evict_idx = (step - 1) % 3
            load_idx = (step + 1) % 3
            kv_queue = [(lid, k, v) for lid, k, v in kv_queue if lid not in groups[evict_idx]]

            for layer_idx in groups[load_idx]:
                k_t, v_t = self.cacher.get_incremental_prefill_decompressed(layer_idx, start_window, end_window)
                if k_t is not None and v_t is not None:
                    kv_queue.append((layer_idx, k_t.to(self.device), v_t.to(self.device)))

            kv_map = {lid: (k, v) for lid, k, v in kv_queue}
            hidden_t = embed_w[next_token_id_t]

            for layer_idx in range(self.num_layers):
                kv = kv_map.get(layer_idx)
                if kv is None:
                    k_t, v_t = self.cacher.get_incremental_prefill_decompressed(layer_idx, start_window, end_window)
                    kv = (k_t.to(self.device), v_t.to(self.device)) if k_t is not None else None

                hidden_t, new_kv = apply_transformer_layer(
                    layer_idx, hidden_t, self.loader, self.attention_layer, self.device, current_seq_len,
                    raw_flat_kv=kv 
                )
                
                # In offload mode, the cacher must handle appending new states internally if it is saving dynamically.

            hidden_norm = rms_norm(hidden_t, final_norm_w)
            logits_t = torch.matmul(hidden_norm[:, -1:, :], lm_head_w.T)

            next_token_id_t = sample_token(logits_t, vocab_size, special_ids)
            token_id = int(next_token_id_t[0, 0])
            print(self.tokenizer.decode([token_id]), end="", flush=True)

            current_seq_len += 1
            if token_id == self.tokenizer.eos_token_id:
                break

    def _generate_no_cache(self, current_input_ids_t, current_seq_len, tensor_names):
        embed_w = self._get_cached_weight(tensor_names["embed"])
        final_norm_w = self._get_cached_weight(tensor_names["norm"])
        lm_head_w = self._get_cached_weight(tensor_names["lm_head"])
        vocab_size = self.tokenizer.vocab_size
        special_ids = self._get_special_token_ids()

        for step in range(self.max_new_tokens):
            hidden_t = embed_w[current_input_ids_t]

            for layer_idx in range(self.num_layers):
                hidden_t, _ = apply_transformer_layer(
                    layer_idx, hidden_t, self.loader, self.attention_layer, self.device, current_seq_len
                )

            hidden_norm = rms_norm(hidden_t, final_norm_w)
            logits_t = torch.matmul(hidden_norm[:, -1:, :], lm_head_w.T)

            next_token_id_t = sample_token(logits_t, vocab_size, special_ids)
            token_id = int(next_token_id_t[0, 0])
            print(self.tokenizer.decode([token_id]), end="", flush=True)

            current_input_ids_t = torch.cat([current_input_ids_t, next_token_id_t], dim=-1)
            current_seq_len += 1

            if token_id == self.tokenizer.eos_token_id:
                break

    def _generate_cell_level(self, current_input_ids_t, current_seq_len, tensor_names):
        """Cell-Level Generation: One token at a time with cached KV states.

        Prefill: uses sequence-level apply_transformer_layer (correct RoPE for all tokens).
        Decode:  uses cell-level transformer_layer_forward_cell (one token at a time).
        """
        embed_w = self._get_cached_weight(tensor_names["embed"])
        final_norm_w = self._get_cached_weight(tensor_names["norm"])
        lm_head_w = self._get_cached_weight(tensor_names["lm_head"])

        vocab_size = self.tokenizer.vocab_size
        special_ids = self._get_special_token_ids()

        # Pre-compute RoPE frequencies once (used in decode phase only)
        rope_freqs = compute_rope_freqs_cell(
            self.attention_layer.head_dim,
            self.attention_layer.rope_theta,
            self.device
        )

        # ===== PREFILL PHASE (sequence-level, correct RoPE for all tokens) =====
        hidden_t = embed_w[current_input_ids_t]

        kv_caches = {}
        for layer_idx in range(self.num_layers):
            hidden_t, new_kv = apply_transformer_layer(
                layer_idx, hidden_t, self.loader, self.attention_layer, self.device, current_seq_len
            )
            kv_caches[layer_idx] = (new_kv[0].contiguous(), new_kv[1].contiguous())

        # Final norm + logits + first token
        hidden_norm = rms_norm(hidden_t, final_norm_w)
        logits_t = torch.matmul(hidden_norm[:, -1:, :], lm_head_w.T)
        next_token_id_t = sample_token(logits_t, vocab_size, special_ids)
        token_id = int(next_token_id_t[0, 0])
        generated_tokens = [token_id]
        print(self.tokenizer.decode([token_id]), end="", flush=True)

        # ===== DECODE PHASE (cell-level, one token at a time) =====
        for step in range(self.max_new_tokens - 1):
            current_seq_len += 1
            current_position = current_seq_len - 1  # position of the new token

            # Embed single new token
            hidden_t = embed_w[next_token_id_t]

            for layer_idx in range(self.num_layers):
                input_ln_w = self.loader.get_matrix(
                    f"layers.{layer_idx}.input_layernorm.weight", target_device=self.device
                )
                post_ln_w = self.loader.get_matrix(
                    f"layers.{layer_idx}.post_attention_layernorm.weight", target_device=self.device
                )
                q_proj = self.loader.get_matrix(
                    f"layers.{layer_idx}.self_attn.q_proj.weight", target_device=self.device
                )
                k_proj = self.loader.get_matrix(
                    f"layers.{layer_idx}.self_attn.k_proj.weight", target_device=self.device
                )
                v_proj = self.loader.get_matrix(
                    f"layers.{layer_idx}.self_attn.v_proj.weight", target_device=self.device
                )
                o_proj = self.loader.get_matrix(
                    f"layers.{layer_idx}.self_attn.o_proj.weight", target_device=self.device
                )
                gate_proj = self.loader.get_matrix(
                    f"layers.{layer_idx}.mlp.gate_proj.weight", target_device=self.device
                )
                up_proj = self.loader.get_matrix(
                    f"layers.{layer_idx}.mlp.up_proj.weight", target_device=self.device
                )
                down_proj = self.loader.get_matrix(
                    f"layers.{layer_idx}.mlp.down_proj.weight", target_device=self.device
                )

                history_k, history_v = kv_caches[layer_idx]
                residual = hidden_t

                hidden_t, k_full, v_full = transformer_layer_forward_cell(
                    hidden_cell=hidden_t,
                    residual=residual,
                    q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,
                    gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj,
                    input_layernorm_weight=input_ln_w,
                    post_attention_layernorm_weight=post_ln_w,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.attention_layer.head_dim,
                    k_cache_cell=history_k,
                    v_cache_cell=history_v,
                    rope_freqs=rope_freqs,
                    current_position=current_position
                )
                kv_caches[layer_idx] = (k_full.contiguous(), v_full.contiguous())

            hidden_norm = rms_norm(hidden_t, final_norm_w)
            logits_t = torch.matmul(hidden_norm, lm_head_w.T)
            next_token_id_t = sample_token(logits_t, vocab_size, special_ids)
            token_id = int(next_token_id_t[0, 0])

            if token_id == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(token_id)
            print(self.tokenizer.decode([token_id]), end="", flush=True)