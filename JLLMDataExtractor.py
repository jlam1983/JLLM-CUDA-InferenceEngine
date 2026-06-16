import os
import sys
import json
import gc
import torch
import numpy as np
from safetensors import safe_open
from pathlib import Path


def auto_detect_model_type(model_dir):
    """
    Summary: Auto-detect the model type from config.json or safetensors index file.
    理論描述: 自動偵測模型類型，優先從 config.json 或 safetensors.index.json 讀取模型架構設定。
    """
    config_paths = [
        os.path.join(model_dir, "config.json"),
        os.path.join(model_dir, "model.safetensors.index.json"),
    ]

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                model_type = config.get("model_type") or config.get("architectures", [None])[0]
                if model_type:
                    print(f"[Auto-Detect] model_type: {model_type}")
                    return model_type, config
            except Exception as e:
                print(f"[Auto-Detect] Warning: Could not read {config_path}: {e}")

    print("[Auto-Detect] Warning: Could not auto-detect, using 'Unknown'")
    return "Unknown", {}


def extract_architecture_from_config(config):
    """
    Summary: Extract model architecture parameters from the config dictionary.
    理論描述: 從設定字典中提取模型架構參數，包括隱藏層大小、層數、注意力頭數等關鍵維度資訊。
    """
    arch = {}
    arch['hidden_size'] = config.get('hidden_size', 4096)
    arch['num_hidden_layers'] = config.get('num_hidden_layers', 32)
    arch['num_attention_heads'] = config.get('num_attention_heads', 32)
    arch['num_key_value_heads'] = config.get('num_key_value_heads', 8)
    arch['vocab_size'] = config.get('vocab_size', 32000)
    arch['rope_theta'] = config.get('rope_theta', 10000.0)
    arch['intermediate_size'] = config.get('intermediate_size', arch['hidden_size'] * 4)
    arch['head_dim'] = config.get('head_dim', arch['hidden_size'] // arch['num_attention_heads'])

    print(f"[Auto-Detect] hidden_size: {arch['hidden_size']}")
    print(f"[Auto-Detect] num_layers: {arch['num_hidden_layers']}")
    print(f"[Auto-Detect] num_heads: {arch['num_attention_heads']}")
    print(f"[Auto-Detect] num_kv_heads: {arch['num_key_value_heads']}")
    print(f"[Auto-Detect] head_dim: {arch['head_dim']}")
    print(f"[Auto-Detect] vocab_size: {arch['vocab_size']}")
    return arch


def fast_quantile(x, q, max_samples=5_000_000):
    """
    Summary: Estimate quantiles of a tensor using random sampling for large tensors.
    理論描述: 對大張量使用隨機採樣估算分位數值，以加速計算並減少記憶體開銷。
    """
    if x.numel() <= max_samples:
        return torch.quantile(x.float(), q).to(x.dtype)
    
    flat = x.view(-1)
    idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device, dtype=torch.long)
    return torch.quantile(flat[idx].float(), q).to(x.dtype)


def extract_weights(model_dir: str, output_file: str, sample_size: int = 5_000_000):
    """
    Summary: Extract weights from safetensors files and write them into a JLLM-formatted file.
    理論描述: 將 safetensors 模型權重提取並轉換為 JLLM 格式，執行離群值偵測、256 層級量化映射與元數據封裝。
    """
    print(f"[Extractor] Starting extraction: {model_dir}")
    
    tensor_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    model_type, config = auto_detect_model_type(model_dir)
    arch = extract_architecture_from_config(config)

    jllm_header = {
        "model_type": model_type,
        "architecture": arch,
        "tensors": {}
    }

    HEADER_SIZE = 2 * 1024 * 1024  # 2MB
    current_offset = HEADER_SIZE

    with open(output_file, "wb") as out_f:
        out_f.write(b"\x00" * HEADER_SIZE)   # Reserve header space

        for file_name in tensor_files:
            file_path = os.path.join(model_dir, file_name)
            print(f"\n[Extractor] Processing {file_name}")

            with safe_open(file_path, framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    print(f"  → {tensor_name} ... ", end="")

                    # === Load tensor correctly ===
                    tensor = f.get_tensor(tensor_name)
                    if tensor.dtype != torch.float16:
                        tensor = tensor.to(torch.float16)

                    use_cuda = torch.cuda.is_available() and tensor.numel() > 20_000_000
                    if use_cuda:
                        tensor = tensor.cuda(non_blocking=True)

                    numel = tensor.numel()

                    # Outlier detection
                    abs_tensor = torch.abs(tensor)
                    outlier_threshold = fast_quantile(abs_tensor, 0.99, max_samples=sample_size)
                    outlier_mask = abs_tensor >= outlier_threshold

                    outliers = tensor[outlier_mask]
                    normal_vals = tensor[~outlier_mask]

                    # Quantile mapping (256 levels)
                    if normal_vals.numel() > 0:
                        q = torch.linspace(0.0, 1.0, 256, device=normal_vals.device)
                        mapping = fast_quantile(normal_vals, q, max_samples=sample_size)
                    else:
                        mapping = torch.zeros(256, dtype=torch.float16, device=tensor.device)

                    # Bucketize → uint8 indices
                    indices = torch.bucketize(normal_vals, mapping, right=True)
                    normal_indices = torch.clamp(indices, 0, 255).to(torch.uint8)

                    # Prepare CPU buffers
                    cpu_outliers = outliers.cpu().numpy().tobytes()
                    cpu_mask = np.packbits(outlier_mask.cpu().numpy().ravel(), bitorder='big').tobytes()
                    cpu_normal = normal_indices.cpu().numpy().tobytes()
                    cpu_mapping = mapping.cpu().numpy().tobytes()

                    # === Incremental sub_offsets (Critical Fix) ===
                    sub_offsets = {}
                    off = 0
                    for key, data in [("outliers", cpu_outliers), ("mask", cpu_mask),
                                    ("normal", cpu_normal), ("mapping", cpu_mapping)]:
                        length = len(data)
                        sub_offsets[key] = [off, off + length]
                        off += length

                    # Write to file
                    out_f.write(cpu_outliers)
                    out_f.write(cpu_mask)
                    out_f.write(cpu_normal)
                    out_f.write(cpu_mapping)

                    # Record metadata
                    jllm_header["tensors"][tensor_name] = {
                        "shape": list(tensor.shape),
                        "data_offset": current_offset,
                        "data_size": off,
                        "sub_offsets": sub_offsets
                    }

                    current_offset += off

                    # Cleanup
                    del tensor, abs_tensor, outlier_mask, outliers, normal_vals
                    del normal_indices, mapping, indices
                    gc.collect()

                    print(f"Done | Size: {off/1024/1024:.2f} MB")

    # Write JSON header
    header_bytes = json.dumps(jllm_header, ensure_ascii=False).encode('utf-8')
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header too big ({len(header_bytes)} > {HEADER_SIZE} bytes). Increase HEADER_SIZE.")

    with open(output_file, "r+b") as out_f:
        out_f.seek(0)
        out_f.write(header_bytes)

    print(f"\n[Extractor] Successfully saved to → {output_file}")
    print(f"Total tensors: {len(jllm_header['tensors'])}")


if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\j_lam\OneDrive\桌面\JProjects\JCUDA\src V5\src\Models\qwen"
    OUTPUT_JLLM = "jqwen.jllm"

    if len(sys.argv) >= 3:
        MODEL_PATH = sys.argv[1]
        OUTPUT_JLLM = sys.argv[2]

    if os.path.exists(MODEL_PATH):
        extract_weights(MODEL_PATH, OUTPUT_JLLM)
    else:
        print(f"[Error] Model path not found: {MODEL_PATH}")
        print("Usage: python JLLMDataExtractor.py <model_dir> <output.jllm>")