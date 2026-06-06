import os
import json
import mmap
import math
import numpy as np
from transformers import AutoTokenizer # 只借用文字編碼功能，推理全自研
import cupy as cp

# =====================================================================
# 延續第三關的零記憶體映射加載器 (JLLM Loader)
# =====================================================================
class JLLMLoader:
    def __init__(self, jllm_filepath):
        self.filepath = jllm_filepath
        self.HEADER_SIZE = 1024 * 1024
        self.file_handle = open(jllm_filepath, "rb")
        self.mmapped_data = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        
        header_bytes = self.mmapped_data[0:self.HEADER_SIZE].rstrip(b'\x00')
        self.header = json.loads(header_bytes.decode('utf-8'))
        print(f"⚙️ [Vault Active] 成功掛載權重庫，內含 {len(self.header['tensors'])} 個矩陣。")
        # 在 __init__ 中加入 KV Cache 容器

    def get_matrix(self, tensor_name, target_device="cuda"):
        if tensor_name not in self.header["tensors"]:
            return None # 容許部分權重不存在時返回 None
        metadata = self.header["tensors"][tensor_name]
        raw_bytes = self.mmapped_data[metadata["offset"]:metadata["offset"]+metadata["size"]]
        weight_np = np.frombuffer(raw_bytes, dtype=np.float16).reshape(metadata["shape"])
        return cp.asarray(weight_np)

    def close(self):
        self.mmapped_data.close()
        self.file_handle.close()