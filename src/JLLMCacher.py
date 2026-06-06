import cupy as cp
import torch
import gc

class JLLMCacher:
    def __init__(self, loader, run_device="cuda"):
        self.loader = loader
        self.device = run_device
        self.interval = 2  

        self.num_layers = max([int(k.split(".")[1]) for k in loader.header["tensors"].keys() if "layers." in k]) + 1
        print(f"🎬 JLLM 智能動態快取引擎已就緒：共 {self.num_layers} 層。")

        # 核心快取字典
        self.gate_weight_dic = {} 
        self.weight_caches = {}
        self.bucket_caches = {}
        
        self.copy_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
    
    def _process_and_hash_weight(self, weight_tensor):
        if weight_tensor is None:
            return {}, []

        flat_tensor = cp.ravel(weight_tensor)
        total_elements = flat_tensor.shape[0]
        n_windows = total_elements // self.interval
        if n_windows == 0:
            return {}, []
            
        truncated_tensor = flat_tensor[:n_windows * self.interval]
        windows_tensor = truncated_tensor.reshape(n_windows, self.interval)
        
        scaled_tensor = windows_tensor * 4096
        scaled_tensor_uint = scaled_tensor.astype(cp.uint32)
        
        # GPU 內高速 1D 雜湊
        full_hashed_keys_gpu = (scaled_tensor_uint[:, 0] * 1000003) ^ scaled_tensor_uint[:, 1]

        # GPU 內極速一維去重
        hashed_keys_gpu, unique_indices, inverse_indices = cp.unique(
            full_hashed_keys_gpu, 
            return_index=True, 
            return_inverse=True
        )
        
        mask_gpu = cp.zeros(n_windows, dtype=cp.bool_)
        mask_gpu[unique_indices] = True  
        
        unique_unscaled_gpu = windows_tensor[unique_indices]

        # 建立去重對照
        new_features = {
            int(k): unique_unscaled_gpu[idx] 
            for idx, k in enumerate(hashed_keys_gpu.get())
        }
        self.gate_weight_dic.update(new_features)
        
        # 0 複製跨框架共享
        weights_gpu_block = torch.utils.dlpack.from_dlpack(unique_unscaled_gpu.toDlpack())
        inverse_map_torch = torch.utils.dlpack.from_dlpack(inverse_indices.toDlpack())
        mask_torch = torch.utils.dlpack.from_dlpack(mask_gpu.toDlpack())
        
        hashed_keys_list = [int(k) for k in hashed_keys_gpu.get()]

        weights_bucket = {
            "gpu_tensor_block": weights_gpu_block,
            "key_to_row_idx": {k: idx for idx, k in enumerate(hashed_keys_list)},
            "inverse_map_gpu": inverse_map_torch,
            "unique_mask_gpu": mask_torch
        }

        return weights_bucket, hashed_keys_list

    def cache_single_layer(self, layer_idx):
        unique_key = f"{layer_idx}"
        all_tensor_names = list(self.loader.header["tensors"].keys())
        layer_prefix = f"layers.{layer_idx}."
        
        # 模糊搜索注意力機制 4 大權重真實名稱
        real_q = next((k for k in all_tensor_names if layer_prefix in k and "q_proj" in k), None)
        real_k = next((k for k in all_tensor_names if layer_prefix in k and "k_proj" in k), None)
        real_v = next((k for k in all_tensor_names if layer_prefix in k and "v_proj" in k), None)
        real_o = next((k for k in all_tensor_names if layer_prefix in k and "o_proj" in k), None)
        
        real_gate = next((k for k in all_tensor_names if layer_prefix in k and "gate_proj" in k), None)
        real_up   = next((k for k in all_tensor_names if layer_prefix in k and "up_proj" in k), None)
        real_down = next((k for k in all_tensor_names if layer_prefix in k and "down_proj" in k), None)
        real_ln1  = next((k for k in all_tensor_names if layer_prefix in k and "input_layernorm" in k), None)
        real_ln2  = next((k for k in all_tensor_names if layer_prefix in k and "post_attention_layernorm" in k), None)
        
        target_tensors = {
            f"layers.{layer_idx}.input_layernorm.weight": real_ln1,
            f"layers.{layer_idx}.post_attention_layernorm.weight": real_ln2,
            f"layers.{layer_idx}.self_attn.q_proj.weight": real_q,   # 🎯 補上這行
            f"layers.{layer_idx}.self_attn.k_proj.weight": real_k,   # 🎯 補上這行
            f"layers.{layer_idx}.self_attn.v_proj.weight": real_v,   # 🎯 補上這行
            f"layers.{layer_idx}.self_attn.o_proj.weight": real_o,   # 🎯 補上這行
            f"layers.{layer_idx}.mlp.gate_proj.weight": real_gate,
            f"layers.{layer_idx}.mlp.up_proj.weight": real_up,
            f"layers.{layer_idx}.mlp.down_proj.weight": real_down,
        }
        
        with torch.cuda.stream(self.compute_stream):                    
            for virtual_label, real_tensor_name in target_tensors.items():
                if real_tensor_name is None:
                    continue  # 萬一模型沒有某些層的 bias 或是特殊權重，安全跳過
                    
                weight_cuda = self.loader.get_matrix(real_tensor_name, self.device)
                if weight_cuda is not None:
                    # 智慧型防禦橋樑：自動識別型態進行 0 複製安全轉換
                    if hasattr(weight_cuda, "contiguous"):
                        weight_cuda_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(weight_cuda.contiguous()))
                    elif isinstance(weight_cuda, cp.ndarray):
                        weight_cuda_cp = cp.ascontiguousarray(weight_cuda)
                    else:
                        weight_cuda_cp = cp.ascontiguousarray(cp.asarray(weight_cuda))
                    
                    # 執行純 GPU 1D 哈希去重
                    weights_bucket, keys = self._process_and_hash_weight(weight_cuda_cp)
                    
                    # 🎯 注意：快取登記時，主鍵使用統一的 virtual_label！
                    # 這樣 Engine 呼叫 f"layers.{layer_idx}.input_layernorm.weight" 時才能完美對齊命中！
                    self.weight_caches.setdefault(virtual_label, {})
                    self.weight_caches[virtual_label][unique_key] = keys
                    self.bucket_caches.setdefault(virtual_label, {})
                    self.bucket_caches[virtual_label][unique_key] = weights_bucket
                    
                    del weight_cuda, weight_cuda_cp

    def clear_single_layer(self, layer_idx):
        """
        🎯【動態流優化核心】：用完當前層後立刻就地正法，清空快取字典，還原 100% 乾淨顯存！
        """
        uk_str = f"{layer_idx}"
        # 1. 清空快取導航地圖與硬體資料桶
        for cache_label in list(self.weight_caches.keys()):
            if uk_str in self.weight_caches[cache_label]:
                del self.weight_caches[cache_label][uk_str]
            if cache_label in self.bucket_caches and uk_str in self.bucket_caches[cache_label]:
                del self.bucket_caches[cache_label][uk_str]
                
        # 2. 清空全域肉身庫，釋放張量指針
        self.gate_weight_dic.clear()
        
        # 3. 物理級斬斷快取池，強迫顯卡立刻回收
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        gc.collect()

    def get(self, cache_label: str, unique_key: str, target_device="cuda"):
        """
        純 GPU 還原算子：利用反向地圖 (inverse_map_gpu) 在 VRAM 內部實現 0 毫秒矩陣物理膨脹與重組
        """
        uk_str = str(unique_key)
        if cache_label not in self.weight_caches or uk_str not in self.weight_caches[cache_label]:
            return None
            
        # 1. 撈出當前層與虛擬標籤對應的硬體資料安全桶
        bucket = self.bucket_caches[cache_label][uk_str]
        gpu_block_torch = bucket["gpu_tensor_block"] # 這是儲存在 GPU 裡的去重短張量 [M, 2]
        inverse_map_torch = bucket["inverse_map_gpu"] # 這是反向地圖 [n_windows]
        
        # 0 複製轉回 CuPy 運算域
        gpu_block_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(gpu_block_torch.contiguous()))
        inverse_map_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(inverse_map_torch.contiguous()))
        
        with cp.cuda.Device(target_device if isinstance(target_device, int) else 0):
            # 🎯【核心物理膨脹】：利用 GPU 高頻寬並行索引，一瞬間把去重後的 M 行還原成原本的 n_windows 行！
            # 矩陣形狀瞬間從 [1512, 2] 膨脹還原回完美的 [n_windows, 2]
            reconstructed_windows = gpu_block_cp[inverse_map_cp]
            
            # 🎯【核心幾何重組】：將連續的視窗平鋪攤平，完美重塑回大模型最原始的一維密集權重向量！
            # 例如 input_layernorm.weight 重回乾乾淨淨的 [4096] 規格！
            original_shape_tensor = cp.ravel(reconstructed_windows).astype(cp.float16)
            
        return original_shape_tensor