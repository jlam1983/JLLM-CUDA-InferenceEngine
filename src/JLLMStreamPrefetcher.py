import torch
import cupy as cp

class JLLMStreamPrefetcher:
    def __init__(self, engine):
        self.engine = engine
        self.loader = engine.loader
        
        # 建立兩個獨立的硬體流
        self.compute_stream = torch.cuda.Stream()
        self.copy_stream = torch.cuda.Stream()
        
        # 預載暫存器 (用來放下一層已經就緒的權重)
        self.next_layer_weights = None

    def prefetch_next_layer(self, next_layer_idx):
        """利用 Copy Stream 在背景默默搬運下一層矩陣，完全不卡 CPU 也不卡計算"""
        if next_layer_idx >= self.engine.num_layers:
            self.next_layer_weights = None
            return

        target_tensors = {
            "gate_proj": f"layers.{next_layer_idx}.mlp.gate_proj.weight",
            "down_proj": f"layers.{next_layer_idx}.mlp.down_proj.weight"
        }

        # 切換到搬運流
        with torch.cuda.stream(self.copy_stream):
            weights_bucket = {}
            for label, tensor_name in target_tensors.items():
                # 🎯 先讀到 CPU RAM (這一步極快，因為 mmap 已經映射了)
                weight_cpu = self.loader.get_matrix(tensor_name, target_device="cpu")
                if weight_cpu is not None:
                    # non_blocking=True 是關鍵！
                    # 它命令 Copy Engine 在背景把權重默默推進 GPU，Python 代碼不需要在這裡等待！
                    weights_bucket[label] = weight_cpu.pin_memory().to("cuda", non_blocking=True)
            
            self.next_layer_weights = weights_bucket

    def run_pipeline(self, max_new_tokens):
        interval = 2
        
        for token_idx in range(max_new_tokens):
            pre = token_idx * max_new_tokens
            
            # 🎬 啟動首層預載
            self.prefetch_next_layer(next_layer_idx=0)
            
            for layer_idx in range(self.engine.num_layers):
                # 1. 確保上一層的背景搬運已經完全結束（硬體同步點）
                self.copy_stream.synchronize()
                
                # 2. 把剛剛在背景搬好的矩陣拿出來用
                current_weights = self.next_layer_weights
                
                # 3. 瞬間向硬體發出指令：立刻在背景開始搬運【再下一層】的矩陣！
                self.prefetch_next_layer(next_layer_idx = layer_idx + 1)
                
                # 4. 切換到計算流，飛速處理當前層的哈希與運算
                with torch.cuda.stream(self.compute_stream):
                    unique_key = pre + layer_idx
                    
                    for cache_label, weight_cuda in current_weights.items():
                        # 在這裡執行你的哈希和矩陣運算，此時 weight_cuda 早就在 GPU 裡等你了！
                        keys = self.engine.cacher._process_and_hash_weight(pre, layer_idx, interval, weight_cuda)
                        self.engine.cacher.weight_caches[cache_label][unique_key] = keys
                        del weight_cuda
                
                # 5. 呼吸式清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()