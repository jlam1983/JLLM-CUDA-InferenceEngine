import os
import numpy as np
from transformers import AutoTokenizer # 只借用文字編碼功能，推理全自研
import cupy as cp
from src.JLLMLoader import *
from src.JLLMInferenceEngine import *
from src.JLLMCacher import *

# =====================================================================
# 啟動自研大模型引擎測試
# =====================================================================
if __name__ == "__main__":
    JLLM_FILE = "deepseek_8b_pure.jllm"
    # 請替換成你電腦上真實的 DeepSeek 模型目錄（需要裡面的 tokenizer 檔案）
    TOKENIZER_DIR = r"C:\Users\j_lam\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B\snapshots\6a6f4aa4197940add57724a7707d069478df56b1"
    
    if os.path.exists(JLLM_FILE) and os.path.exists(TOKENIZER_DIR):
        
        # 初始化加載器
        loader = JLLMLoader(JLLM_FILE)
        
        cacher = JLLMCacher(loader, "cuda")

        # 啟動我們完全控制的引擎
        engine = JLLMInferenceEngine(loader, cacher, tokenizer_path=TOKENIZER_DIR, run_device="cuda", max_new_tokens=10)
        
        # 讓它開口說話！(測試生成 10 個 token)
        engine.generate_stream(prompt="IT technology trend is")
        
        loader.close()
    else:
        print("📢 請確認：1. 第二關生成的 .jllm 檔案存在。 2. TOKENIZER_DIR 已改為你電腦中的模型路徑。")