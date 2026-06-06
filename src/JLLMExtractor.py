import os
import json
import torch
import gc
from safetensors import safe_open

def extract_deepseek_weights(model_dir, output_file):
    """
    JLLM 權重手術刀：從原始 Safetensors 中精準提取 FP16 矩陣並格式化為自研 .jllm 格式
    """
    print(f"🔍 開始掃描原始模型目錄: {model_dir}")
    
    # 1. 尋找目錄下所有的 .safetensors 檔案
    tensor_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if not tensor_files:
        raise FileNotFoundError(f"❌ 在路徑 {model_dir} 下找不到任何 .safetensors 權重檔案，請確認路徑！")
    
    print(f"📦 偵測到 {len(tensor_files)} 個權重分片檔案。")
    
    # 2. 初始化自研的權重索引表表頭（Index Header）
    jllm_header = {
        "model_type": "DeepSeek-R1-Distill-Llama",
        "tensors": {}  # 用來記錄每個矩陣在我們新檔案中的絕對座標
    }
    
    # 建立乾淨的輸出二進位檔案
    with open(output_file, "wb") as out_f:
        # 預留 1MB 空間寫入 JSON 索引表頭
        HEADER_SIZE = 1024 * 1024 
        out_f.write(b"\x00" * HEADER_SIZE)
        
        current_offset = HEADER_SIZE
        
        # 3. 開始對真實模型進行矩陣解剖
        for file_name in tensor_files:
            file_path = os.path.join(model_dir, file_name)
            print(f"📖 正在手術解剖權重分片: {file_name} ...")
            
            # 使用 safe_open 以唯讀、零拷貝（Zero-Copy）方式打開矩陣
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for tensor_name in f.keys():
                    # 🎯【精準過濾】：唯讀取權重矩陣（Weight），暫時忽略 Bias 與優化器參數
                    if not tensor_name.endswith(".weight"):
                        continue
                        
                    tensor = f.get_tensor(tensor_name)
                    
                    # 強制對齊至 float16 半精度，並轉換為緊湊的 Raw Bytes
                    tensor_data = tensor.to(torch.float16).numpy().tobytes()
                    tensor_size = len(tensor_data)
                    
                    # 簡化張量名稱，去掉 model. 前綴，完美銜接 JLLMLoader 命名空間
                    clean_name = tensor_name.replace("model.", "")
                    
                    # 記錄絕對座標與幾何規格
                    jllm_header["tensors"][clean_name] = {
                        "shape": list(tensor.shape),
                        "offset": current_offset,
                        "size": tensor_size
                    }
                    
                    # 將矩陣的原始二進位數據流打入硬碟
                    out_f.write(tensor_data)
                    current_offset += tensor_size
                    
                    # 🚀【工業級優化】：即時斬斷臨時張量引用，防止解剖大模型時 CPU 記憶體溢出
                    del tensor, tensor_data
                
            # 分片處理完畢後，強迫垃圾回收器清理 CPU 記憶體殘渣
            gc.collect()
                    
        # 4. 回馬槍：把我們的自研索引表頭寫回檔案的最開頭 1MB 空間
        out_f.seek(0)
        header_bytes = json.dumps(jllm_header, ensure_ascii=False).encode('utf-8')
        if len(header_bytes) > HEADER_SIZE:
            raise RuntimeError(f"❌ 錯誤：自研表頭體積 ({len(header_bytes)} bytes) 超過預留的 1MB 空間！")
            
        # 寫入真正的表頭，剩下的空間用空字節補滿對齊
        out_f.write(header_bytes)
        out_f.write(b"\x00" * (HEADER_SIZE - len(header_bytes)))

    print(f"\n🎉 恭喜！自研二進位格式重組完畢！")
    print(f"💾 專屬權重庫已生成: {output_file}")
    print(f"📊 共成功提取並對齊了 {len(jllm_header['tensors'])} 個核心權重矩陣。")

if __name__ == "__main__":
    # 配置你的本地原始模型路徑與輸出目標
    MODEL_PATH = r"C:\Users\j_lam\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B\snapshots\6a6f4aa4197940add57724a7707d069478df56b1"
    OUTPUT_JLLM = "deepseek_8b_pure.jllm"
    
    if os.path.exists(MODEL_PATH):
        extract_deepseek_weights(MODEL_PATH, OUTPUT_JLLM)
    else:
        print(f"📢 提示：未偵測到預設路徑，請先將代碼中的 MODEL_PATH 修改為你電腦中真實的 Safetensors 目錄。")