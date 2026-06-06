# JMultiHeadAttention ⚡

`JMultiHeadAttention` 是專為大模型自注意力機制量身手搓的純 GPU 原生（CUDA-Native）加速算子。它解決了跨框架記憶體拼接與複雜解碼遮罩的效能瓶頸。

## ⚙️ 核心技術演進 (Technical Deep-Dive)

### 1. GQA (Grouped-Query Attention) 幾何自適應
本算子徹底拋棄了死板的硬編碼尺寸。在加載 Q、K、V 投影權重時，它會利用 `-1` 動態形狀推導（Dynamic Shape Inference）自動看穿 K/V 矩陣的真實頭數。無論模型是標準 MHA 還是 Llama 3 官方最核心的 8 頭 GQA 架構，算子都能全自動識別，並在計算注意力分數前透過 **`cp.repeat`** 在顯存內部將 K/V 廣播拉伸，與 Q 頭完美共振嚙合。

### 2. 智慧型解碼遮罩分流 (Causal Mask Division)
在模型每秒吐字（Decode 階段，`seq_len == 1`）時，當前 Token 理應能看見過去所有的歷史 KV 快取。本算子引入了智慧分流：在 Decode 階段**自動跳過複雜的 `cp.triu` 上三角矩陣建立與 `cp.where` 遮罩覆蓋算子**。這在長文本推理時，能為 GPU 節省出大量的無效算力。

### 3. 純 CuPy 閉環 KV Cache
為了防止顯存大洩漏，本算子將 `past_kv` 的生命週期 100% 鎖死在純原生的 CuPy 狀態。歷史快取與當前快取的拼接直接在 CuPy 底層 C++ 發生，徹底切斷了頻繁執行「CuPy ↔ PyTorch」轉換引發的顯存碎片化（VRAM Fragmentation）大爆炸。