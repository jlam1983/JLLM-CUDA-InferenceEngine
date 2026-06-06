# JLLMLoader 📦

`JLLMLoader` 是整個 JLLM 引擎的數據入口。它負責直接與底層儲存裝置（如 NVMe SSD）進行高效的二進位數據對接，並解析大模型權重檔案（如 `.safetensors` 或自研二進位格式）的物理檔案頭（Header）。

## ⚙️ 核心職責 (Core Responsibilities)

1. **檔案頭安全解析 (Header Parsing)**：
   讀取權重檔案的初始二進位區塊，解析出包含所有張量（Tensors）名稱、幾何形狀（Shapes）、數據類型（DataType，如 FP16）以及檔案內物理偏移量（Offsets）的字典地圖。
2. **零記憶體浪費加載 (Memory-Mapped I/O)**：
   利用記憶體映射或高效的流式讀取，防止在 CPU 記憶體中重複堆疊未使用的權重，確保只有被引擎索取的矩陣才會被拉進記憶體。
3. **跨框架型態輸出 (Multi-Backend Output)**：
   向下游提供 `get_matrix(tensor_name, device)` 統一介面，能自動將磁碟中的二進位數據流精準定位，並直接在目標裝置上包裝成連續的矩陣物件。

## 🧠 設計思維 (Design Patterns)

在 Llama 3 或 DeepSeek 架構中，全模型包含數百個密集矩陣，檔案體積通常在十幾 GB 以上。`JLLMLoader` 的設計核心在於**「隨用隨讀，絕不常駐」**。它不維護任何矩陣的「肉身快取」，僅維護一份不到 1MB 的 `header` 快取導航圖，徹底解放了開機時的系統記憶體（RAM）壓力。