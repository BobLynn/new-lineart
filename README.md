# New Lineart - Streamline Lineart Generator

這是一個基於 **SAM 2 (Segment Anything Model 2)** 與 **張量場流線渲染 (Tensor Field Streamline Rendering)** 的圖像藝術化工具。透過 Gradio 網頁介面，使用者可以上傳圖片，利用 SAM 2 自動分割圖像區域，並生成極具美感的流線型線稿 (Line Art)。

## ✨ 功能特色

- **自動化圖像分割**：利用 Meta 的 SAM 2 模型強大的分割能力，自動識別圖像中的不同區域。
- **互動式選擇**：在網頁介面上點選感興趣的分割區域，即時預覽。
- **張量場生成**：根據圖像結構計算張量場 (Tensor Field)，引導線條走向。
- **流線渲染 (Streamline Rendering)**：基於張量場生成平滑、自然的流線藝術效果。
- **參數調整**：可調整線條密度、長度與平滑度，客製化輸出結果。

## 🛠️ 安裝與部署

### 1. 環境要求
- Windows / Linux / macOS
- Python 3.9+
- CUDA 支援的 GPU (強烈建議，用於加速 SAM 2 推理)

### 2. 下載專案
```bash
git clone <repository_url>
cd new_lineart
```

### 3. 安裝依賴
建議建立一個虛擬環境 (Virtual Environment) 進行安裝：

```bash
# 建立虛擬環境
python -m venv venv
# 啟用虛擬環境 (Windows)
.\venv\Scripts\activate
# 啟用虛擬環境 (Linux/macOS)
source venv/bin/activate

# 安裝專案依賴
pip install -r requirements.txt
```

> **注意**：本專案依賴 Meta 的 `sam2` 套件。如果 `requirements.txt` 中未包含或安裝失敗，請參考 [SAM 2 官方倉庫](https://github.com/facebookresearch/sam2) 進行安裝。通常可以透過以下指令安裝：
> ```bash
> pip install git+https://github.com/facebookresearch/sam2.git
> ```

### 4. 模型權重 (Checkpoints)
請確保您已下載 SAM 2 的模型權重與設定檔，並放置於正確位置：

- **模型權重 (.pt)**：放置於 `00_testing_field/sam2_hiera_large.pt` (預設路徑，可於 `app.py` 中修改)
- **設定檔 (.yaml)**：放置於 `configs/sam2/sam2_hiera_l.yaml`

您可以從 [SAM 2 Checkpoints](https://github.com/facebookresearch/sam2#model-checkpoints) 下載對應的模型。

## 🚀 執行教學

完成安裝後，執行 `app.py` 啟動 Gradio 網頁介面：

```bash
python app.py
```

啟動成功後，終端機將顯示本地訪問地址 (通常為 `http://127.0.0.1:7860`)，請用瀏覽器開啟該網址。

### 使用流程
1. **上傳圖片**：在左側面板上傳您想要轉換的圖片。
2. **生成分割**：系統會自動呼叫 SAM 2 進行分割 (可能需要幾秒鐘)。
3. **選擇區域**：點選圖片上的分割區塊 (選中區域會高亮顯示)。
4. **生成線稿**：點擊「生成線稿」按鈕，右側將顯示渲染結果。
5. **調整參數**：調整線條密度或平滑度，重新生成以獲得最佳效果。

## 📂 專案結構

```
new_lineart/
├── app.py                  # 主程式 (Gradio Web UI)
├── core/                   # 核心邏輯模組
│   ├── segmentation.py     # SAM 2 分割引擎封裝
│   ├── tensor_solver.py    # 張量場計算
│   ├── renderer.py         # 流線渲染器
│   └── interactive_window.py
├── utils/                  # 工具函式庫
├── configs/                # 模型設定檔
├── checkpoints/            # (可選) 模型存放區
├── 00_testing_field/       # 測試腳本與臨時模型存放區
└── requirements.txt        # 專案依賴列表
```

## 📝 備註

- 本專案主要用於實驗與藝術創作，線條生成的品質取決於圖像的複雜度與分割的準確性。
- 若遇到 GPU 記憶體不足 (OOM) 的問題，請嘗試使用較小的 SAM 2 模型 (如 `sam2_hiera_small`) 並修改 `app.py` 中的設定。
