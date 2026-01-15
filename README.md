![Alt](streamline.0.png=300x)

## 📦 模型下載與設定 (Model Setup)

本專案使用 Meta 的 SAM 3 模型。由於模型檔案較大且需授權，請依照以下步驟自行下載：

1.  **申請權限**：
    前往 [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3) 頁面，登入帳號並申請模型使用權限（Accept License）。

2.  **下載模型**：
    權限通過後，請下載模型權重檔（例如 `sam3.pt`，請依需求選擇版本）。

3.  **放置位置**：
    請在專案根目錄下建立一個名為 `checkpoint` 的資料夾，並將下載的檔案放入其中。
    
    目錄結構範例：
    ```text
    new-lineart/
    ├── main.py
    ├── README.md
    └── checkpoint/      <-- 這裡新增checkpoint資料夾
        └── sam3.pt      <-- 將模型放在這裡
    ```

4.  **執行專案**：
    現在你可以正常執行程式了！