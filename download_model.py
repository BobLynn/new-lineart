# download_model.py
from huggingface_hub import snapshot_download
import os

# 設定下載目錄
save_dir = "./checkpoint"
os.makedirs(save_dir, exist_ok=True)

print("正在從 Hugging Face 下載 SAM 3 模型...")
print("請確保你已經執行過 'huggingface-cli login' 並取得權限。")

try:
    # 這裡只下載特定的檔案，避免下載整個倉庫
    snapshot_download(
        repo_id="facebook/sam3", 
        allow_patterns=["*.pt"], # 只下載權重檔
        local_dir=save_dir,
        local_dir_use_symlinks=False
    )
    print(f"下載完成！模型已儲存於 {save_dir}")
except Exception as e:
    print(f"下載失敗：{e}")
    print("請確認你是否有權限存取該模型，或手動下載。")