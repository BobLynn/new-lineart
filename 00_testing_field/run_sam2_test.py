import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 設定基本參數
# 請將這裡換成你下載的模型權重路徑
# checkpoint_path = "sam2_hiera_large.pt"
checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "sam2_hiera_large.pt")
# 對應的模型設定檔名稱 (通常在庫裡面已經內建)
# 修正: Hydra 默認會從 configs 目錄開始搜尋，但 sam2 的 package 結構可能將 configs 包在內
# 根據錯誤訊息，Search path 包含 pkg://sam2
# 而我們看到實體文件在 site-packages/sam2/configs/sam2/sam2_hiera_l.yaml
# 這意味著相對於 pkg://sam2 (即 site-packages/sam2)，路徑應該是 "configs/sam2/sam2_hiera_l.yaml"
model_cfg = "configs/sam2/sam2_hiera_l.yaml"
# 你的圖片路徑
image_path = "horse_0000.png"

# 設定運算裝置 (有顯卡用 cuda，Mac 用 mps，都沒有就用 cpu)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"正在使用裝置: {device}，請稍等...")

# 2. 載入 SAM2 模型
sam2_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)

# 3. 初始化「自動遮罩生成器」
# 這個工具會自動掃描整張圖，找出所有物體
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,      # 掃描的密度 (64 -> 32)
    pred_iou_thresh=0.8,     # 信心門檻 (0.86 -> 0.8)
    stability_score_thresh=0.9, # 穩定性門檻 (0.92 -> 0.9)
    crop_n_layers=0,
    min_mask_region_area=100 # (50 -> 100)
)

# 4. 讀取圖片並轉換格式
image = cv2.imread(image_path)
if image is None:
    print("找不到圖片，請檢查路徑！")
    exit()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 轉成 RGB 格式

# 5. 開始生成遮罩 (Magic happens here!)
print("正在分割圖片中...")
masks = mask_generator.generate(image)

print(f"原始分割數量: {len(masks)}")

# 6. 過濾遮罩 (排除黑色背景與白色輪廓)
# 用戶需求: 不希望將圖片中的白色輪廓線條和主題以外的黑底納入SAM2分割的目標
filtered_masks = []
for ann in masks:
    m = ann['segmentation']
    
    # 計算 Mask 區域內的平均顏色
    # image 是 RGB 格式
    masked_pixels = image[m]
    if masked_pixels.size == 0:
        continue
        
    avg_color = np.mean(masked_pixels, axis=0)
    
    # 判斷是否為黑色背景 (RGB 數值都很低，例如 < 30)
    is_black = np.all(avg_color < 30)
    
    # 判斷是否為白色輪廓 (RGB 數值都很高，例如 > 240)
    is_white = np.all(avg_color > 240)
    
    if not is_black and not is_white:
        filtered_masks.append(ann)
    else:
        # Debug: 印出被過濾掉的顏色資訊
        color_str = "Black" if is_black else "White"
        print(f"過濾掉一個 {color_str} 區域，平均顏色: {avg_color}")
        pass

masks = filtered_masks
print(f"過濾後剩餘數量: {len(masks)}")

print(f"成功分割出 {len(masks)} 個區塊！")

# --- 以下是視覺化程式碼，讓你看看結果 ---

def show_anns(anns):
    if len(anns) == 0:
        return
    # 根據面積大小排序，確保小區塊疊在大區塊上面
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    # 建立一個透明圖層
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        # 隨機產生顏色
        color_mask = np.concatenate([np.random.random(3), [0.5]]) # Alpha = 0.5
        img[m] = color_mask
        
    ax.imshow(img)

# 顯示結果
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.title("SAM2 Segmentation Result")
plt.show()
