import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor # 改用指定預測器
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 設定基本參數
# checkpoint_path = "sam2_hiera_large.pt"
checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "sam2_hiera_large.pt")
# 維持你原本設定的路徑修正
model_cfg = "configs/sam2/sam2_hiera_l.yaml"
image_path = "horse_0000.png"

# 設定運算裝置
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"正在使用裝置: {device}，請稍等...")

# 2. 載入 SAM2 模型
# 注意：這裡改用 SAM2ImagePredictor，因為我們要指定它去切哪裡，而不是讓它全圖亂切
sam2_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=True)
predictor = SAM2ImagePredictor(sam2_model)

# 3. 讀取圖片並前處理 (關鍵步驟：找出我們要的色塊在哪裡)
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    print("找不到圖片，請檢查路徑！")
    exit()
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print("正在分析圖片結構...")

# --- OpenCV 處理區段 (負責忽略白線與黑底) ---
# 3.1 轉灰階
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 3.2 二值化：過濾掉全黑背景 (像素值 > 10 才算有東西)
_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# 3.3 腐蝕 (Erosion)：這是去除白線的關鍵！
# 原理：讓白色區域向內縮。因為白線很細，一縮就不見了；色塊很大，縮了還在。
kernel = np.ones((7, 7), np.uint8) # 7x7 的核心，專門用來吃掉粗線條
eroded = cv2.erode(thresh, kernel, iterations=2) # 執行兩次確保線條斷開

# 3.4 找出剩下色塊的中心點
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
input_points = []

for cnt in contours:
    # 忽略太小的雜點
    if cv2.contourArea(cnt) < 100:
        continue
    # 計算質心 (中心點)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        input_points.append([cX, cY])

print(f"偵測到 {len(input_points)} 個主要色塊，準備進行分割...")

# 4. 開始生成遮罩 (使用 SAM2 針對每個點進行精確分割)
predictor.set_image(image_rgb)
masks_result = []

for point in input_points:
    # 告訴 SAM：這個座標 (point) 是前景 (label=1)，請幫我切出來
    masks, scores, logits = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=False # 我們只要一個最準確的遮罩
    )
    masks_result.append(masks[0])

print(f"成功分割出 {len(masks_result)} 個區塊！")

# --- 以下是視覺化程式碼 (已更新為適應新的輸出格式) ---

def show_results(image, masks, points):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    # 畫出每個遮罩
    for mask in masks:
        # 產生隨機顏色 (R, G, B, Alpha)
        color = np.concatenate([np.random.random(3), [0.6]])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    # (選用) 畫出我們告訴 SAM 的中心點，讓你確認位置對不對
    points_np = np.array(points)
    if len(points_np) > 0:
        ax.scatter(points_np[:, 0], points_np[:, 1], color='red', marker='*', s=100, label='Prompt Points')
    
    plt.axis('off')
    plt.title(f"SAM2 Segmentation (No Lines/No Background)\nFound {len(masks)} blocks")
    plt.legend()
    plt.show()

# 顯示結果
show_results(image_rgb, masks_result, input_points)