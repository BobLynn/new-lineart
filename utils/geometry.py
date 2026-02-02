# # utils/geometry.py
# import cv2
# import numpy as np
# from skimage.morphology import skeletonize
# import os
# import time

# def log_debug(msg):
#     with open("geometry_debug.log", "a", encoding='utf-8') as f:
#         f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

# def parse_gradio_sketch(original_img, sketch_dict):
#     """
#     gradio sketch 返回一個字典：{'image': 原圖, 'mask': 只有畫筆痕跡的黑白圖}
#     或者 ImageEditor 的字典：{'background': ..., 'layers': [...], 'composite': ...}
#     """
#     log_debug("--- parse_gradio_sketch called ---")
    
#     mask = None
    
#     if isinstance(sketch_dict, dict):
#         keys = list(sketch_dict.keys())
#         log_debug(f"Dict keys: {keys}")
        
#         # 1. 嘗試 'layers' (ImageEditor 標準格式)
#         if 'layers' in sketch_dict and sketch_dict['layers']:
#             log_debug(f"Found {len(sketch_dict['layers'])} layers")
#             # 圖層是 RGBA。我們需要合併所有圖層（筆觸）的 Alpha 通道
#             # 假設 layers[0] 具有正確的形狀
#             if len(sketch_dict['layers']) > 0 and sketch_dict['layers'][0] is not None:
#                 h, w = sketch_dict['layers'][0].shape[:2]
#                 log_debug(f"Layer 0 shape: {sketch_dict['layers'][0].shape}")
                
#                 combined_alpha = np.zeros((h, w), dtype=np.uint8)
                
#                 for i, layer in enumerate(sketch_dict['layers']):
#                     if layer is None: continue
#                     log_debug(f"Processing layer {i}, shape: {layer.shape}, dtype: {layer.dtype}")
                    
#                     if len(layer.shape) == 3 and layer.shape[2] == 4:
#                         # 使用 Alpha 通道
#                         alpha = layer[:, :, 3]
#                         log_debug(f"Layer {i} max alpha: {np.max(alpha)}")
#                         combined_alpha = np.maximum(combined_alpha, alpha)
#                     elif len(layer.shape) == 3 and layer.shape[2] == 3:
#                         # RGB 圖層 - 轉換為灰階並進行閾值處理
#                         # 如果瀏覽器返回 RGB 圖層，可能會發生這種情況
#                         gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
#                         # 假設任何非黑色的部分都是筆觸？或者檢查是否與空白不同？
#                         # 通常筆觸是有顏色的。
#                         log_debug(f"Layer {i} (RGB) max val: {np.max(gray)}")
#                         combined_alpha = np.maximum(combined_alpha, gray)
#                     elif len(layer.shape) == 2:
#                         # 灰階圖層？視為遮罩
#                         combined_alpha = np.maximum(combined_alpha, layer)
                
#                 max_val = np.max(combined_alpha)
#                 log_debug(f"Combined alpha max value: {max_val}")
                
#                 if max_val > 10:
#                     mask = combined_alpha
#                     log_debug("Mask extracted from layers")

#         # 2. 嘗試 'mask' 鍵 (Sketchpad 風格)
#         if mask is None and 'mask' in sketch_dict:
#             log_debug("Trying 'mask' key")
#             mask = sketch_dict['mask']

#         # 3. 回退到 'composite' (如果背景不是黑色會有風險)
#         if mask is None and 'composite' in sketch_dict:
#             log_debug("Trying 'composite' key")
#             comp = sketch_dict['composite']
#             if len(comp.shape) == 3 and comp.shape[2] == 4:
#                 mask = comp[:, :, 3]
#             else:
#                 if len(comp.shape) == 3:
#                     gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
#                 else:
#                     gray = comp
#                 # 風險：如果背景不是黑色，這可能會選中背景
#                 _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
#             log_debug("Mask extracted from composite")
                
#     elif isinstance(sketch_dict, np.ndarray):
#         log_debug("Input is ndarray")
#         comp = sketch_dict
#         if len(comp.shape) == 3 and comp.shape[2] == 4:
#             mask = comp[:, :, 3]
#         else:
#             if len(comp.shape) == 3:
#                 gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
#             else:
#                 gray = comp
#             _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
#     if mask is None:
#         log_debug("No mask found in sketch_dict")
#         return []
    
#     # 確保遮罩是單通道
#     if len(mask.shape) == 3:
#         mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

#     # 如果需要，調整大小以匹配原始圖像
#     if original_img is not None:
#         h, w = original_img.shape[:2]
#         if mask.shape[:2] != (h, w):
#             log_debug(f"Resizing mask from {mask.shape} to {(h, w)}")
#             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

#     _, binary = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
    
#     # 2. 骨架化 (Skeletonize) - 獲取單像素寬的路徑
#     skeleton = skeletonize(binary).astype(np.uint8)
    
#     # 3. 提取座標點
#     ys, xs = np.where(skeleton > 0)
#     points = list(zip(xs, ys))
#     log_debug(f"Skeleton points found: {len(points)}")
    
#     constraints = []
    
#     # 4. 計算切向量 (局部鄰域 PCA)
#     for x, y in points:
#         # 尋找 5x5 鄰域內的相鄰點
#         local_pts = []
#         for dy in range(-2, 3):
#             for dx in range(-2, 3):
#                 nx, ny = x+dx, y+dy
#                 if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
#                     if skeleton[ny, nx] > 0:
#                         local_pts.append((nx, ny))
        
#         if len(local_pts) >= 3:
#             # PCA 計算主方向
#             pts_array = np.array(local_pts, dtype=np.float32)
            
#             # 數據中心化
#             mean = np.mean(pts_array, axis=0)
#             centered = pts_array - mean
            
#             # 共變異數矩陣
#             cov = np.dot(centered.T, centered) / (len(pts_array) - 1)
            
#             # 特徵值和特徵向量
#             # eigh 按升序返回特徵值
#             evals, evecs = np.linalg.eigh(cov)
            
#             # 對應最大特徵值的特徵向量即為切線方向
#             tangent = evecs[:, 1] # 索引 1 是 2D 中最大的
            
#             ux, uy = tangent[0], tangent[1]
            
#             constraints.append((x, y, ux, uy))
            
#     # 記錄約束摘要
#     log_debug(f"Total constraints generated: {len(constraints)}")
#     if len(constraints) > 0:
#         log_debug(f"First constraint: {constraints[0]}")
                
#     # 不要降採樣，保留所有約束以增強影響力
#     return constraints


# utils/geometry.py
import cv2
import numpy as np
from skimage.morphology import skeletonize
import os
import time

# --- 輔助函數：記錄除錯訊息 ---
def log_debug(msg):
    # 將訊息寫入 geometry_debug.log，方便追蹤問題
    with open("geometry_debug.log", "a", encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

# --- 主要函數：解析筆觸 ---
def parse_gradio_sketch(original_img, sketch_dict):
    """
    解析 Gradio 畫板輸入，提取骨架並計算平滑的切向向量場約束。
    """
    log_debug("--- parse_gradio_sketch called (Optimized) ---")
    
    mask = None
    
    # 1. 遮罩提取邏輯 (處理各種 Gradio 輸入格式)
    if isinstance(sketch_dict, dict):
        # 處理 ImageEditor 的 layers
        if 'layers' in sketch_dict and sketch_dict['layers']:
            if len(sketch_dict['layers']) > 0 and sketch_dict['layers'][0] is not None:
                h, w = sketch_dict['layers'][0].shape[:2]
                combined_alpha = np.zeros((h, w), dtype=np.uint8)
                
                for layer in sketch_dict['layers']:
                    if layer is None: continue
                    
                    if len(layer.shape) == 3 and layer.shape[2] == 4:
                        # RGBA
                        combined_alpha = np.maximum(combined_alpha, layer[:, :, 3])
                    elif len(layer.shape) == 3 and layer.shape[2] == 3:
                        # RGB
                        gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
                        combined_alpha = np.maximum(combined_alpha, gray)
                    elif len(layer.shape) == 2:
                        # Grayscale
                        combined_alpha = np.maximum(combined_alpha, layer)
                
                if np.max(combined_alpha) > 10:
                    mask = combined_alpha

        # 處理 mask 鍵
        if mask is None and 'mask' in sketch_dict:
            mask = sketch_dict['mask']

        # 處理 composite 鍵 (最後手段)
        if mask is None and 'composite' in sketch_dict:
            comp = sketch_dict['composite']
            if len(comp.shape) == 3 and comp.shape[2] == 4:
                mask = comp[:, :, 3]
            else:
                gray = comp if len(comp.shape) == 2 else cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                
    elif isinstance(sketch_dict, np.ndarray):
        comp = sketch_dict
        if len(comp.shape) == 3 and comp.shape[2] == 4:
            mask = comp[:, :, 3]
        else:
            gray = comp if len(comp.shape) == 2 else cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
    if mask is None:
        log_debug("No mask found")
        return []
    
    # 確保格式正確 (轉為單通道並調整大小)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    if original_img is not None:
        h, w = original_img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 二值化
    _, binary = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
    
    # 2. 骨架化 (Skeletonize) - 獲取單像素寬的路徑
    skeleton = skeletonize(binary).astype(np.uint8)
    
    # 提取所有骨架點
    ys, xs = np.where(skeleton > 0)
    all_points = list(zip(xs, ys))
    log_debug(f"Skeleton points found: {len(all_points)}")
    
    constraints = []
    
    # --- 關鍵修改設定 ---
    # 降採樣步長：每隔 5 個點取一個，避免約束過於密集
    sample_step = 5 
    # 鄰域半徑：擴大到 9x9 (半徑4) 以獲得更平滑的方向
    radius = 4 
    
    h_img, w_img = mask.shape[:2]

    # 3. 計算切向量 (局部 PCA)
    # 使用 sample_step 進行迴圈，減少計算量
    for i in range(0, len(all_points), sample_step):
        x, y = all_points[i]
        
        # 定義局部窗口邊界
        y_min, y_max = max(0, y - radius), min(h_img, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(w_img, x + radius + 1)
        
        # 提取窗口內的骨架點
        patch = skeleton[y_min:y_max, x_min:x_max]
        py, px = np.where(patch > 0)
        
        # 至少需要 4 個點才能計算出可靠的方向
        if len(px) >= 4:
            # 建立座標矩陣
            pts_array = np.stack([px, py], axis=1).astype(np.float32)
            
            # PCA 核心計算
            mean = np.mean(pts_array, axis=0)
            centered = pts_array - mean
            cov = np.dot(centered.T, centered)
            
            # 計算特徵值與特徵向量
            evals, evecs = np.linalg.eigh(cov)
            
            # 最大特徵值對應的特徵向量即為切線方向
            tangent = evecs[:, 1]
            ux, uy = tangent[0], tangent[1]
            
            # 正規化
            norm = np.sqrt(ux**2 + uy**2)
            if norm > 1e-6:
                ux /= norm
                uy /= norm
                
                # --- 方向一致性修正 (Orientation Consistency) ---
                # 強制向量指向右側半平面 (x > 0)，若垂直則指向上方
                # 這能避免相鄰向量互相抵消
                if ux < 0 or (abs(ux) < 1e-5 and uy < 0):
                    ux = -ux
                    uy = -uy
                
                constraints.append((x, y, ux, uy))
            
    log_debug(f"Optimized constraints generated: {len(constraints)}")
    
    return constraints