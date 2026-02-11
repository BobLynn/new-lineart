# utils/geometry.py
from typing import Any


import cv2  # 導入 OpenCV 庫，用於圖像處理
import numpy as np  # 導入 NumPy 庫，用於數值計算和矩陣操作
from skimage.morphology import skeletonize  # 從 skimage 導入骨架化函數，用於提取線條骨架
import os  # 導入 os 模組，用於操作系統相關功能
import time  # 導入 time 模組，用於時間相關操作

def log_debug(msg):
    """
    記錄調試訊息到日誌文件。
    """
    # 以追加模式打開日誌文件，指定編碼為 utf-8
    with open("geometry_debug.log", "a", encoding='utf-8') as f:
        # 寫入帶有時間戳的訊息
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
def _save_mask_img(mask, tag):
    ts = time.strftime('%Y%m%d_%H%M%S')
    fname = f"debug_mask_{tag}_{ts}.png"
    try:
        cv2.imwrite(fname, mask)
        log_debug(f"Mask image saved: {fname}")
    except Exception as e:
        log_debug(f"Failed to save mask image ({tag}): {e}")

def parse_gradio_sketch(original_img, sketch_dict):
    """
    解析 Gradio 畫板的輸出，提取筆觸路徑並計算方向約束。
    
    參數:
        original_img: 原始圖像 (numpy array)
        sketch_dict: Gradio 返回的字典，包含畫板資訊
        
    返回:
        constraints: 約束列表，每個元素為 (x, y, ux, uy)
    """
    # 記錄函數調用
    log_debug("--- parse_gradio_sketch called ---")
    
    mask = None  # 初始化遮罩變量
    
    # 檢查輸入是否為字典類型
    if isinstance(sketch_dict, dict):
        keys = list(sketch_dict.keys())  # 獲取字典的所有鍵
        log_debug(f"Dict keys: {keys}")  # 記錄鍵名（用於除錯）
        
        # 1. 嘗試處理 'layers' 鍵 (Gradio ImageEditor 的標準格式)
        if 'layers' in sketch_dict and sketch_dict['layers']:
            log_debug(f"Found {len(sketch_dict['layers'])} layers")  # 記錄圖層數量
            if 'mask' in sketch_dict:
                log_debug(f"Found {len(sketch_dict['mask'])} masks")
                try:
                    log_debug(f"Mask shapes: {[m.shape for m in sketch_dict['mask']]}")
                except Exception as e:
                    log_debug(f"Mask shapes logging failed: {e}")
            # 圖層通常是 RGBA 格式。我們需要合併所有圖層（筆觸）的 Alpha 通道
            
            # 檢查圖層列表是否非空且第一個圖層存在
            if len(sketch_dict['layers']) > 0 and sketch_dict['layers'][0] is not None:
                # 獲取圖像的高度和寬度
                h, w = sketch_dict['layers'][0].shape[:2]
                log_debug(f"Layer 0 shape: {sketch_dict['layers'][0].shape}")  # 記錄圖層形狀
                print(f"Layer 0 shape: {sketch_dict['layers'][0].shape}")       # HSIN: print第一個圖層形狀
                
                # 初始化合併後的 Alpha 通道，全為 0
                combined_alpha = np.zeros((h, w), dtype=np.uint8)
                
                # 遍歷所有圖層
                for i, layer in enumerate(sketch_dict['layers']):
                    if layer is None: continue  # 跳過空圖層
                    log_debug(f"Processing layer {i}, shape: {layer.shape}, dtype: {layer.dtype}")  # 記錄當前處理的圖層資訊
                    
                    # 情況 A: RGBA 圖像 (4通道)
                    if len(layer.shape) == 3 and layer.shape[2] == 4:
                        # 提取 Alpha 通道 (第4個通道)
                        alpha = layer[:, :, 3]
                        log_debug(f"Layer {i} max alpha: {np.max(alpha)}")  # 記錄最大 Alpha 值
                        # 將當前 Alpha 通道與已合併的 Alpha 通道取最大值，疊加筆觸
                        combined_alpha = np.maximum(combined_alpha, alpha)
                    # 情況 B: RGB 圖像 (3通道)
                    elif len(layer.shape) == 3 and layer.shape[2] == 3:
                        # 如果瀏覽器返回 RGB 圖層，將其轉換為灰階
                        gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
                        log_debug(f"Layer {i} (RGB) max val: {np.max(gray)}")  # 記錄最大灰階值
                        # 將灰階值作為 Alpha 值合併
                        combined_alpha = np.maximum(combined_alpha, gray)
                    # 情況 C: 灰階圖像 (2維)
                    elif len(layer.shape) == 2:
                        # 直接視為遮罩合併
                        combined_alpha = np.maximum(combined_alpha, layer)
                
                # 獲取合併後的最大值
                max_val = np.max(combined_alpha)
                log_debug(f"Combined alpha max value: {max_val}")  # 記錄最大值
                
                # 如果最大值大於 10，認為提取到了有效的筆觸遮罩
                if max_val > 10:
                    mask = combined_alpha
                    log_debug("Mask extracted from layers")  # 記錄成功提取遮罩
                    _save_mask_img(mask, "layers")
                else:
                    log_debug("No valid mask found from layers")  # 記錄未找到有效遮罩

        # 2. 如果尚未提取到遮罩，嘗試 'mask' 鍵 (舊版或 Sketchpad 風格)
        if mask is None and 'mask' in sketch_dict:
            log_debug("Trying 'mask' key")  # 記錄嘗試使用 'mask' 鍵
            mask = sketch_dict['mask']  # 直接獲取遮罩
            _save_mask_img(mask, "mask")

        # 3. 如果仍未提取到遮罩，回退到 'composite' 鍵 (合成圖)
        if mask is None and 'composite' in sketch_dict:
            log_debug("Trying 'composite' key")  # 記錄嘗試使用 'composite' 鍵
            comp = sketch_dict['composite']  # 獲取合成圖
            
            # 如果合成圖是 RGBA
            if len(comp.shape) == 3 and comp.shape[2] == 4:
                mask = comp[:, :, 3]  # 使用 Alpha 通道
            else:
                # 如果是 RGB 或其他格式
                if len(comp.shape) == 3:
                    gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)  # 轉灰階
                else:
                    gray = comp  # 已經是灰階
                
                # 風險提示：如果背景不是純黑，這裡可能會錯誤地選中背景
                # 進行二值化處理，閾值設為 10
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            log_debug("Mask extracted from composite")  # 記錄從合成圖提取
            _save_mask_img(mask, "composite")
                
    # 如果輸入本身就是 numpy 陣列 (而不是字典)
    elif isinstance(sketch_dict, np.ndarray):
        log_debug("Input is ndarray")  # 記錄輸入為陣列
        comp = sketch_dict  # 直接作為合成圖
        # 同樣的處理邏輯：提取 Alpha 或轉灰階後二值化
        if len(comp.shape) == 3 and comp.shape[2] == 4:
            mask = comp[:, :, 3]
        else:
            if len(comp.shape) == 3:
                gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
            else:
                gray = comp
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        _save_mask_img(mask, "ndarray")
            
    # 如果最終沒有提取到遮罩，返回空列表
    if mask is None:
        log_debug("No mask found in sketch_dict")  # 記錄未找到遮罩
        return []
    
    # 確保遮罩是單通道 (二維)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)  # 轉為灰階

    # 如果提供了原始圖像，調整遮罩大小以匹配原圖
    if original_img is not None:
        h, w = original_img.shape[:2]  # 獲取原圖尺寸
        if mask.shape[:2] != (h, w):
            log_debug(f"Resizing mask from {mask.shape} to {(h, w)}")  # 記錄調整大小
            # 使用最近鄰插值調整大小，保持邊緣清晰
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            _save_mask_img(mask, "resized")

    # 將遮罩二值化，值為 0 或 1，用於骨架化
    _, binary = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
    
    # 2. 骨架化 (Skeletonize) - 獲取單像素寬的路徑
    # 這是為了找到筆觸的中心線
    skeleton = skeletonize(binary).astype(np.uint8)
    
    # 3. 提取骨架上的所有座標點
    ys, xs = np.where(skeleton > 0)  # 獲取非零點的索引
    points = list(zip(xs, ys))  # 組合為 (x, y) 坐標對
    log_debug(f"Skeleton points found: {len(points)}")  # 記錄找到的點數
    
    constraints = []  # 初始化約束列表
    
    # 4. 計算切向量 (使用局部鄰域 PCA 方法)
    for x, y in points:
        # 尋找 5x5 鄰域內的相鄰點
        local_pts = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x+dx, y+dy
                # 檢查邊界條件
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    if skeleton[ny, nx] > 0:  # 如果鄰域點也在骨架上
                        local_pts.append((nx, ny))
        
        # 如果局部點數量足夠 (至少3點才能較好地擬合直線)
        if len(local_pts) >= 3:
            # 使用 PCA (主成分分析) 計算主方向
            pts_array = np.array(local_pts, dtype=np.float32)
            
            # 數據中心化：減去均值
            mean = np.mean(pts_array, axis=0)
            centered = pts_array - mean
            
            # 計算共變異數矩陣
            cov = np.dot(centered.T, centered) / (len(pts_array) - 1)
            
            # 計算特徵值和特徵向量
            # eigh 用於對稱矩陣，返回特徵值按升序排列
            evals, evecs = np.linalg.eigh(cov)
            
            # 對應最大特徵值的特徵向量即為切線方向 (主方向)
            tangent = evecs[:, 1] # 索引 1 是 2D 中較大的那個特徵值對應的向量
            
            ux, uy = tangent[0], tangent[1]  # 提取 x, y 分量
            
            # 添加約束：位置 (x, y) 和方向 (ux, uy)
            constraints.append((x, y, ux, uy))
            
    # 記錄生成的約束摘要
    log_debug(f"Total constraints generated: {len(constraints)}")
    if len(constraints) > 0:
        log_debug(f"First constraint: {constraints[0]}")
                
    # 返回所有計算出的約束，不進行降採樣，以保留最大細節
    return constraints
