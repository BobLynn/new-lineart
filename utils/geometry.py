# utils/geometry.py
import cv2
import numpy as np
from skimage.morphology import skeletonize

def parse_gradio_sketch(original_img, sketch_dict):
    """
    gradio sketch 返回一个 dict: {'image': 原图, 'mask': 只有画笔痕迹的黑白图}
    """
    if isinstance(sketch_dict, dict) and 'mask' in sketch_dict:
        mask = sketch_dict['mask']
    elif isinstance(sketch_dict, dict) and 'composite' in sketch_dict:
        comp = sketch_dict['composite']
        if len(comp.shape) == 3 and comp.shape[2] == 4:
            mask = comp[:, :, 3]
        else:
            if len(comp.shape) == 3:
                gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
            else:
                gray = comp
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    elif isinstance(sketch_dict, np.ndarray):
        comp = sketch_dict
        if len(comp.shape) == 3 and comp.shape[2] == 4:
            mask = comp[:, :, 3]
        else:
            if len(comp.shape) == 3:
                gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY)
            else:
                gray = comp
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    else:
        return []
    
    # 1. 轉灰度並二值化
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
    
    # 2. 骨架化 (Skeletonize) - 獲取單像素寬的路徑
    skeleton = skeletonize(binary).astype(np.uint8)
    
    # 3. 提取坐標點
    ys, xs = np.where(skeleton > 0)
    points = list(zip(xs, ys))
    
    constraints = []
    
    # 4. 計算切向量 (簡單差分法)
    # 實際專案建議用樣條插值 (Spline) 這裡先用簡單的鄰域查找
    for x, y in points:
        # 尋找 5x5 鄰域內的相鄰點來估算方向
        local_pts = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x+dx, y+dy
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                    if skeleton[ny, nx] > 0 and (dx!=0 or dy!=0):
                        local_pts.append((nx, ny))
        
        if len(local_pts) > 0:
            # 使用 PCA 或簡單平均向量計算局部方向
            # 這裡簡化：取鄰域點的重心方向
            center_x = np.mean([p[0] for p in local_pts])
            center_y = np.mean([p[1] for p in local_pts])
            
            # 切向量 (注意：結構張量不分正負，只看軸向)
            ux = x - center_x
            uy = y - center_y
            
            # 如果太接近中心（雜訊），則忽略
            if abs(ux) > 0.1 or abs(uy) > 0.1:
                constraints.append((x, y, ux, uy))
                
    # 降採樣 constraints (不必每個像素都當約束，太慢)
    return constraints[::5]
