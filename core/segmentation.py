import torch  # 導入 PyTorch 深度學習框架
import numpy as np  # 導入 NumPy 用於數值計算
import cv2  # 導入 OpenCV 用於圖像處理
import time  # 導入 time 用於計時
from sam2.build_sam import build_sam2  # 導入 SAM 2 模型構建函數
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # 導入 SAM 2 自動遮罩生成器
from ultralytics import SAM  # 導入 Ultralytics 的 SAM 封裝 (用於 SAM 3)

class SAM3AutoEngine:
    """
    SAM 3 自動分割引擎封裝類
    """
    def __init__(self, model_path, device="cuda"):
        # 設置設備，優先使用 CUDA
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM 3 from {model_path} on {self.device}...")
        
        try:
            # 加載 SAM 模型
            self.model = SAM(model_path)
            print("SAM 3 Loaded Successfully.")
        except Exception as e:
            # 捕獲加載錯誤
            print(f"Error loading SAM 3: {e}")
            self.model = None

        self.current_masks = []  # 存儲當前生成的遮罩列表
        self.image_shape = None  # 存儲當前處理的圖像形狀

    def generate_masks(self, image_rgb):
        """
        使用 SAM 3 為圖像生成所有遮罩。
        
        參數:
            image_rgb: 輸入的 RGB 圖像 (numpy array)
            
        返回:
            masks_list: 過濾後的遮罩字典列表
        """
        # 檢查模型是否加載且圖像是否有效
        if self.model is None or image_rgb is None:
            return []
        
        self.image_shape = image_rgb.shape[:2]  # 記錄圖像尺寸 (H, W)
        
        print("Generating masks with SAM 3...")
        start_time = time.time()  # 開始計時
        
        # 使用 Ultralytics SAM 進行自動分割
        try:
            results = self.model(image_rgb, verbose=False)
        except Exception as e:
            print(f"SAM 3 Inference Error: {e}")
            return []

        # 轉換結果格式
        masks_list = []
        # 檢查是否有結果且有遮罩數據
        if results and results[0].masks is not None:
            masks_data = results[0].masks.data # 獲取遮罩數據 Tensor (N, H, W)
            if masks_data is not None:
                # 將 Tensor 轉為 numpy bool 數組
                masks_np = masks_data.cpu().numpy().astype(bool)
                
                # 遍歷每個遮罩
                for i in range(masks_np.shape[0]):
                    m = masks_np[i]
                    
                    # 確保遮罩尺寸匹配 (Ultralytics 可能返回縮放後的遮罩)
                    if m.shape != self.image_shape:
                        # 如果尺寸不匹配，調整大小
                        m = cv2.resize(m.astype(np.uint8), (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    
                    area = np.sum(m)  # 計算遮罩面積
                    # 過濾過小的遮罩 (小於 100 像素)
                    if area < 100: 
                        continue

                    # 計算遮罩區域的平均顏色以進行過濾
                    masked_pixels = image_rgb[m]  # 提取遮罩區域的像素
                    if masked_pixels.size == 0:
                        continue
                        
                    avg_color = np.mean(masked_pixels, axis=0)  # 計算平均顏色
                    
                    # 判斷是否為純黑色背景 (平均值 < 30)
                    is_black = np.all(avg_color < 30)
                    # 判斷是否為純白色背景 (平均值 > 240)
                    is_white = np.all(avg_color > 240)
                    
                    # 如果不是純黑也不是純白，則保留該遮罩
                    if not is_black and not is_white:
                        ann = {
                            'segmentation': m,  # 布林遮罩
                            'area': area,  # 面積
                            'segmentation_uint8': m.astype(np.uint8) * 255  # uint8 格式遮罩 (0-255)
                        }
                        masks_list.append(ann)
        
        print(f"SAM 3 generated {len(masks_list)} masks.")
        self.current_masks = masks_list  # 更新當前遮罩緩存
        return masks_list

    def get_mask_at_point(self, x, y):
        """
        查找包含指定點 (x, y) 的遮罩。
        
        參數:
            x, y: 點的坐標
            
        返回:
            遮罩在列表中的索引，如果未找到則返回 -1
        """
        if not self.current_masks:
            return -1
            
        candidates = []  # 候選遮罩列表
        for i, ann in enumerate(self.current_masks):
            # 檢查點是否在遮罩內
            if ann['segmentation'][y, x]:
                candidates.append((i, ann['area']))
        
        if not candidates:
            return -1
            
        # 按面積排序，優先選擇面積最小的（通常是最具體的對象）
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]  # 返回最佳候選者的索引

class SAM2AutoEngine:
    """
    SAM 2 自動分割引擎封裝類
    """
    def __init__(self, checkpoint_path, model_cfg, device="cuda"):
        # 設置設備
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM 2 on {self.device}...")
        
        try:
            # 構建 SAM 2 模型
            self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device, apply_postprocessing=False)
            # 初始化自動遮罩生成器
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=64,     # 每邊採樣點數，越高越精細但越慢 (原為 32)
                pred_iou_thresh=0.8,    # 預測 IOU 閾值
                stability_score_thresh=0.9, # 穩定性分數閾值
                crop_n_layers=0,        # 裁剪層數
                min_mask_region_area=100 # 最小遮罩區域面積
            )
            print("SAM 2 Loaded Successfully.")
        except Exception as e:
            print(f"Error loading SAM 2: {e}")
            self.mask_generator = None

        self.current_masks = []
        self.image_shape = None

    def generate_masks(self, image_rgb):
        """
        為圖像生成所有遮罩並進行過濾。
        """
        if self.mask_generator is None or image_rgb is None:
            return []
        
        self.image_shape = image_rgb.shape[:2]
        
        print("Generating masks with SAM 2...")
        start_time = time.time()
        # 調用 SAM 2 生成器
        masks = self.mask_generator.generate(image_rgb)
        print(f"Original masks: {len(masks)}")
        
        # 過濾邏輯 (參考 run_sam2_test.py)
        filtered_masks = []
        for ann in masks:
            m = ann['segmentation']  # 獲取分割掩碼
            
            # 計算平均顏色
            masked_pixels = image_rgb[m]
            if masked_pixels.size == 0:
                continue
                
            avg_color = np.mean(masked_pixels, axis=0)
            
            # 過濾黑色背景 (RGB 平均值 < 30)
            is_black = np.all(avg_color < 30)
            
            # 過濾白色輪廓/背景 (RGB 平均值 > 240)
            is_white = np.all(avg_color > 240)
            
            if not is_black and not is_white:
                # 為了方便後續處理，將遮罩轉換為 uint8 255 格式
                ann['segmentation_uint8'] = m.astype(np.uint8) * 255
                filtered_masks.append(ann)
            else:
                # 調試輸出：顯示過濾掉的顏色
                # color_str = "Black" if is_black else "White"
                # print(f"Filtered {color_str}, avg: {avg_color}")
                pass
                
        print(f"Filtered masks: {len(filtered_masks)}")
        end_time = time.time()
        print(f"Total segmentation time: {end_time - start_time:.2f} seconds")
        self.current_masks = filtered_masks  # 更新緩存
        return filtered_masks

    def get_mask_at_point(self, x, y):
        """
        查找包含點 (x, y) 的遮罩。
        返回 self.current_masks 中遮罩的索引，如果沒有則返回 -1。
        如果多個遮罩重疊，則返回最小的一個（通常是最好的）。
        """
        if not self.current_masks:
            return -1
            
        candidates = []
        for i, ann in enumerate(self.current_masks):
            # segmentation 是布林遮罩，直接索引檢查
            if ann['segmentation'][y, x]:
                candidates.append((i, ann['area']))
        
        if not candidates:
            return -1
            
        # 返回面積最小的一個（最特定的）
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
