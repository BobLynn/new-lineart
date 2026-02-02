import torch
import numpy as np
import cv2
import time
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from ultralytics import SAM

class SAM3AutoEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM 3 from {model_path} on {self.device}...")
        
        try:
            self.model = SAM(model_path)
            print("SAM 3 Loaded Successfully.")
        except Exception as e:
            print(f"Error loading SAM 3: {e}")
            self.model = None

        self.current_masks = []
        self.image_shape = None

    def generate_masks(self, image_rgb):
        """
        使用 SAM 3 為圖像生成所有遮罩。
        """
        if self.model is None or image_rgb is None:
            return []
        
        self.image_shape = image_rgb.shape[:2]
        
        print("Generating masks with SAM 3...")
        start_time = time.time()
        # Ultralytics SAM 自動分割
        try:
            results = self.model(image_rgb, verbose=False)
        except Exception as e:
            print(f"SAM 3 Inference Error: {e}")
            return []

        # 轉換結果格式
        masks_list = []
        if results and results[0].masks is not None:
            masks_data = results[0].masks.data # (N, H, W) Tensor
            if masks_data is not None:
                # 轉為 numpy bool
                masks_np = masks_data.cpu().numpy().astype(bool)
                
                for i in range(masks_np.shape[0]):
                    m = masks_np[i]
                    
                    # 確保尺寸匹配 (Ultralytics 可能返回縮放後的遮罩)
                    if m.shape != self.image_shape:
                        m = cv2.resize(m.astype(np.uint8), (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    
                    area = np.sum(m)
                    # 過濾過小的遮罩
                    if area < 100: 
                        continue

                    # 計算平均顏色過濾 (保留原始邏輯)
                    masked_pixels = image_rgb[m]
                    if masked_pixels.size == 0:
                        continue
                        
                    avg_color = np.mean(masked_pixels, axis=0)
                    is_black = np.all(avg_color < 30)
                    is_white = np.all(avg_color > 240)
                    
                    if not is_black and not is_white:
                        ann = {
                            'segmentation': m,
                            'area': area,
                            'segmentation_uint8': m.astype(np.uint8) * 255
                        }
                        masks_list.append(ann)
        
        print(f"SAM 3 generated {len(masks_list)} masks.")
        self.current_masks = masks_list
        return masks_list

    def get_mask_at_point(self, x, y):
        """
        查找包含點 (x, y) 的遮罩。
        """
        if not self.current_masks:
            return -1
            
        candidates = []
        for i, ann in enumerate(self.current_masks):
            if ann['segmentation'][y, x]:
                candidates.append((i, ann['area']))
        
        if not candidates:
            return -1
            
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

class SAM2AutoEngine:
    def __init__(self, checkpoint_path, model_cfg, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM 2 on {self.device}...")
        
        try:
            self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device, apply_postprocessing=False)
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=64,     #32
                pred_iou_thresh=0.8,
                stability_score_thresh=0.9,
                crop_n_layers=0,
                min_mask_region_area=100
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
        masks = self.mask_generator.generate(image_rgb)
        print(f"Original masks: {len(masks)}")
        
        # 過濾邏輯 (來自 run_sam2_test.py)
        filtered_masks = []
        for ann in masks:
            m = ann['segmentation']
            
            # 計算平均顏色
            masked_pixels = image_rgb[m]
            if masked_pixels.size == 0:
                continue
                
            avg_color = np.mean(masked_pixels, axis=0)
            
            # 過濾黑色背景 (< 30)
            is_black = np.all(avg_color < 30)
            
            # 過濾白色輪廓 (> 240)
            is_white = np.all(avg_color > 240)
            
            if not is_black and not is_white:
                # 為了方便起見，將遮罩轉換為 uint8 255
                ann['segmentation_uint8'] = m.astype(np.uint8) * 255
                filtered_masks.append(ann)
            else:
                # color_str = "Black" if is_black else "White"
                # print(f"Filtered {color_str}, avg: {avg_color}")
                pass
                
        print(f"Filtered masks: {len(filtered_masks)}")
        end_time = time.time()
        print(f"Total segmentation time: {end_time - start_time:.2f} seconds")
        self.current_masks = filtered_masks
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
            # segmentation 是布林遮罩
            if ann['segmentation'][y, x]:
                candidates.append((i, ann['area']))
        
        if not candidates:
            return -1
            
        # 返回面積最小的一個（最特定的）
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]