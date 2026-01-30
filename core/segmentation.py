import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2AutoEngine:
    def __init__(self, checkpoint_path, model_cfg, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM 2 on {self.device}...")
        
        try:
            self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device, apply_postprocessing=False)
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=32,
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