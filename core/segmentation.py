import torch
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

class SAM3Engine:
    def __init__(self, checkpoint_path, model_type="sam3_hiera_large", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM 3 ({model_type}) on {self.device}...")
        
        # 使用 Ultralytics 的 SAM3 實現，繞過 triton 依賴
        # checkpoint_path 作為 model 參數傳入
        self.predictor = SAM3SemanticPredictor(overrides={'model': checkpoint_path})
        self.current_image = None
        self.current_image_set = False

    def set_image(self, image_rgb):
        """
        Encoding 圖像。
        """
        if image_rgb is None: return
        self.current_image = image_rgb
        # Ultralytics 的 set_image 需要傳入圖像
        try:
            self.predictor.set_image(image_rgb)
            self.current_image_set = True
        except Exception as e:
            print(f"Warning: set_image failed (possibly due to missing model file): {e}")
            self.current_image_set = False

    def predict_click(self, point_coords, point_labels):
        """
        點擊互動模式 (Point Prompts)
        """
        if not self.current_image_set: return None
        
        try:
            # Ultralytics SAM3 的 prompt_inference API 似乎有問題或簽名不匹配
            # 我們改用標準的 __call__ 接口 (即 self.predictor(source=...))
            # 注意: Ultralytics 的 points 格式通常是 [[x, y]]
            
            # 獲取圖像尺寸以構建全圖 Bounding Box
            # 這是一個 Workaround，因為 Ultralytics SAM3 實現在只有點提示時似乎會崩潰 (torch.cat error)
            h, w = self.current_image.shape[:2]
            dummy_box = [0, 0, w, h]

            results = self.predictor(
                source=self.current_image,
                points=point_coords,
                labels=point_labels,
                bboxes=[dummy_box], # 傳入全圖 Box 以避免 crash
                save=False,
                verbose=False
            )
            
            # 解析結果
            result = results[0] if isinstance(results, list) else results
            
            if result.masks is None:
                return None
                
            # masks.data is (N, H, W) tensor
            masks = result.masks.data.cpu().numpy()
            
            # 返回最高分的 mask (通常是第一個)
            return masks[0].astype(np.uint8) * 255
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None

    def predict_text(self, text_prompt):
        """
        SAM 3 獨有的文字提示模式 (Promptable Concept Segmentation)
        """
        if not self.current_image_set: return None
        
        try:
            # 使用 __call__ 進行文字提示推理
            results = self.predictor(
                source=self.current_image,
                text=[text_prompt],
                save=False,
                verbose=False
            )
            
            result = results[0] if isinstance(results, list) else results
            
            if result.masks is None:
                return None
                
            masks = result.masks.data.cpu().numpy()
            return masks[0].astype(np.uint8) * 255
        except Exception as e:
            print(f"Text prediction failed: {e}")
            return None