import gradio as gr  # 導入 Gradio 套件，用於建立互動式網頁介面
import numpy as np   # 導入 NumPy 套件，用於處理陣列與數值運算
import cv2           # 導入 OpenCV 套件，用於影像處理
import matplotlib.pyplot as plt  # 導入 Matplotlib 的 pyplot，通常用於繪圖 (此處可能備用)
from scipy.interpolate import griddata  # 從 SciPy 導入 griddata，用於網格插值運算
from sam2.sam2_image_predictor import SAM2ImagePredictor  # 導入 SAM2 的影像預測器類別
from sam2.build_sam import build_sam2  # 導入建構 SAM2 模型的函式
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # 導入 SAM2 自動遮罩生成器
import sam2  # 導入 SAM2 主套件
import torch  # 導入 PyTorch 套件，用於深度學習模型運算
import os    # 導入 os 套件，用於作業系統相關操作 (如路徑處理)

# ==========================================
# 核心邏輯區 (Backend Logic)
# ==========================================

class NewLineArtGenerator:  # 定義一個名為 NewLineArtGenerator 的類別，作為藝術生成器的核心邏輯控制器
    def __init__(self):  # 類別的初始化函式 (Constructor)
        self.original_image = None       # 初始化原始圖片變數為 None
        self.masks = None                # 初始化遮罩 (Masks) 變數為 None
        self.user_guidance_map = None    # 初始化用戶導向圖 (Guidance Map) 為 None
        self.streamline_result = None    # 初始化流線生成結果為 None

    def _render_masks(self, show_numbers=False):
        """
        輔助函式：繪製遮罩與編號
        """
        if self.original_image is None or self.masks is None:
            return None
            
        # 準備視覺化分割結果的畫布，大小與原圖相同，初始為全黑
        vis_img = np.zeros_like(self.original_image)
        
        # 遍歷每一個生成的遮罩
        for ann in self.masks:
            m = ann['segmentation']  # 取得遮罩的二值化陣列 (True/False)
            # 設定 RGB 顏色為黃色 (255, 255, 0)
            color = np.array([255, 255, 0], dtype=np.uint8)
            # 將遮罩區域 (True 的部分) 填入該顏色
            vis_img[m] = color
            
        if show_numbers:
            # 第二次遍歷：只繪製數字，確保數字在所有遮罩的上層
            for i, ann in enumerate(self.masks):
                m = ann['segmentation']
                # 計算質心以標示數字
                ys, xs = np.where(m)
                if len(xs) > 0 and len(ys) > 0:
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    # 繪製數字 (紅色，字體大小 1.0，粗細 2)
                    cv2.putText(vis_img, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.0, (255, 0, 0), 2, cv2.LINE_AA)
                                
        return vis_img

    def update_view(self, show_numbers):
        """
        更新顯示 (切換是否顯示編號)
        """
        if self.masks is None:
            return None
        return self._render_masks(show_numbers)

    def segment_image_sam2(self, image, show_numbers=False):  # 定義使用 SAM2 進行影像分割的函式
        """
        Tab 1: 使用 SAM2 進行分割
        使用 Meta SAM2 模型進行圖像自動分割 (Automatic Mask Generation)。
        """
        if image is None:  # 檢查輸入的圖片是否為空
            return None, "請先上傳圖片！"  # 如果圖片為空，回傳 None 並顯示提示訊息
        
        self.original_image = image  # 將輸入的圖片儲存到類別屬性 original_image 中
        
        # [SAM2 真實模式] 使用 SAM2 自動分割
        
        # 1. 檢查並加載模型 (Lazy Loading - 延遲加載策略)
        # 檢查是否已經建立了 mask_generator，如果沒有則進行初始化
        if not hasattr(self, 'mask_generator') or self.mask_generator is None:
            try:  # 嘗試執行模型加載程式碼
                # 設定模型權重檔 (.pt) 的路徑，使用 os.path.join 組合相對路徑
                checkpoint_path = os.path.join("checkpoints", "sam2_hiera_large.pt")
                # 設定模型設定檔 (.yaml) 的路徑，動態尋找 sam2 套件安裝位置下的設定檔
                model_cfg = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2", "sam2_hiera_l.yaml")
                # 檢查是否有可用的 CUDA 裝置 (GPU)，否則使用 CPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # 印出正在加載模型的訊息，包含路徑與使用的裝置
                print(f"Loading SAM2 from {checkpoint_path} on {device}...")
                # 使用 build_sam2 函式建構 SAM2 模型實例
                sam2_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
                
                # 初始化 SAM2 自動遮罩生成器 (Automatic Mask Generator)
                self.mask_generator = SAM2AutomaticMaskGenerator(
                    model=sam2_model,             # 傳入已加載的 SAM2 模型
                    points_per_side=32,           # 每邊取樣點數 (控制分割細粒度)
                    pred_iou_thresh=0.86,         # 預測 IOU 閾值 (過濾低品質遮罩)
                    stability_score_thresh=0.92,  # 穩定性分數閾值 (確保遮罩穩定)
                    crop_n_layers=0,              # 裁切層數 (0 表示不進行額外裁切)
                    min_mask_region_area=100      # 最小遮罩區域面積 (過濾過小的雜訊)
                )
                # 印出 SAM2 加載成功的訊息
                print("Auto Segmentation -- Step1: SAM2 Loaded Successfully!")
            except Exception as e:  # 捕捉加載過程中的任何錯誤
                # 回傳 None 並顯示詳細的錯誤訊息，提示用戶檢查路徑或環境
                return None, f"SAM2 模型加載失敗: {e}\n請確認 checkpoints 路徑與環境安裝。"

        # 2. 執行分割
        print("Auto Segmentation -- Step2: Generating masks with SAM2...")  # 印出開始生成遮罩的訊息
        try:  # 嘗試執行遮罩生成
            masks = self.mask_generator.generate(image)  # 呼叫 generate 方法對圖片進行分割
        except Exception as e:  # 捕捉分割過程中的錯誤
            return None, f"SAM2 分割執行錯誤: {e}"  # 回傳錯誤訊息

        # 過濾掉白色和黑色區域的遮罩
        filtered_masks = []
        # 確保 image 是 RGB (只取前三個通道)
        img_rgb = image[:, :, :3] if image.ndim == 3 and image.shape[2] >= 3 else image
        
        for ann in masks:
            m = ann['segmentation']
            
            # 找出圖像中是白色的像素 (R,G,B 均 > 240)
            if img_rgb.ndim == 3:
                is_white = np.all(img_rgb > 240, axis=-1)
                is_black = np.all(img_rgb < 15, axis=-1)
            else:
                is_white = img_rgb > 240
                is_black = img_rgb < 15
                
            # 將遮罩中的白色和黑色區域移除
            m[is_white] = False
            m[is_black] = False
            
            ann['segmentation'] = m
            
            # 如果遮罩還有剩餘區域，則保留
            if np.any(m):
                filtered_masks.append(ann)
        
        masks = filtered_masks

        # 3. 處理結果與視覺化
        if len(masks) == 0:  # 檢查生成的遮罩數量是否為 0
            return image, "SAM2 未檢測到任何區域。"  # 如果沒有檢測到區域，回傳原圖與提示訊息
            
        # 按面積大小對遮罩進行排序 (從大到小)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        self.masks = masks # 將排序後的遮罩列表儲存到類別屬性 self.masks
        
        # 使用新函式繪製
        vis_img = self._render_masks(show_numbers)
            
        # 回傳視覺化結果圖片與完成訊息 (包含檢測到的區域數量)
        return vis_img, f"SAM2 分割完成！共檢測到 {len(masks)} 個區域。"

    def save_guidance(self, drawing_dict):  # 定義儲存用戶繪製導向線的函式
        """
        Tab 2: 處理用戶畫的參考線
        Gradio 的 Sketch 工具會回傳一個字典 {'background': ..., 'layers': ...}
        """
        if drawing_dict is None:  # 檢查輸入的繪圖字典是否為空
            return "請繪製參考線！"  # 如果為空，回傳提示訊息

        # 取得用戶畫的筆畫圖層 (通常是透明背景的 PNG)
        # Gradio 新版 Image Editor 回傳的是 'composite' 或 'layers'
        # 我們假設 drawing_dict 是一個包含 'layers' 的字典
        
        # 提取畫筆圖層 (如果是 ImageEditor，通常在 layers 列表的第一個元素)
        canvas = drawing_dict["layers"][0]  # 取第一層塗鴉層
        print ("cavas shape!", canvas.shape)
        
        # 將圖層轉為灰階圖像，有畫筆跡的地方會有非零數值
        gray_guide = cv2.cvtColor(canvas, cv2.COLOR_RGBA2GRAY)
        # 使用二值化處理，將灰階圖轉為純黑白圖 (閾值設為 10)，生成導向圖
        _, self.user_guidance_map = cv2.threshold(gray_guide, 10, 255, cv2.THRESH_BINARY)
        
        # 回傳儲存成功的訊息
        return "參考線已儲存！請前往下一步。"

    def generate_streamlines(self, stroke_length=20, density=2):  # 定義生成流線的函式
        """
        Tab 3: 使用 Structure Tensor 生成流線
        """
        # 確保參數為整數，避免 range() 報錯
        stroke_length = int(stroke_length)
        density = int(density)

        if self.user_guidance_map is None:  # 檢查是否已經有導向圖
            return None, "找不到參考線，請回到 Tab 2 繪製。"  # 如果沒有，提示用戶回去繪製

        # 取得導向圖的高度 (h) 和寬度 (w)
        h, w = self.user_guidance_map.shape
        
        # 1. 計算結構張量 (Structure Tensor) 或 簡單梯度場
        # 為了簡化，我們直接計算用戶畫線的梯度方向
        # 對畫線圖做高斯模糊，讓梯度場擴散開來，使線條影響範圍變大
        blurred_guide = cv2.GaussianBlur(self.user_guidance_map, (51, 51), 0)
        
        # 使用 Sobel 算子計算 X 方向的梯度 (Gx)
        gx = cv2.Sobel(blurred_guide, cv2.CV_64F, 1, 0, ksize=5)
        # 使用 Sobel 算子計算 Y 方向的梯度 (Gy)
        gy = cv2.Sobel(blurred_guide, cv2.CV_64F, 0, 1, ksize=5)
        
        # 將直角坐標系的梯度 (Gx, Gy) 轉換為極座標系的 (強度, 角度)
        magnitude, angle = cv2.cartToPolar(gx, gy)
        
        # 將角度轉為向量場 (Vector Field)
        # 注意：流線應該順著畫線方向。梯度方向通常垂直於線條邊緣。
        # 如果畫的是線本身，梯度垂直於線，所以我們要旋轉 90 度 (pi/2) 來取切線方向。
        angle = angle + np.pi / 2 

        # 計算向量場的 X 分量 (vx)
        vx = np.cos(angle)
        # 計算向量場的 Y 分量 (vy)
        vy = np.sin(angle)

        # 處理 Mask 限制
        combined_mask = None
        if self.masks is not None and len(self.masks) > 0:
            combined_mask = np.zeros((h, w), dtype=bool)
            for ann in self.masks:
                combined_mask = np.logical_or(combined_mask, ann['segmentation'])
            
            # 將 Mask 以外的區域向量場設為 0
            vx[~combined_mask] = 0
            vy[~combined_mask] = 0

        # 2. 繪製流線 (Streamlines)
        # 建立一個全黑的 RGB 畫布
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 在畫布上隨機撒點，作為流線生成的起始種子點
        seed_points = []
        
        if combined_mask is not None:
            # 找出 mask 內的有效座標
            ys, xs = np.where(combined_mask)
            if len(xs) > 0:
                # 根據密度參數計算撒點數量 (基於有效面積)
                num_seeds = int(len(xs) / (100 // density))
                if num_seeds > 0:
                    # 隨機選取
                    indices = np.random.choice(len(xs), num_seeds, replace=True)
                    for idx in indices:
                        seed_points.append((xs[idx], ys[idx]))
        else:
            # 舊邏輯：全圖隨機撒點
            # 根據密度參數計算撒點數量，密度越高點越多
            for _ in range(int(w * h / (100 // density))): # 密度控制公式
                r = np.random.randint(0, h)  # 隨機產生 Y 座標
                c = np.random.randint(0, w)  # 隨機產生 X 座標
                seed_points.append((c, r))   # 將座標加入種子點列表

        # 使用簡單的積分法繪製流線
        for x, y in seed_points:  # 遍歷每個種子點
            points = [(x, y)]     # 記錄流線路徑的點列表，初始為種子點
            curr_x, curr_y = x, y # 設定當前追蹤座標
            
            for _ in range(stroke_length):  # 根據設定的流線長度進行迭代
                ix, iy = int(curr_x), int(curr_y)  # 將座標轉為整數索引
                # 檢查是否超出邊界，若超出則停止該條流線
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    break
                
                # 如果有 mask，且跑出 mask 範圍，也停止
                if combined_mask is not None and not combined_mask[iy, ix]:
                    break

                # 從向量場獲取該點的速度向量，並乘以 2 作為步長
                dx = vx[iy, ix] * 2 
                dy = vy[iy, ix] * 2
                
                # 更新當前座標
                curr_x += dx
                curr_y += dy
                # 將新座標加入路徑點列表
                points.append((int(curr_x), int(curr_y)))

            # 畫線
            if len(points) > 1:  # 如果路徑點超過 1 個才畫線
                # 隨機產生稍微不同的白色/灰色亮度值 (200-255)，增加藝術感
                color_val = np.random.randint(200, 255)
                # 使用 OpenCV 的 polylines 函式將點連接成線
                cv2.polylines(canvas, [np.array(points)], False, (color_val, color_val, color_val), 1, cv2.LINE_AA)

        # 將生成的流線畫布儲存到類別屬性
        self.streamline_result = canvas
        # 回傳流線畫布與完成訊息
        return canvas, "流線生成完畢！"

    def final_composite(self, contour_img):  # 定義最終合成函式
        """
        Tab 4: 疊加輪廓與流線
        """
        if self.streamline_result is None:  # 檢查是否已生成流線
            return None, "請先生成流線！"  # 若無流線，提示用戶
        
        if contour_img is None:  # 檢查是否上傳了輪廓圖
            return self.streamline_result, "未上傳輪廓圖，僅顯示流線。"  # 若無輪廓圖，直接回傳流線圖
        
        # 取得流線圖的高度和寬度
        h, w, _ = self.streamline_result.shape
        # 將輪廓圖調整大小以符合流線圖的尺寸
        contour_resized = cv2.resize(contour_img, (w, h))
        
        # 假設輪廓是黑底白線，或是白底黑線，這裡做個處理確保是白線疊加
        # 使用 addWeighted 進行簡單的線性疊加 (Add) 效果
        # 參數: src1, alpha, src2, beta, gamma -> src1*alpha + src2*beta + gamma
        final_img = cv2.addWeighted(self.streamline_result, 1.0, contour_resized, 0.8, 0)
        
        # 回傳最終合成圖
        return final_img

# 初始化處理器實例
processor = NewLineArtGenerator()

# ==========================================
# Gradio 介面構建 (Frontend UI)
# ==========================================

# 建立 Gradio Blocks 介面，設定標題與主題
custom_css = """
#adaptive_drawer {
    height: 80vh !important;
}
"""
with gr.Blocks(title="AI 線條藝術生成器", css=custom_css) as demo:
    gr.Markdown("# 🦉 AI 線條藝術生成器 (Line Art Generator)")  # 顯示主標題
    gr.Markdown("這是一個協助你不具備藝術背景也能畫出精密流線藝術的工具。請依照 Tab 順序操作。")  # 顯示說明文字

    with gr.Tabs():  # 建立分頁標籤 (Tabs)
        # --- Tab 1: 分割區塊 ---
        with gr.TabItem("Step 1: 區域分割 (SAM2)"):  # 第一個分頁：SAM2 分割
            with gr.Row():  # 建立水平排列的 Row
                with gr.Column():  # 左側欄位
                    input_img_1 = gr.Image(label="上傳色塊標記圖", type="numpy")  # 圖片上傳元件
                    show_nums_chk = gr.Checkbox(label="顯示區域編號", value=False) # 新增 Checkbox
                    seg_btn = gr.Button("執行 SAM2 分割", variant="primary")  # 分割執行按鈕
                with gr.Column():  # 右側欄位
                    seg_output = gr.Image(label="分割結果確認")  # 顯示分割結果的圖片元件
                    seg_msg = gr.Textbox(label="狀態", interactive=False)  # 顯示狀態訊息的文字框
            
            confirm_btn_1 = gr.Button("確認分割正確，下一步 👉")  # 確認並前往下一步的按鈕

        # --- Tab 2: 繪製導向 ---
        with gr.TabItem("Step 2: 繪製流線方向"):  # 第二個分頁：繪製流向
            gr.Markdown("請使用畫筆在不同區塊畫出你想要的線條流動方向。")  # 說明文字
            with gr.Row():  # 建立水平排列
                # 使用 ImageEditor 讓用戶可以繪畫
                # 注意：將 Tab 1 的分割結果傳過來當底圖
                draw_input = gr.ImageEditor(  # 圖片編輯器元件
                    label="繪製導向線 (畫筆)", 
                    type="numpy", 
                    elem_id="adaptive_drawer",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=5),  # 設定畫筆預設為白色，大小 5
                    eraser=gr.Eraser(),  # 啟用橡皮擦功能
                    interactive=True     # 設定為可互動
                )
            
            with gr.Row():  # 按鈕列
                clear_draw_btn = gr.Button("清除所有筆畫 (重置)")  # 清除按鈕
                save_draw_btn = gr.Button("儲存筆畫方向", variant="primary")  # 儲存按鈕
            
            draw_msg = gr.Textbox(label="狀態", interactive=False)  # 狀態訊息框
            confirm_btn_2 = gr.Button("確認繪製完成，下一步 👉")  # 確認按鈕

        # --- Tab 3: 生成流線 ---
        with gr.TabItem("Step 3: 生成藝術流線"):  # 第三個分頁：生成流線
            with gr.Row():
                with gr.Column():
                    param_len = gr.Slider(5, 50, value=20, label="流線長度")  # 滑桿：調整流線長度
                    param_den = gr.Slider(1, 5, value=2, label="流線密度")   # 滑桿：調整流線密度
                    gen_btn = gr.Button("使用 Structure Tensor 生成", variant="primary")  # 生成按鈕
                with gr.Column():
                    streamline_output = gr.Image(label="流線預覽")  # 顯示流線結果
                    gen_msg = gr.Textbox(label="狀態")  # 狀態訊息
            
            confirm_btn_3 = gr.Button("滿意結果，下一步 👉")  # 確認按鈕

        # --- Tab 4: 最終合成 ---
        with gr.TabItem("Step 4: 輪廓疊加"):  # 第四個分頁：輪廓疊加
            with gr.Row():
                with gr.Column():
                    contour_input = gr.Image(label="上傳純輪廓線圖", type="numpy")  # 上傳輪廓圖
                    merge_btn = gr.Button("合成最終作品", variant="primary")  # 合成按鈕
                with gr.Column():
                    final_output = gr.Image(label="最終成品")  # 顯示最終成品
    
    # ==========================================
    # 事件串接 (Event Handling)
    # ==========================================
    
    # Tab 1 Events (事件綁定)
    # 當按下 seg_btn 時，呼叫 processor.segment_image_sam2 函式
    seg_btn.click(fn=processor.segment_image_sam2, inputs=[input_img_1, show_nums_chk], outputs=[seg_output, seg_msg])
    
    # Checkbox 切換時更新顯示
    show_nums_chk.change(fn=processor.update_view, inputs=[show_nums_chk], outputs=[seg_output])
    
    # 當按下 Tab 1 的確認，自動把圖片轉發到 Tab 2 的編輯器中
    def send_to_editor(img):  # 定義轉發圖片的輔助函式
        return img  # 直接回傳圖片
    
    # 綁定確認按鈕事件：輸入為 Tab 1 的分割結果圖，輸出到 Tab 2 的編輯器
    confirm_btn_1.click(fn=send_to_editor, inputs=[seg_output], outputs=[draw_input])

    # Tab 2 Events
    # ImageEditor 輸出的是 dictionary，綁定儲存按鈕
    save_draw_btn.click(fn=processor.save_guidance, inputs=[draw_input], outputs=[draw_msg])
    
    # Tab 3 Events
    # 綁定生成按鈕，輸入包含滑桿參數
    gen_btn.click(fn=processor.generate_streamlines, inputs=[param_len, param_den], outputs=[streamline_output, gen_msg])

    # Tab 4 Events
    # 綁定合成按鈕
    merge_btn.click(fn=processor.final_composite, inputs=[contour_input], outputs=[final_output])

# 啟動應用程式
if __name__ == "__main__":  # 程式進入點
    demo.launch(theme=gr.themes.Soft())  # 啟動 Gradio 伺服器，並套用 Soft 主題
