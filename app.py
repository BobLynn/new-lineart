import gradio as gr
import numpy as np
import cv2
import os
import sys
import pickle
import json
import subprocess
import sam2
from core.segmentation import SAM2AutoEngine
from core.tensor_solver import TensorFieldGenerator
from core.renderer import StreamlineRenderer
from utils.geometry import parse_gradio_sketch

# --- 初始化 SAM 2 ---
# Checkpoint 路徑 (來自 00_testing_field)
CHECKPOINT_PATH = os.path.join("00_testing_field", "sam2_hiera_large.pt")
# 模型配置 (指向環境中的 sam2 設定檔)
MODEL_CFG = os.path.join(os.path.dirname(sam2.__file__), "configs", "sam2", "sam2_hiera_l.yaml")

sam_engine = SAM2AutoEngine(checkpoint_path=CHECKPOINT_PATH, model_cfg=MODEL_CFG)

class SessionState:
    def __init__(self):
        self.raw_image = None       # 原始圖片
        self.active_mask = None     # 當前選中的區域 (二值遮罩) 合併後
        self.tensor_field = None    # 緩存的張量場
        self.cached_lines = None    # 緩存的線條 (已平滑)
        self.last_density = None    # 上一次渲染的密度
        
        # SAM 2 相關
        self.sam2_masks = []          # 自動生成器過濾後的遮罩列表
        self.selected_indices = set() # 當前選中遮罩的索引

def combine_masks(masks, selected_indices, shape):
    """將選中的遮罩合併為一個二值遮罩"""
    final_mask = np.zeros(shape, dtype=np.uint8)
    if not masks or not selected_indices:
        return final_mask
        
    for idx in selected_indices:
        if idx < len(masks):
            # masks[idx]['segmentation'] 是布林值
            m = masks[idx]['segmentation']
            final_mask[m] = 255
            
    return final_mask

def draw_sam2_overlay(image, masks, selected_indices):
    """
    繪製所有遮罩。
    選中的遮罩 = 亮色（隨機但一致）。
    未選中的遮罩 = 暗淡輪廓或非常淡的顏色。
    """
    if image is None: return None
    overlay = image.copy()
    
    # 1. 繪製未選中的遮罩 (暗淡)
    # 2. 繪製選中的遮罩 (明亮)
    
    # 預生成顏色以保持一致性？
    # 我們可以對索引進行雜湊以獲取顏色
    np.random.seed(42)
    colors = [np.random.randint(0, 255, 3).tolist() for _ in range(len(masks))]
    
    # 創建一個用於遮罩混合的畫布
    mask_layer = np.zeros_like(overlay)
    
    for i, ann in enumerate(masks):
        m = ann['segmentation']
        color = colors[i]
        
        if i in selected_indices:
            # 選中：明亮的 Alpha 混合
            mask_layer[m] = color
        else:
            # 未選中：淡色或輪廓？
            # 讓我們做非常淡的填充
            # 加深顏色
            dim_color = [c // 4 for c in color]
            mask_layer[m] = dim_color
            
    # 合成
    # 遮罩區域 (mask_layer > 0)
    mask_bool = np.any(mask_layer > 0, axis=2)
    if np.any(mask_bool):
        overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.5, mask_layer[mask_bool], 0.5, 0)
        
    # 繪製輪廓以提高可見度
    for i, ann in enumerate(masks):
        m_uint8 = ann['segmentation_uint8']
        contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if i in selected_indices:
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2) # 白色粗邊框
        else:
            cv2.drawContours(overlay, contours, -1, (128, 128, 128), 1) # 灰色細邊框

    return overlay

def on_upload(image, state):
    state = SessionState()
    state.raw_image = image
    
    if image is not None:
        # 執行 SAM 2 自動分割
        state.sam2_masks = sam_engine.generate_masks(image)
        
        # 默認：選中所有過濾後的遮罩
        # 過濾邏輯應該只保留對象，所以全選是一個很好的默認值。
        state.selected_indices = set(range(len(state.sam2_masks)))
        
        # 合併
        state.active_mask = combine_masks(state.sam2_masks, state.selected_indices, image.shape[:2])
        
        # 繪製疊加層
        overlay = draw_sam2_overlay(image, state.sam2_masks, state.selected_indices)
        return image, overlay, state
    
    return image, None, state

def clear_field_cache(state):
    """當用戶修改繪圖時，清除緩存的張量場"""
    state.tensor_field = None
    state.cached_lines = None
    state.last_density = None
    return state

def on_click(state, evt: gr.SelectData):
    """處理滑鼠點擊 -> 切換遮罩選擇"""
    if state.raw_image is None or not state.sam2_masks: return None, state
    
    # 找到被點擊的遮罩
    idx = sam_engine.get_mask_at_point(evt.index[0], evt.index[1])
    
    if idx != -1:
        if idx in state.selected_indices:
            state.selected_indices.remove(idx)
        else:
            state.selected_indices.add(idx)
            
        # 重新合併
        state.active_mask = combine_masks(state.sam2_masks, state.selected_indices, state.raw_image.shape[:2])
        
        # 遮罩改變，清除緩存
        state.tensor_field = None
        state.cached_lines = None
        state.last_density = None
        
        # 視覺化
        overlay = draw_sam2_overlay(state.raw_image, state.sam2_masks, state.selected_indices)
            
        return overlay, state
    
    return None, state

def prepare_drawing_canvas(state):
    """
    確認分割後，進入 Step 2。
    將非 Mask 區域變暗，讓用戶專注於在 Mask 內畫筆觸。
    """
    if state.active_mask is None: return None
    
    # 創建一個 "變暗" 背景
    canvas_bg = state.raw_image.copy()
    # 將背景壓暗 70%
    bg_mask = state.active_mask == 0
    canvas_bg[bg_mask] = (canvas_bg[bg_mask] * 0.3).astype(np.uint8)
    
    return canvas_bg

def update_preview(drawing_dict, density, width, sharpness, state):
    """
    預覽生成：檢查緩存，如果沒有則計算張量場，然後渲染。
    """
    if state.raw_image is None or state.active_mask is None:
        return None, state
        
    # 1. 檢查/計算張量場
    if state.tensor_field is None:
        print("正在為預覽計算張量場...")
        stroke_constraints = parse_gradio_sketch(state.raw_image, drawing_dict)
        h, w = state.raw_image.shape[:2]
        solver = TensorFieldGenerator(h, w)
        state.tensor_field = solver.solve_field_with_mask(stroke_constraints, state.active_mask)
    
    # 2. 渲染
    h, w = state.raw_image.shape[:2]
    renderer = StreamlineRenderer(state.tensor_field, h, w)
    
    # 檢查緩存
    need_new_lines = True
    if state.cached_lines is not None and state.last_density is not None:
        if abs(state.last_density - density) < 0.1:
            need_new_lines = False
            
    if need_new_lines:
        print(f"正在生成新的流線 (密度: {density})...")
        # 確保渲染器使用正確的遮罩
        renderer.mask = state.active_mask
        
        auto_min_len = int(max(15, density * 1.5))
        raw_lines = renderer.generate_streamlines(density, min_len=auto_min_len, show_progress=False)
        
        smoothed_lines = []
        for l in raw_lines:
            smoothed_lines.append(renderer.smooth_line(l))
            
        state.cached_lines = smoothed_lines
        state.last_density = density
    else:
        print("使用緩存的流線進行預覽...")

    # 預覽生成
    preview_image = renderer.render_from_lines(
        state.cached_lines,
        line_width=width,
        taper_sharpness=sharpness
    )
    return preview_image, state

def launch_interactive_tuner(state):
    """
    啟動 Pygame 互動視窗
    """
    if state.tensor_field is None:
        return "請先生成預覽（或等待自動生成）。"
    
    # 保存數據
    data = {
        'tensor_field': state.tensor_field,
        'mask': state.active_mask
    }
    
    pkl_path = os.path.abspath("temp_preview.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
        
    # 啟動子進程
    script_path = os.path.join("core", "interactive_window.py")
    if not os.path.exists(script_path):
        return f"錯誤: 找不到 {script_path}。"
        
    # 運行 python
    cmd = [sys.executable, script_path, pkl_path]
    subprocess.Popen(cmd)
    
    return "互動調節器已啟動！請檢查您的工作列。"

def load_tuner_params(state):
    """
    從 tuner_params.json 讀取參數並返回給滑桿
    """
    json_path = os.path.abspath("tuner_params.json")
    if not os.path.exists(json_path):
        # 如果文件未找到，返回當前值或默認值
        # Gradio 更新: (density, width, sharpness, status)
        return gr.update(), gr.update(), gr.update(), "錯誤: 未找到保存的參數。請先在調節器中點擊 '保存到 Web UI'。"
    
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        d = params.get("density", 20)
        w = params.get("width", 2)
        s = params.get("sharpness", 0.5)
        
        return d, w, s, f"已加載參數: D={d}, W={w}, S={s}"
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), f"加載參數時出錯: {e}"

def run_hypnotic_gen(drawing_dict, density, width, sharpness, state):
    """
    最終生成
    """
    img, state = update_preview(drawing_dict, density, width, sharpness, state)
    return img, state

# --- 佈局 ---
css = """
#drawing-board {
    height: 80vh !important;
    max-height: 80vh !important;
}
.gradio-container {
    max_width: 100% !important;
}
"""
with gr.Blocks(title="SAM 2 催眠藝術") as demo:
    state = gr.State(SessionState())
    
    with gr.Tab("第一步: 選擇對象 (SAM 2)"):
        gr.Markdown("上傳圖片。SAM 2 將自動對其進行分割。點擊區域以選擇/取消選擇。")
        with gr.Row():
            input_img = gr.Image(label="上傳圖片", type="numpy")
            seg_preview = gr.Image(label="分割預覽", interactive=False)
        
        with gr.Row():
            confirm_btn = gr.Button("確認區域並下一步", variant="primary")

    with gr.Tab("第二步: 繪製流向"):
        gr.Markdown("畫紅線以引導流向。")
        drawing_board = gr.ImageEditor(label="繪製筆觸", type="numpy", elem_id="drawing-board")
        
    with gr.Tab("第三步: 預覽與調整"):
        with gr.Row():
            density_slider = gr.Slider(5, 50, value=20, label="間距密度")
            width_slider = gr.Slider(1, 10, value=2, label="基礎寬度")
            sharp_slider = gr.Slider(0, 1, value=0.5, label="漸變清晰度")
        
        btn_refresh_preview = gr.Button("刷新預覽", variant="secondary")
        btn_interactive = gr.Button("啟動互動調節器 (Pygame)", variant="primary")
        btn_sync = gr.Button("從調節器同步參數", variant="secondary") # 新按鈕
        status_msg = gr.Textbox(label="狀態", interactive=False)
        preview_view = gr.Image(label="線稿預覽", interactive=False)
        
    with gr.Tab("第四步: 結果"):
        gen_btn = gr.Button("開始催眠！", variant="stop")
        result_view = gr.Image()
        
    # 事件
    input_img.upload(on_upload, [input_img, state], [input_img, seg_preview, state])
    input_img.select(on_click, [state], [seg_preview, state])
    
    confirm_btn.click(prepare_drawing_canvas, inputs=[state], outputs=[drawing_board])
    
    # 當繪圖改變時，清除緩存
    drawing_board.change(clear_field_cache, inputs=[state], outputs=[state])
    
    # 預覽標籤頁事件
    slider_inputs = [drawing_board, density_slider, width_slider, sharp_slider, state]
    
    # 滑桿實時更新
    density_slider.change(update_preview, inputs=slider_inputs, outputs=[preview_view, state])
    width_slider.change(update_preview, inputs=slider_inputs, outputs=[preview_view, state])
    sharp_slider.change(update_preview, inputs=slider_inputs, outputs=[preview_view, state])
    
    # 手動刷新按鈕 (Force Re-compute)
    btn_refresh_preview.click(clear_field_cache, inputs=[state], outputs=[state]).then(
        update_preview, inputs=slider_inputs, outputs=[preview_view, state]
    )
    
    # 啟動 Pygame
    btn_interactive.click(launch_interactive_tuner, inputs=[state], outputs=[status_msg])
    
    # 同步參數
    btn_sync.click(load_tuner_params, inputs=[state], outputs=[density_slider, width_slider, sharp_slider, status_msg])

    # 最終生成
    gen_btn.click(run_hypnotic_gen,  
                 inputs=slider_inputs,
                 outputs=[result_view, state])

if __name__ == "__main__":
    demo.launch()
