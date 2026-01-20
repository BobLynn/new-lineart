import gradio as gr
import numpy as np
import cv2
import os
import sys
import pickle
import json
import subprocess
from core.segmentation import SAM3Engine
from core.tensor_solver import TensorFieldGenerator
from core.renderer import StreamlineRenderer
from utils.geometry import parse_gradio_sketch

# --- 初始化 SAM 3 ---
sam_engine = SAM3Engine(checkpoint_path="checkpoints/sam3.pt")

class SessionState:
    def __init__(self):
        self.raw_image = None       # 原始圖片
        self.active_mask = None     # 當前選中的區域 (Binary Mask)
        self.click_points = []
        self.click_labels = []
        self.tensor_field = None    # 緩存的張量場
        self.cached_lines = None    # 緩存的線條 (Smoothed)
        self.last_density = None    # 上一次渲染的密度

def on_upload(image, state):
    state = SessionState()
    state.raw_image = image
    sam_engine.set_image(image)
    return image, state

def clear_field_cache(state):
    """當用戶修改繪圖時，清除緩存的張量場"""
    state.tensor_field = None
    state.cached_lines = None
    state.last_density = None
    return state

def on_click(evt: gr.SelectData, state):
    """處理滑鼠點擊 -> SAM 3 推理"""
    if state.raw_image is None: return None, state
    
    # 記錄點擊 (左鍵=1, 前景)
    state.click_points.append([evt.index[0], evt.index[1]])
    state.click_labels.append(1)
    
    # SAM 3 推理
    mask = sam_engine.predict_click(state.click_points, state.click_labels)
    state.active_mask = mask
    
    # Mask changed, clear cache
    state.tensor_field = None
    state.cached_lines = None
    state.last_density = None
    
    # 視覺化：Overlay 紅色遮罩
    overlay = state.raw_image.copy()
    
    if mask is not None:
        colored_mask = np.zeros_like(overlay)
        colored_mask[:,:,0] = 255 # Red
        
        # Alpha blending
        mask_bool = mask > 0
        overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.6, colored_mask[mask_bool], 0.4, 0)
    else:
        # 如果推理失敗 (可能是沒模型)，顯示提示或原圖
        pass # 這裡可以加一些警告文字在圖上，但目前先保持原圖
    
    # 畫出點擊點
    for p in state.click_points:
        cv2.circle(overlay, tuple(p), 5, (0, 255, 0), -1)
        
    return overlay, state

def on_text_prompt(text, state):
    """處理文字輸入 -> SAM 3 推理 (適合選取複雜紋理區域)"""
    if not text or state.raw_image is None: return None, state
    
    mask = sam_engine.predict_text(text) # SAM 3 feature
    state.active_mask = mask
    
    # Mask changed, clear cache
    state.tensor_field = None
    state.cached_lines = None
    state.last_density = None

    # 視覺化：Overlay 紅色遮罩
    overlay = state.raw_image.copy()
    
    if mask is not None:
        colored_mask = np.zeros_like(overlay)
        colored_mask[:,:,0] = 255 # Red
        
        # Alpha blending
        mask_bool = mask > 0
        overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.6, colored_mask[mask_bool], 0.4, 0)
        
    return overlay, state

def prepare_drawing_canvas(state):
    """
    確認分割後，進入 Step 2。
    將非 Mask 區域變暗，讓用戶專注於在 Mask 內畫 Stroke。
    """
    if state.active_mask is None: return None
    
    # 創建一個 "Dimmed" 背景
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
        print("Computing Tensor Field for Preview...")
        stroke_constraints = parse_gradio_sketch(state.raw_image, drawing_dict)
        h, w = state.raw_image.shape[:2]
        solver = TensorFieldGenerator(h, w)
        state.tensor_field = solver.solve_field_with_mask(stroke_constraints, state.active_mask)
    
    # 2. 渲染
    h, w = state.raw_image.shape[:2]
    renderer = StreamlineRenderer(state.tensor_field, h, w)
    
    # Check cache
    need_new_lines = True
    if state.cached_lines is not None and state.last_density is not None:
        if abs(state.last_density - density) < 0.1:
            need_new_lines = False
            
    if need_new_lines:
        print(f"Generating new streamlines (Density: {density})...")
        # Ensure renderer uses correct mask
        renderer.mask = state.active_mask
        
        auto_min_len = int(max(15, density * 1.5))
        raw_lines = renderer.generate_streamlines(density, min_len=auto_min_len, show_progress=False)
        
        smoothed_lines = []
        for l in raw_lines:
            smoothed_lines.append(renderer.smooth_line(l))
            
        state.cached_lines = smoothed_lines
        state.last_density = density
    else:
        print("Using cached streamlines for preview...")

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
        return "Please generate a preview first (or wait for auto-generation)."
    
    # Save data
    data = {
        'tensor_field': state.tensor_field,
        'mask': state.active_mask
    }
    
    pkl_path = os.path.abspath("temp_preview.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
        
    # Launch subprocess
    script_path = os.path.join("core", "interactive_window.py")
    if not os.path.exists(script_path):
        return f"Error: {script_path} not found."
        
    # Run python
    cmd = [sys.executable, script_path, pkl_path]
    subprocess.Popen(cmd)
    
    return "Interactive Tuner Launched! Check your taskbar."

def load_tuner_params(state):
    """
    從 tuner_params.json 讀取參數並返回給滑桿
    """
    json_path = os.path.abspath("tuner_params.json")
    if not os.path.exists(json_path):
        # Return current values or defaults if file not found
        # Gradio updates: (density, width, sharpness, status)
        return gr.update(), gr.update(), gr.update(), "Error: No saved parameters found. Click 'Save to Web UI' in the Tuner first."
    
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        d = params.get("density", 20)
        w = params.get("width", 2)
        s = params.get("sharpness", 0.5)
        
        return d, w, s, f"Loaded parameters: D={d}, W={w}, S={s}"
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), f"Error loading parameters: {e}"

def run_hypnotic_gen(drawing_dict, density, width, sharpness, state):
    """
    最終生成
    """
    img, state = update_preview(drawing_dict, density, width, sharpness, state)
    return img, state

# --- Layout ---
css = """
#drawing-board {
    height: 80vh !important;
    max-height: 80vh !important;
}
.gradio-container {
    max_width: 100% !important;
}
"""
with gr.Blocks(title="SAM 3 Hypnotic Art", css=css) as demo:
    state = gr.State(SessionState())
    
    with gr.Tab("Step 1: Segment (SAM 3)"):
        with gr.Row():
            input_img = gr.Image(label="Upload Image", type="numpy")
            seg_preview = gr.Image(label="Segmentation Preview", interactive=False)
        
        with gr.Row():
            text_prompt = gr.Textbox(label="Text Prompt (Optional)", placeholder="e.g., 'cat eyes'")
            confirm_btn = gr.Button("Confirm Region & Next", variant="primary")

    with gr.Tab("Step 2: Draw Flow"):
        gr.Markdown("Draw red lines to guide the flow direction.")
        drawing_board = gr.ImageEditor(label="Draw Strokes", type="numpy", elem_id="drawing-board")
        
    with gr.Tab("Step 3: Preview & Tune"):
        with gr.Row():
            density_slider = gr.Slider(5, 50, value=20, label="Spacing Density")
            width_slider = gr.Slider(1, 10, value=2, label="Base Width")
            sharp_slider = gr.Slider(0, 1, value=0.5, label="Tapering Sharpness")
        
        btn_refresh_preview = gr.Button("Refresh Preview", variant="secondary")
        btn_interactive = gr.Button("Launch Interactive Tuner (Pygame)", variant="primary")
        btn_sync = gr.Button("Sync Parameters from Tuner", variant="secondary") # New Button
        status_msg = gr.Textbox(label="Status", interactive=False)
        preview_view = gr.Image(label="Lineart Preview", interactive=False)
        
    with gr.Tab("Step 4: Result"):
        gen_btn = gr.Button("Hypnotize!", variant="stop")
        result_view = gr.Image()
        
    # Events
    input_img.upload(on_upload, [input_img, state], [input_img, state])
    input_img.select(on_click, [state], [seg_preview, state]) # 修正: select 傳遞的是 evt
    text_prompt.submit(on_text_prompt, [text_prompt, state], [seg_preview, state])
    
    confirm_btn.click(prepare_drawing_canvas, inputs=[state], outputs=[drawing_board])
    
    # 當繪圖改變時，清除緩存
    drawing_board.change(clear_field_cache, inputs=[state], outputs=[state])
    
    # Preview Tab Events
    slider_inputs = [drawing_board, density_slider, width_slider, sharp_slider, state]
    
    # 滑桿實時更新
    density_slider.change(update_preview, inputs=slider_inputs, outputs=[preview_view, state])
    width_slider.change(update_preview, inputs=slider_inputs, outputs=[preview_view, state])
    sharp_slider.change(update_preview, inputs=slider_inputs, outputs=[preview_view, state])
    
    # 手動刷新按鈕
    btn_refresh_preview.click(update_preview, inputs=slider_inputs, outputs=[preview_view, state])
    
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
