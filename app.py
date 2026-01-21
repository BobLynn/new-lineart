import gradio as gr
import numpy as np
import cv2
import os
import sys
import pickle
import json
import subprocess
from core.segmentation import SAM2AutoEngine
from core.tensor_solver import TensorFieldGenerator
from core.renderer import StreamlineRenderer
from utils.geometry import parse_gradio_sketch

# --- 初始化 SAM 2 ---
# Checkpoint path from 00_testing_field
CHECKPOINT_PATH = os.path.join("00_testing_field", "sam2_hiera_large.pt")
# Model Config
MODEL_CFG = "configs/sam2/sam2_hiera_l.yaml"

sam_engine = SAM2AutoEngine(checkpoint_path=CHECKPOINT_PATH, model_cfg=MODEL_CFG)

class SessionState:
    def __init__(self):
        self.raw_image = None       # 原始圖片
        self.active_mask = None     # 當前選中的區域 (Binary Mask) Combined
        self.tensor_field = None    # 緩存的張量場
        self.cached_lines = None    # 緩存的線條 (Smoothed)
        self.last_density = None    # 上一次渲染的密度
        
        # SAM 2 Specific
        self.sam2_masks = []          # List of filtered masks from auto-generator
        self.selected_indices = set() # Indices of masks currently selected

def combine_masks(masks, selected_indices, shape):
    """Combine selected masks into one binary mask"""
    final_mask = np.zeros(shape, dtype=np.uint8)
    if not masks or not selected_indices:
        return final_mask
        
    for idx in selected_indices:
        if idx < len(masks):
            # masks[idx]['segmentation'] is boolean
            m = masks[idx]['segmentation']
            final_mask[m] = 255
            
    return final_mask

def draw_sam2_overlay(image, masks, selected_indices):
    """
    Draw all masks. 
    Selected masks = Bright colors (Random but consistent).
    Unselected masks = Dim outlines or very faint color.
    """
    if image is None: return None
    overlay = image.copy()
    
    # 1. Draw Unselected Masks (Dim)
    # 2. Draw Selected Masks (Bright)
    
    # Pre-generate colors for consistency? 
    # We can hash the index to get a color
    np.random.seed(42)
    colors = [np.random.randint(0, 255, 3).tolist() for _ in range(len(masks))]
    
    # Create a canvas for mask blending
    mask_layer = np.zeros_like(overlay)
    
    for i, ann in enumerate(masks):
        m = ann['segmentation']
        color = colors[i]
        
        if i in selected_indices:
            # Selected: Bright alpha blend
            mask_layer[m] = color
        else:
            # Unselected: Faint or Outline?
            # Let's do very faint fill
            # Darken the color
            dim_color = [c // 4 for c in color]
            mask_layer[m] = dim_color
            
    # Composite
    # Mask areas where mask_layer > 0
    mask_bool = np.any(mask_layer > 0, axis=2)
    if np.any(mask_bool):
        overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.5, mask_layer[mask_bool], 0.5, 0)
        
    # Draw Contours for better visibility
    for i, ann in enumerate(masks):
        m_uint8 = ann['segmentation_uint8']
        contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if i in selected_indices:
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2) # White thick border
        else:
            cv2.drawContours(overlay, contours, -1, (128, 128, 128), 1) # Gray thin border

    return overlay

def on_upload(image, state):
    state = SessionState()
    state.raw_image = image
    
    if image is not None:
        # Run SAM 2 Auto Segmentation
        state.sam2_masks = sam_engine.generate_masks(image)
        
        # Default: Select ALL filtered masks
        # The filter logic is supposed to keep only the object, so selecting all is a good default.
        state.selected_indices = set(range(len(state.sam2_masks)))
        
        # Combine
        state.active_mask = combine_masks(state.sam2_masks, state.selected_indices, image.shape[:2])
        
        # Draw Overlay
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
    """處理滑鼠點擊 -> Toggle Mask Selection"""
    if state.raw_image is None or not state.sam2_masks: return None, state
    
    # Find which mask was clicked
    idx = sam_engine.get_mask_at_point(evt.index[0], evt.index[1])
    
    if idx != -1:
        if idx in state.selected_indices:
            state.selected_indices.remove(idx)
        else:
            state.selected_indices.add(idx)
            
        # Re-combine
        state.active_mask = combine_masks(state.sam2_masks, state.selected_indices, state.raw_image.shape[:2])
        
        # Mask changed, clear cache
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
with gr.Blocks(title="SAM 2 Hypnotic Art", css=css) as demo:
    state = gr.State(SessionState())
    
    with gr.Tab("Step 1: Select Objects (SAM 2)"):
        gr.Markdown("Upload an image. SAM 2 will automatically segment it. Click on regions to Select/Deselect them.")
        with gr.Row():
            input_img = gr.Image(label="Upload Image", type="numpy")
            seg_preview = gr.Image(label="Segmentation Preview", interactive=False)
        
        with gr.Row():
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
    input_img.upload(on_upload, [input_img, state], [input_img, seg_preview, state])
    input_img.select(on_click, [state], [seg_preview, state])
    
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
