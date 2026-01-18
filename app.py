import gradio as gr
import numpy as np
import cv2
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

def on_upload(image, state):
    state = SessionState()
    state.raw_image = image
    sam_engine.set_image(image)
    return image, state

def on_click(evt: gr.SelectData, state):
    """處理滑鼠點擊 -> SAM 3 推理"""
    if state.raw_image is None: return None, state
    
    # 記錄點擊 (左鍵=1, 前景)
    state.click_points.append([evt.index[0], evt.index[1]])
    state.click_labels.append(1)
    
    # SAM 3 推理
    mask = sam_engine.predict_click(state.click_points, state.click_labels)
    state.active_mask = mask
    
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

def run_hypnotic_gen(drawing_dict, density, width, sharpness, state):
    """
    最終生成：結合 Mask 約束和手繪 Stroke 約束
    """
    # 1. 提取用戶畫的紅線 (Strokes)
    stroke_constraints = parse_gradio_sketch(state.raw_image, drawing_dict)
    
    # 2. 構建張量場 (傳入 Mask 避免流線跑出邊界)
    h, w = state.raw_image.shape[:2]
    solver = TensorFieldGenerator(h, w)
    
    # 這是關鍵：Solver 必須同時尊重 "用戶筆畫方向" 和 "SAM 3 分割邊界"
    tensor_field = solver.solve_field_with_mask(stroke_constraints, state.active_mask)
    
    renderer = StreamlineRenderer(tensor_field, h, w)
    final_image = renderer.render_image(
        density=int(density),
        line_width=width,
        bg_image=None,
        show_progress=True,
        mask=state.active_mask,
        taper_sharpness=sharpness,
    )
    return final_image

# --- Layout ---
with gr.Blocks(title="SAM 3 Hypnotic Art") as demo:
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
        drawing_board = gr.ImageEditor(label="Draw Strokes", type="numpy")
        
    with gr.Tab("Step 3: Result"):
        with gr.Row():
            density_slider = gr.Slider(5, 50, value=20, label="Spacing Density")
            width_slider = gr.Slider(1, 10, value=2, label="Base Width")
            sharp_slider = gr.Slider(0, 1, value=0.5, label="Tapering Sharpness")
        gen_btn = gr.Button("Hypnotize!", variant="stop")
        result_view = gr.Image()
        
    # Events
    input_img.upload(on_upload, [input_img, state], [input_img, state])
    input_img.select(on_click, [state], [seg_preview, state]) # 修正: select 傳遞的是 evt
    text_prompt.submit(on_text_prompt, [text_prompt, state], [seg_preview, state])
    
    confirm_btn.click(prepare_drawing_canvas, inputs=[state], outputs=[drawing_board])
    
    gen_btn.click(run_hypnotic_gen, 
                 inputs=[drawing_board, density_slider, width_slider, sharp_slider, state],
                 outputs=[result_view])

if __name__ == "__main__":
    demo.launch()
