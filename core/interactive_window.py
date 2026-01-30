import pygame
import pickle
import sys
import numpy as np
import cv2
import os
import json

# 將項目根目錄添加到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.renderer import StreamlineRenderer

# 顏色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 100, 255)
BLUE_HOVER = (80, 130, 255)

class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                self.callback()
                return True
        return False

    def draw(self, screen, font):
        color = BLUE_HOVER if self.hovered else BLUE
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        pygame.draw.rect(screen, WHITE, self.rect, width=2, border_radius=6)
        
        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.dragging = False
        self.circle_r = h // 2 + 4

    def handle_event(self, event):
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_over(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x = event.pos[0]
                ratio = (mouse_x - self.rect.x) / self.rect.w
                ratio = max(0, min(1, ratio))
                self.val = self.min_val + ratio * (self.max_val - self.min_val)
                changed = True
        return changed

    def is_over(self, pos):
        # 檢查滑鼠是否在手柄或條附近
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + ratio * self.rect.w
        handle_y = self.rect.y + self.rect.h // 2
        dx = pos[0] - handle_x
        dy = pos[1] - handle_y
        return (dx*dx + dy*dy) <= (self.circle_r * self.circle_r * 2) or self.rect.collidepoint(pos)

    def draw(self, screen, font):
        # 繪製標籤
        label_surf = font.render(f"{self.label}: {self.val:.2f}", True, WHITE)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 25))

        # 繪製條
        pygame.draw.rect(screen, GRAY, self.rect, border_radius=4)
        
        # 繪製手柄
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + ratio * self.rect.w
        handle_y = self.rect.y + self.rect.h // 2
        pygame.draw.circle(screen, GREEN, (int(handle_x), int(handle_y)), self.circle_r)

def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_window.py <path_to_pkl>")
        return

    pkl_path = sys.argv[1]
    if not os.path.exists(pkl_path):
        print("Data file not found.")
        return

    print(f"Loading data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    tensor_field = data['tensor_field']
    mask = data['mask']
    h, w = tensor_field.shape[:2]
    
    # 初始化渲染器
    renderer = StreamlineRenderer(tensor_field, h, w)
    if mask is not None:
        renderer.mask = mask
    
    # 初始化 Pygame
    pygame.init()
    
    CTRL_PANEL_WIDTH = 300
    screen_w = w + CTRL_PANEL_WIDTH
    screen_h = max(h, 400) # 確保控制項有足夠的高度
    
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Lineart Real-time Tuner (Pygame)")
    font = pygame.font.SysFont("Arial", 16)
    
    # 滑塊 (左側面板，垂直佈局)
    # 密度: 5 到 50
    # 寬度: 1 到 10
    # 清晰度: 0.0 到 1.0
    
    slider_x = 30
    slider_w = CTRL_PANEL_WIDTH - 60
    start_y = 50
    gap_y = 80
    
    sliders = [
        Slider(slider_x, start_y, slider_w, 10, 5, 50, 20, "Density"),
        Slider(slider_x, start_y + gap_y, slider_w, 10, 1, 10, 2, "Width"),
        Slider(slider_x, start_y + gap_y * 2, slider_w, 10, 0.0, 1.0, 0.5, "Sharpness")
    ]
    
    # 按鈕
    def save_params():
        params = {
            "density": float(sliders[0].val),
            "width": float(sliders[1].val),
            "sharpness": float(sliders[2].val)
        }
        json_path = os.path.abspath("tuner_params.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(params, f, indent=4)
            print(f"Parameters saved to {json_path}")
        except Exception as e:
            print(f"Error saving parameters: {e}")

    save_btn = Button(slider_x, start_y + gap_y * 3 + 20, slider_w, 40, "Save to Web UI", save_params)
    
    clock = pygame.time.Clock()
    running = True
    
    # 緩存
    cached_lines = None
    last_rendered_density = -1
    last_rendered_width = -1
    last_rendered_sharpness = -1
    
    # 藝術品表面
    art_surface = None
    
    def update_artwork(density, width, sharpness):
        nonlocal cached_lines, last_rendered_density, last_rendered_width, last_rendered_sharpness, art_surface
        
        print(f"Updating artwork... (D={density:.1f}, W={width:.1f}, S={sharpness:.2f})")
        
        # 1. 如果密度發生顯著變化，則重新積分
        if cached_lines is None or abs(density - last_rendered_density) > 0.1:
            print("Re-integrating streamlines...")
            # 使用渲染器的 ESSP 生成
            if mask is not None:
                renderer.mask = mask
            
            # 積極過濾短線以防止碎片化
            auto_min_len = int(max(15, density * 1.5))
            raw_lines = renderer.generate_streamlines(density, min_len=auto_min_len, show_progress=False)
            
            cached_lines = []
            for pts in raw_lines:
                 smoothed = renderer.smooth_line(pts)
                 cached_lines.append(smoothed)
            
            print(f"Generated {len(cached_lines)} lines.")
        
        # 2. 渲染到畫布
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        for line_pts in cached_lines:
            if len(line_pts) < 2: continue
            
            if sharpness <= 0:
                 # 快速折線繪製
                 pts = np.array(line_pts, np.int32)
                 cv2.polylines(canvas, [pts], False, (255, 255, 255), int(width), cv2.LINE_AA)
            else:
                # 使用絲帶渲染
                renderer.draw_tapered_line(canvas, line_pts, width, sharpness)

        # 3. 創建 Surface
        art_surface = pygame.image.frombuffer(canvas.tobytes(), (w, h), "BGR")
        
        # 更新狀態
        last_rendered_density = density
        last_rendered_width = width
        last_rendered_sharpness = sharpness
    # 初始渲染
    update_artwork(sliders[0].val, sliders[1].val, sliders[2].val)

    while running:
        screen.fill((30, 30, 30)) # 深灰色 UI 背景
        
        # 事件處理
        mouse_released = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_released = True
            
            for s in sliders:
                s.handle_event(event)
            
            save_btn.handle_event(event)
        
        # 檢查是否需要更新 (僅在釋放時)
        if mouse_released:
            curr_d = sliders[0].val
            curr_w = sliders[1].val
            curr_s = sliders[2].val
            
            # 檢查數值是否與已渲染的不同
            if (abs(curr_d - last_rendered_density) > 0.01 or 
                abs(curr_w - last_rendered_width) > 0.01 or 
                abs(curr_s - last_rendered_sharpness) > 0.01):
                
                update_artwork(curr_d, curr_w, curr_s)
        
        # 繪製藝術品
        if art_surface:
            # 如果畫布小於螢幕，則垂直居中
            art_y = (screen_h - h) // 2 if h < screen_h else 0
            screen.blit(art_surface, (CTRL_PANEL_WIDTH, art_y))
        
        # 繪製分隔線
        pygame.draw.line(screen, (80, 80, 80), (CTRL_PANEL_WIDTH, 0), (CTRL_PANEL_WIDTH, screen_h), 2)
        
        # 繪製滑塊
        for s in sliders:
            s.draw(screen, font)
        
        # 繪製按鈕
        save_btn.draw(screen, font)
            
        pygame.display.flip()
        clock.tick(30) # 限制 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
