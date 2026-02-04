import pygame  # 導入 Pygame 庫，用於創建互動視窗
import pickle  # 導入 pickle 模組，用於序列化/反序列化數據
import sys  # 導入 sys 模組，用於處理系統參數和路徑
import numpy as np  # 導入 NumPy 庫，用於數值計算
import cv2  # 導入 OpenCV 庫，用於圖像處理
import os  # 導入 os 模組，用於文件系統操作
import json  # 導入 json 模組，用於 JSON 文件讀寫

# 將項目根目錄添加到路徑，以便導入 core 模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.renderer import StreamlineRenderer  # 導入流線渲染器

# 定義顏色常量 (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 100, 255)
BLUE_HOVER = (80, 130, 255)

class Button:
    """
    簡單的 Pygame 按鈕類
    """
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)  # 按鈕矩形區域
        self.text = text  # 按鈕文字
        self.callback = callback  # 點擊回調函數
        self.hovered = False  # 懸停狀態

    def handle_event(self, event):
        """
        處理滑鼠事件
        """
        if event.type == pygame.MOUSEMOTION:
            # 更新懸停狀態
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 如果在懸停狀態下點擊，觸發回調
            if self.hovered:
                self.callback()
                return True
        return False

    def draw(self, screen, font):
        """
        繪製按鈕
        """
        color = BLUE_HOVER if self.hovered else BLUE  # 根據狀態選擇顏色
        pygame.draw.rect(screen, color, self.rect, border_radius=6)  # 繪製背景
        pygame.draw.rect(screen, WHITE, self.rect, width=2, border_radius=6)  # 繪製邊框
        
        text_surf = font.render(self.text, True, WHITE)  # 渲染文字
        text_rect = text_surf.get_rect(center=self.rect.center)  # 文字居中
        screen.blit(text_surf, text_rect)  # 繪製文字

class Slider:
    """
    簡單的 Pygame 滑桿類
    """
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, h)  # 滑動條矩形
        self.min_val = min_val  # 最小值
        self.max_val = max_val  # 最大值
        self.val = initial_val  # 當前值
        self.label = label  # 標籤
        self.dragging = False  # 拖動狀態
        self.circle_r = h // 2 + 4  # 滑塊半徑

    def handle_event(self, event):
        """
        處理滑鼠事件，返回數值是否改變
        """
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_over(event.pos):
                self.dragging = True  # 開始拖動
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False  # 停止拖動
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                # 計算新值
                mouse_x = event.pos[0]
                ratio = (mouse_x - self.rect.x) / self.rect.w
                ratio = max(0, min(1, ratio))  # 限制在 [0, 1] 範圍
                self.val = self.min_val + ratio * (self.max_val - self.min_val)  # 映射到數值範圍
                changed = True
        return changed

    def is_over(self, pos):
        """
        檢查滑鼠是否在滑塊或滑動條附近
        """
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + ratio * self.rect.w
        handle_y = self.rect.y + self.rect.h // 2
        dx = pos[0] - handle_x
        dy = pos[1] - handle_y
        # 檢查是否在滑塊圓形範圍內或矩形條範圍內
        return (dx*dx + dy*dy) <= (self.circle_r * self.circle_r * 2) or self.rect.collidepoint(pos)

    def draw(self, screen, font):
        """
        繪製滑桿
        """
        # 繪製標籤和當前值
        label_surf = font.render(f"{self.label}: {self.val:.2f}", True, WHITE)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 25))

        # 繪製滑動條背景
        pygame.draw.rect(screen, GRAY, self.rect, border_radius=4)
        
        # 計算滑塊位置
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + ratio * self.rect.w
        handle_y = self.rect.y + self.rect.h // 2
        # 繪製滑塊
        pygame.draw.circle(screen, GREEN, (int(handle_x), int(handle_y)), self.circle_r)

def main():
    """
    主程序入口
    """
    # 檢查命令行參數
    if len(sys.argv) < 2:
        print("Usage: python interactive_window.py <path_to_pkl>")
        return

    pkl_path = sys.argv[1]
    # 檢查數據文件是否存在
    if not os.path.exists(pkl_path):
        print("Data file not found.")
        return

    print(f"Loading data from {pkl_path}...")
    # 加載 pickle 數據 (張量場和遮罩)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    tensor_field = data['tensor_field']  # 獲取張量場
    mask = data['mask']  # 獲取遮罩
    h, w = tensor_field.shape[:2]  # 獲取尺寸
    
    # 初始化渲染器
    renderer = StreamlineRenderer(tensor_field, h, w)
    if mask is not None:
        renderer.mask = mask  # 設置遮罩
    
    # 初始化 Pygame
    pygame.init()
    
    CTRL_PANEL_WIDTH = 300  # 控制面板寬度
    screen_w = w + CTRL_PANEL_WIDTH  # 總窗口寬度
    screen_h = max(h, 400) # 確保高度足夠容納控制項
    
    # 創建窗口
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Lineart Real-time Tuner (Pygame)")  # 設置標題
    font = pygame.font.SysFont("Arial", 16)  # 設置字體
    
    # 初始化滑桿 (左側面板，垂直佈局)
    # 密度: 5 到 50
    # 寬度: 1 到 10
    # 清晰度: 0.0 到 1.0
    
    slider_x = 30
    slider_w = CTRL_PANEL_WIDTH - 60
    start_y = 50
    gap_y = 80
    
    sliders = [
        Slider(slider_x, start_y, slider_w, 10, 5, 50, 20, "Density"),  # 密度滑桿
        Slider(slider_x, start_y + gap_y, slider_w, 10, 1, 10, 2, "Width"),  # 寬度滑桿
        Slider(slider_x, start_y + gap_y * 2, slider_w, 10, 0.0, 1.0, 0.5, "Sharpness")  # 清晰度滑桿
    ]
    
    # 定義保存參數的回調函數
    def save_params():
        params = {
            "density": float(sliders[0].val),
            "width": float(sliders[1].val),
            "sharpness": float(sliders[2].val)
        }
        json_path = os.path.abspath("tuner_params.json")  # 保存路徑
        try:
            with open(json_path, 'w') as f:
                json.dump(params, f, indent=4)  # 寫入 JSON
            print(f"Parameters saved to {json_path}")
        except Exception as e:
            print(f"Error saving parameters: {e}")

    # 創建保存按鈕
    save_btn = Button(slider_x, start_y + gap_y * 3 + 20, slider_w, 40, "Save to Web UI", save_params)
    
    clock = pygame.time.Clock()  # 創建時鐘對象
    running = True  # 運行標誌
    
    # 緩存變量
    cached_lines = None
    last_rendered_density = -1
    last_rendered_width = -1
    last_rendered_sharpness = -1
    
    # 藝術品表面 (緩存渲染結果)
    art_surface = None
    
    def update_artwork(density, width, sharpness):
        """
        更新藝術品渲染
        """
        nonlocal cached_lines, last_rendered_density, last_rendered_width, last_rendered_sharpness, art_surface
        
        print(f"Updating artwork... (D={density:.1f}, W={width:.1f}, S={sharpness:.2f})")
        
        # 1. 如果密度發生顯著變化，則重新生成流線
        if cached_lines is None or abs(density - last_rendered_density) > 0.1:
            print("Re-integrating streamlines...")
            # 確保渲染器使用最新的遮罩
            if mask is not None:
                renderer.mask = mask
            
            # 根據密度自動計算最小長度，防止線條碎片化
            auto_min_len = int(max(15, density * 1.5))
            # 生成原始流線
            raw_lines = renderer.generate_streamlines(density, min_len=auto_min_len, show_progress=False)
            
            # 對流線進行平滑處理
            cached_lines = []
            for pts in raw_lines:
                 smoothed = renderer.smooth_line(pts)
                 cached_lines.append(smoothed)
            
            print(f"Generated {len(cached_lines)} lines.")
        
        # 2. 渲染到畫布
        canvas = np.zeros((h, w, 3), dtype=np.uint8)  # 創建黑色背景
        
        for line_pts in cached_lines:
            if len(line_pts) < 2: continue  # 跳過無效線條
            
            if sharpness <= 0:
                 # 如果清晰度為 0，使用快速折線繪製
                 pts = np.array(line_pts, np.int32)
                 cv2.polylines(canvas, [pts], False, (255, 255, 255), int(width), cv2.LINE_AA)
            else:
                # 否則使用絲帶渲染 (支持漸變寬度)
                renderer.draw_tapered_line(canvas, line_pts, width, sharpness)

        # 3. 創建 Pygame Surface
        art_surface = pygame.image.frombuffer(canvas.tobytes(), (w, h), "BGR")
        
        # 更新最後渲染的參數狀態
        last_rendered_density = density
        last_rendered_width = width
        last_rendered_sharpness = sharpness
        
    # 初始渲染
    update_artwork(sliders[0].val, sliders[1].val, sliders[2].val)

    # 主循環
    while running:
        screen.fill((30, 30, 30)) # 填充深灰色背景
        
        # 事件處理
        mouse_released = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # 退出循環
            
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_released = True  # 標記滑鼠釋放
            
            # 傳遞事件給滑桿
            for s in sliders:
                s.handle_event(event)
            
            # 傳遞事件給按鈕
            save_btn.handle_event(event)
        
        # 檢查是否需要更新 (僅在滑鼠釋放時，避免拖動時頻繁重繪)
        if mouse_released:
            curr_d = sliders[0].val
            curr_w = sliders[1].val
            curr_s = sliders[2].val
            
            # 檢查當前參數是否與已渲染的參數不同
            if (abs(curr_d - last_rendered_density) > 0.01 or 
                abs(curr_w - last_rendered_width) > 0.01 or 
                abs(curr_s - last_rendered_sharpness) > 0.01):
                
                # 觸發更新
                update_artwork(curr_d, curr_w, curr_s)
        
        # 繪製藝術品
        if art_surface:
            # 如果畫布高度小於螢幕高度，則垂直居中顯示
            art_y = (screen_h - h) // 2 if h < screen_h else 0
            screen.blit(art_surface, (CTRL_PANEL_WIDTH, art_y))
        
        # 繪製面板分隔線
        pygame.draw.line(screen, (80, 80, 80), (CTRL_PANEL_WIDTH, 0), (CTRL_PANEL_WIDTH, screen_h), 2)
        
        # 繪製所有滑桿
        for s in sliders:
            s.draw(screen, font)
        
        # 繪製按鈕
        save_btn.draw(screen, font)
            
        pygame.display.flip()  # 更新顯示
        clock.tick(30) # 限制幀率為 30 FPS，降低 CPU 佔用

    pygame.quit()  # 退出 Pygame

if __name__ == "__main__":
    main()
