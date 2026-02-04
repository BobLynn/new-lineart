
import numpy as np
import cv2
import random
try:
    import svgwrite
except ImportError:
    svgwrite = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from structure_tensor import eig_special_2d
except ImportError:
    eig_special_2d = None

try:
    from scipy.interpolate import splprep, splev
except ImportError:
    splprep = None
    splev = None

class StreamlineRenderer:
    """
    流線渲染器類
    負責根據張量場生成和渲染流線
    """
    def __init__(self, tensor_field, height, width):
        """
        初始化流線渲染器

        參數:
            tensor_field: 輸入的張量場 (H, W, 3) 或 (H, W, 2)
            height: 圖像高度
            width: 圖像寬度
        """
        self.field = tensor_field
        self.h = height
        self.w = width
        self.dir_field = None
        self.mask = None
        if eig_special_2d is not None:
            if (
                isinstance(self.field, np.ndarray)
                and self.field.ndim == 3
                and self.field.shape[2] == 3
            ):
                s = np.stack(
                    [self.field[:, :, 0], self.field[:, :, 1], self.field[:, :, 2]],
                    axis=0,
                )
                _, vec = eig_special_2d(s)
                if isinstance(vec, np.ndarray) and vec.shape[0] == 2:
                    self.dir_field = vec
        
    def get_direction(self, x, y):
        """
        獲取指定坐標處的流線方向

        參數:
            x: X 坐標
            y: Y 坐標

        返回:
            方向向量 (vx, vy) 或 None (如果坐標越界或向量長度過小)
        """
        xi, yi = int(x), int(y)
        if xi < 0 or xi >= self.w - 1 or yi < 0 or yi >= self.h - 1:
            return None
        if self.dir_field is not None:
            vx = float(self.dir_field[0, yi, xi])
            vy = float(self.dir_field[1, yi, xi])
            v = np.array([vx, vy], dtype=np.float32)
            n = np.linalg.norm(v)
            if n < 1e-8:
                return None
            return v / n
        t11 = self.field[yi, xi, 0]
        t12 = self.field[yi, xi, 1]
        t22 = self.field[yi, xi, 2]
        t = np.array([[t11, t12], [t12, t22]])
        _, evecs = np.linalg.eigh(t)
        return evecs[:, 1]

    def integrate_streamline(self, start_x, start_y, step_size=1.0, max_steps=100, collision_mask=None):
        """
        積分生成流線

        參數:
            start_x: 起始點 X 坐標
            start_y: 起始點 Y 坐標
            step_size: 積分步長
            max_steps: 最大步數
            collision_mask: 碰撞遮罩，用於檢查是否與現有流線重疊

        返回:
            points: 流線點列表 [(x, y), ...]
        """
        points = [(start_x, start_y)]
        cx, cy = start_x, start_y
        
        for _ in range(max_steps):
            vec = self.get_direction(cx, cy)
            if vec is None: break
            
            if len(points) > 1:
                prev_vec = np.array([cx - points[-2][0], cy - points[-2][1]])
                if np.dot(vec, prev_vec) < 0:
                    vec = -vec
            
            cx += vec[0] * step_size
            cy += vec[1] * step_size
            
            if not (0 <= cx < self.w and 0 <= cy < self.h):
                break
            if self.mask is not None:
                mi, mj = int(cy), int(cx)
                if mi < 0 or mi >= self.h or mj < 0 or mj >= self.w:
                    break
                if self.mask[mi, mj] == 0:
                    break
            
            if collision_mask is not None:
                mi, mj = int(cy), int(cx)
                if 0 <= mi < self.h and 0 <= mj < self.w:
                    if collision_mask[mi, mj] > 0:
                        break
                
            points.append((cx, cy))
            
        cx, cy = start_x, start_y
        backward_points = []
        first_vec = self.get_direction(start_x, start_y)
        if first_vec is None: return points
        
        curr_vec = -first_vec
        
        for _ in range(max_steps):
            vec = self.get_direction(cx, cy)
            if vec is None: break
            
            # 與當前後退方向對齊
            if np.dot(vec, curr_vec) < 0:
                vec = -vec
            
            curr_vec = vec # 更新下一步檢查的向量
            
            cx += vec[0] * step_size
            cy += vec[1] * step_size
            
            if not (0 <= cx < self.w and 0 <= cy < self.h):
                break
            if self.mask is not None:
                mi, mj = int(cy), int(cx)
                if mi < 0 or mi >= self.h or mj < 0 or mj >= self.w:
                    break
                if self.mask[mi, mj] == 0:
                    break
            
            if collision_mask is not None:
                mi, mj = int(cy), int(cx)
                if 0 <= mi < self.h and 0 <= mj < self.w:
                    if collision_mask[mi, mj] > 0:
                        break

            backward_points.append((cx, cy))
            
        return backward_points[::-1] + points

    def smooth_line(self, points):
        """
        應用 B-Spline 插值來平滑流線點。
        """
        if splprep is None or len(points) < 4:
            return points
        
        try:
            # 過濾接近點以避免錯誤
            unique_points = [points[0]]
            for i in range(1, len(points)):
                dist = np.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
                if dist > 0.1:
                    unique_points.append(points[i])
            
            if len(unique_points) < 4:
                return points

            x = [p[0] for p in unique_points]
            y = [p[1] for p in unique_points]
            
            # s=0 的 splprep 強制插值通過所有點
            # 如果需要，我們可以使用較小的 s 進行平滑，例如 s=len(points)*0.1
            tck, u = splprep([x, y], s=0, k=3) 
            
            # 生成平滑點
            u_new = np.linspace(u.min(), u.max(), len(unique_points))
            x_new, y_new = splev(u_new, tck)
            
            return list(zip(x_new, y_new))
        except Exception:
            return points

    def draw_tapered_line(self, canvas, points, width, sharpness):
        """
        使用多邊形（絲帶）方法以亞像素精度繪製錐形線。
        """
        if len(points) < 2:
            return

        pts = np.array(points, np.float32)
        n = pts.shape[0]
        
        # 計算法線
        # 為了簡單起見，使用有限差分
        # T[i] = P[i+1] - P[i-1] (中心差分)
        # 處理端點：T[0] = P[1] - P[0], T[n-1] = P[n-1] - P[n-2]
        
        # 點之間的向量
        diffs = pts[1:] - pts[:-1] # 形狀 (n-1, 2)
        
        # 歸一化差值以獲得切線段
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # 避免除以零
        tangents = diffs / norms
        
        # 頂點法線（相鄰線段切線的平均值）
        # N_i 垂直於 T_i
        # 我們需要每個點的法線。
        # N[i] = 線段 i-1 和線段 i 的法線的平均值
        
        # 線段法線：(-dy, dx)
        seg_normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        
        # 點法線
        point_normals = np.zeros((n, 2), dtype=np.float32)
        point_normals[0] = seg_normals[0]
        point_normals[-1] = seg_normals[-1]
        point_normals[1:-1] = (seg_normals[:-1] + seg_normals[1:]) * 0.5
        
        # 重新歸一化點法線
        pn_norms = np.linalg.norm(point_normals, axis=1, keepdims=True)
        pn_norms[pn_norms == 0] = 1.0
        point_normals = point_normals / pn_norms
        
        # 計算寬度
        t = np.linspace(0.0, 1.0, n)
        # 輪廓：sin(pi * t)
        profiles = np.sin(np.pi * t)
        widths = width * ((1.0 - sharpness) + sharpness * profiles)
        
        # 計算絲帶頂點
        # 左：P + N * w/2
        # 右：P - N * w/2
        half_widths = (widths * 0.5)[:, np.newaxis]
        offsets = point_normals * half_widths
        
        left_pts = pts + offsets
        right_pts = pts - offsets
        
        # 構建多邊形：左點向前，然後右點向後
        poly_pts = np.concatenate([left_pts, right_pts[::-1]], axis=0)
        
        # 亞像素渲染位移
        SHIFT = 4 # 2^4 = 16
        SCALE = 1 << SHIFT
        
        poly_pts_fixed = (poly_pts * SCALE).astype(np.int32)
        
        cv2.fillPoly(canvas, [poly_pts_fixed], (255, 255, 255), lineType=cv2.LINE_AA, shift=SHIFT)

    def render_svg(self, density=20, line_width=2.0, output_path="output.svg"):
        """
        將流線渲染為 SVG 文件

        參數:
            density: 流線密度
            line_width: 線條寬度
            output_path: 輸出文件路徑

        返回:
            SVG 內容字符串 或 None (如果 svgwrite 未安裝)
        """
        if svgwrite is None:
            print("svgwrite module not found. Skipping SVG generation.")
            return None
            
        # 使用一致的生成方法
        # 過濾碎片
        auto_min_len = int(max(15, density * 1.5))
        raw_lines = self.generate_streamlines(density, min_len=auto_min_len, show_progress=False)
                
        dwg = svgwrite.Drawing(output_path, profile='tiny', size=(self.w, self.h))
        
        # 背景
        # dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None, fill='white'))
        
        for line_pts in raw_lines:
            # 平滑處理
            smoothed = self.smooth_line(line_pts)
            if len(smoothed) < 2: continue
            
            # 轉換為 SVG path
            path_data = "M" + " ".join([f"{p[0]},{p[1]}" for p in smoothed])
            dwg.add(dwg.path(d=path_data, stroke="black", stroke_width=line_width, fill="none"))
            
        return dwg.tostring()

    def generate_streamlines(self, density, min_len=5, show_progress=False):
        """
        使用帶有佔用檢查的隨機網格生成均勻間隔的流線。

        參數:
            density: 流線密度 (控制網格步長和碰撞厚度)
            min_len: 最小流線長度 (點數)
            show_progress: 是否顯示進度條

        返回:
            lines: 流線列表，每個元素為點列表
        """
        occupancy_grid = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # 種子生成：使用比密度更細的網格以確保覆蓋
        # step = max(1, int(density / 3))
        # 但為了性能，也許 density / 2 就足夠了
        step = max(2, int(density / 2))
        
        seeds = []
        for y in range(0, self.h, step):
            for x in range(0, self.w, step):
                # 優化：也在這裡檢查遮罩
                if self.mask is not None:
                    if self.mask[y, x] == 0:
                        continue
                seeds.append((x, y))
        
        # 隨機種子以避免掃描偽影
        # random.shuffle(seeds)
        # 禁用隨機洗牌以優先考慮長連續線（掃描線播種）。
        # 這可以防止「碎片化」，即兩個種子落在同一路徑上並相互阻擋，從而留下間隙。
        # 通過系統地迭代，第一個種子會生長到全長，從而防止間隙。
        
        lines = []
        
        iterator = seeds
        if show_progress and tqdm is not None:
             iterator = tqdm(seeds, desc="Generating streamlines", leave=False)
        
        collision_thickness = int(density)
        # 確保最小厚度
        if collision_thickness < 1: collision_thickness = 1
        
        for sx, sy in iterator:
            # 檢查種子處的佔用情況
            if occupancy_grid[sy, sx] > 0:
                continue
                
            # 集成碰撞檢查
            # 使用較小的步長以提高精度
            # 增加最大步數以確保線條可以遍歷整個圖像
            pts = self.integrate_streamline(sx, sy, step_size=0.7, max_steps=4000, collision_mask=occupancy_grid)
            
            if len(pts) < min_len:
                continue
            
            # 如果線條有效，標記佔用
            # 我們使用折線繪製排除區域
            pts_i32 = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(occupancy_grid, [pts_i32], False, 255, thickness=collision_thickness)
            
            lines.append(pts)
            
        return lines

    def render_from_lines(self, lines, line_width, taper_sharpness, canvas=None):
        """
        將已計算（和平滑）的線條渲染到畫布上。

        參數:
            lines: 流線列表
            line_width: 線條寬度
            taper_sharpness: 錐度銳利度 (0.0 表示無錐度)
            canvas: 目標畫布 (可選)

        返回:
            canvas: 渲染後的畫布
        """
        if canvas is None:
            canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        for line_pts in lines:
            if len(line_pts) < 2: continue
            
            pts = np.array(line_pts, np.float32)
            if taper_sharpness <= 0:
                pts_i32 = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    canvas,
                    [pts_i32],
                    False,
                    (255, 255, 255),
                    thickness=int(line_width),
                    lineType=cv2.LINE_AA,
                )
            else:
                # 使用絲帶渲染繪製錐形線
                self.draw_tapered_line(canvas, line_pts, line_width, taper_sharpness)
        return canvas

    def render_image(self, density=20, line_width=2, bg_image=None, show_progress=False, mask=None, taper_sharpness=0.0):
        """
        渲染完整圖像

        參數:
            density: 流線密度
            line_width: 線條寬度
            bg_image: 背景圖像 (目前未使用)
            show_progress: 是否顯示進度條
            mask: 限制流線生成的遮罩
            taper_sharpness: 錐度銳利度

        返回:
            渲染後的圖像
        """
        self.mask = None
        if isinstance(mask, np.ndarray):
            if mask.shape[:2] == (self.h, self.w):
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                self.mask = mask
        
        # 使用新的生成方法
        # 注意：generate_streamlines 處理種子、遮罩和進度
        # 但我們需要傳遞 show_progress
        
        # 最小長度應與密度成正比以避免「碎片」
        # 如果一條線短於 1.5 倍間距，它很可能是兩條其他線之間的片段。
        # 通過拒絕它，我們為可能連接間隙的更好的種子留出了空間。
        auto_min_len = int(max(15, density * 1.5))
        
        raw_lines = self.generate_streamlines(density, min_len=auto_min_len, show_progress=show_progress)
        
        smoothed_lines = []
        iterator = raw_lines
        if show_progress and tqdm is not None:
             iterator = tqdm(raw_lines, desc="Rendering lines", leave=True)
        
        for line_pts in iterator:
            # 使用樣條平滑線條
            smoothed_lines.append(self.smooth_line(line_pts))
            
        return self.render_from_lines(smoothed_lines, line_width, taper_sharpness)
