
import numpy as np
import cv2
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

class StreamlineRenderer:
    def __init__(self, tensor_field, height, width):
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

    def integrate_streamline(self, start_x, start_y, step_size=1.0, max_steps=100):
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
                
            points.append((cx, cy))
            
        cx, cy = start_x, start_y
        backward_points = []
        first_vec = self.get_direction(start_x, start_y)
        if first_vec is None: return points
        
        curr_vec = -first_vec
        
        for _ in range(max_steps):
            vec = self.get_direction(cx, cy)
            if vec is None: break
            
            # Align with current backward direction
            if np.dot(vec, curr_vec) < 0:
                vec = -vec
            
            curr_vec = vec # Update for next step check
            
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
                
            backward_points.append((cx, cy))
            
        return backward_points[::-1] + points

    def render_svg(self, density=20, line_width=2.0, output_path="output.svg"):
        if svgwrite is None:
            print("svgwrite module not found. Skipping SVG generation.")
            return None
            
        # 簡單的種子點生成：網格採樣
        seeds = []
        for y in range(0, self.h, density):
            for x in range(0, self.w, density):
                seeds.append((x, y))
                
        dwg = svgwrite.Drawing(output_path, profile='tiny', size=(self.w, self.h))
        
        # 背景
        # dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None, fill='white'))
        
        for sx, sy in seeds:
            line_pts = self.integrate_streamline(sx, sy, step_size=2.0, max_steps=50)
            if len(line_pts) < 5: continue
            
            # 轉換為 SVG path
            path_data = "M" + " ".join([f"{p[0]},{p[1]}" for p in line_pts])
            dwg.add(dwg.path(d=path_data, stroke="black", stroke_width=line_width, fill="none"))
            
        return dwg.tostring()

    def render_image(self, density=20, line_width=2, bg_image=None, show_progress=False, mask=None, taper_sharpness=0.0):
        self.mask = None
        if isinstance(mask, np.ndarray):
            if mask.shape[:2] == (self.h, self.w):
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                self.mask = mask
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        seeds = []
        for y in range(int(density / 2), self.h, density):
            for x in range(int(density / 2), self.w, density):
                if self.mask is not None:
                    if self.mask[y, x] == 0:
                        continue
                seeds.append((x, y))
        iterator = seeds
        if show_progress:
            if tqdm is None:
                print("tqdm not installed, skipping progress bar (pip install tqdm to enable).")
            else:
                print(f"Rendering {len(seeds)} streamlines with progress bar...")
                iterator = tqdm(seeds, desc="Rendering streamlines", total=len(seeds), leave=True)
        for sx, sy in iterator:
            line_pts = self.integrate_streamline(sx, sy, step_size=0.7, max_steps=400)
            if len(line_pts) < 2:
                continue
            pts = np.array(line_pts, np.float32)
            n = pts.shape[0]
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
                continue
            t = np.linspace(0.0, 1.0, n)
            for i in range(n - 1):
                p0_arr = pts[i]
                p1_arr = pts[i + 1]
                seg_vec = p1_arr - p0_arr
                seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
                if seg_len <= 0:
                    continue
                samples = max(2, int(np.ceil(seg_len / 0.7)))
                for j in range(samples):
                    s_local = (j + 0.5) / samples
                    pos = p0_arr + seg_vec * s_local
                    tj = (i + s_local) / (n - 1)
                    profile = np.sin(np.pi * tj)
                    w = line_width * ((1.0 - taper_sharpness) + taper_sharpness * profile)
                    radius = max(0.5, 0.5 * w)
                    r_int = max(1, int(round(radius)))
                    cx = int(round(pos[0]))
                    cy = int(round(pos[1]))
                    cv2.circle(
                        canvas,
                        (cx, cy),
                        r_int,
                        (255, 255, 255),
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )
        return canvas
