
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

    def integrate_streamline(self, start_x, start_y, step_size=1.0, max_steps=100, collision_mask=None):
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
            
            if collision_mask is not None:
                mi, mj = int(cy), int(cx)
                if 0 <= mi < self.h and 0 <= mj < self.w:
                    if collision_mask[mi, mj] > 0:
                        break

            backward_points.append((cx, cy))
            
        return backward_points[::-1] + points

    def smooth_line(self, points):
        """
        Apply B-Spline interpolation to smooth the streamline points.
        """
        if splprep is None or len(points) < 4:
            return points
        
        try:
            # Filter close points to avoid errors
            unique_points = [points[0]]
            for i in range(1, len(points)):
                dist = np.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
                if dist > 0.1:
                    unique_points.append(points[i])
            
            if len(unique_points) < 4:
                return points

            x = [p[0] for p in unique_points]
            y = [p[1] for p in unique_points]
            
            # splprep with s=0 forces interpolation through all points
            # We can use a small s for smoothing if needed, e.g., s=len(points)*0.1
            tck, u = splprep([x, y], s=0, k=3) 
            
            # Generate smooth points
            u_new = np.linspace(u.min(), u.max(), len(unique_points))
            x_new, y_new = splev(u_new, tck)
            
            return list(zip(x_new, y_new))
        except Exception:
            return points

    def draw_tapered_line(self, canvas, points, width, sharpness):
        """
        Draw a tapered line using a polygon (ribbon) approach with sub-pixel precision.
        """
        if len(points) < 2:
            return

        pts = np.array(points, np.float32)
        n = pts.shape[0]
        
        # Calculate normals
        # For simplicity, use finite difference
        # T[i] = P[i+1] - P[i-1] (central difference)
        # Handle ends: T[0] = P[1] - P[0], T[n-1] = P[n-1] - P[n-2]
        
        # Vectors between points
        diffs = pts[1:] - pts[:-1] # Shape (n-1, 2)
        
        # Normalize diffs to get tangent segments
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Avoid div by zero
        tangents = diffs / norms
        
        # Vertex normals (average of adjacent segment tangents)
        # N_i is perpendicular to T_i
        # We need normals at each point.
        # N[i] = average of normal of segment i-1 and segment i
        
        # Segment normals: (-dy, dx)
        seg_normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        
        # Point normals
        point_normals = np.zeros((n, 2), dtype=np.float32)
        point_normals[0] = seg_normals[0]
        point_normals[-1] = seg_normals[-1]
        point_normals[1:-1] = (seg_normals[:-1] + seg_normals[1:]) * 0.5
        
        # Re-normalize point normals
        pn_norms = np.linalg.norm(point_normals, axis=1, keepdims=True)
        pn_norms[pn_norms == 0] = 1.0
        point_normals = point_normals / pn_norms
        
        # Calculate widths
        t = np.linspace(0.0, 1.0, n)
        # Profile: sin(pi * t)
        profiles = np.sin(np.pi * t)
        widths = width * ((1.0 - sharpness) + sharpness * profiles)
        
        # Calculate ribbon vertices
        # Left: P + N * w/2
        # Right: P - N * w/2
        half_widths = (widths * 0.5)[:, np.newaxis]
        offsets = point_normals * half_widths
        
        left_pts = pts + offsets
        right_pts = pts - offsets
        
        # Construct polygon: Left points forward, then Right points backward
        poly_pts = np.concatenate([left_pts, right_pts[::-1]], axis=0)
        
        # Sub-pixel rendering shift
        SHIFT = 4 # 2^4 = 16
        SCALE = 1 << SHIFT
        
        poly_pts_fixed = (poly_pts * SCALE).astype(np.int32)
        
        cv2.fillPoly(canvas, [poly_pts_fixed], (255, 255, 255), lineType=cv2.LINE_AA, shift=SHIFT)

    def render_svg(self, density=20, line_width=2.0, output_path="output.svg"):
        if svgwrite is None:
            print("svgwrite module not found. Skipping SVG generation.")
            return None
            
        # Use consistent generation method
        # Filter debris
        auto_min_len = int(max(15, density * 1.5))
        raw_lines = self.generate_streamlines(density, min_len=auto_min_len, show_progress=False)
                
        dwg = svgwrite.Drawing(output_path, profile='tiny', size=(self.w, self.h))
        
        # 背景
        # dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), rx=None, ry=None, fill='white'))
        
        for line_pts in raw_lines:
            # Smooth
            smoothed = self.smooth_line(line_pts)
            if len(smoothed) < 2: continue
            
            # 轉換為 SVG path
            path_data = "M" + " ".join([f"{p[0]},{p[1]}" for p in smoothed])
            dwg.add(dwg.path(d=path_data, stroke="black", stroke_width=line_width, fill="none"))
            
        return dwg.tostring()

    def generate_streamlines(self, density, min_len=5, show_progress=False):
        """
        Generate evenly spaced streamlines using a randomized grid with occupancy check.
        """
        occupancy_grid = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # Seed generation: Use a grid finer than density to ensure coverage
        # step = max(1, int(density / 3))
        # But for performance, maybe density / 2 is enough
        step = max(2, int(density / 2))
        
        seeds = []
        for y in range(0, self.h, step):
            for x in range(0, self.w, step):
                # Optimization: Check mask here too
                if self.mask is not None:
                    if self.mask[y, x] == 0:
                        continue
                seeds.append((x, y))
        
        # Shuffle seeds to avoid scanning artifacts
        # random.shuffle(seeds)
        # DISABLE SHUFFLE to prioritize long continuous lines (Scanline Seeding).
        # This prevents "fragmentation" where two seeds land on the same path and block each other, leaving a gap.
        # By iterating systematically, the first seed grows to full length, preventing gaps.
        
        lines = []
        
        iterator = seeds
        if show_progress and tqdm is not None:
             iterator = tqdm(seeds, desc="Generating streamlines", leave=False)
        
        collision_thickness = int(density)
        # Ensure minimal thickness
        if collision_thickness < 1: collision_thickness = 1
        
        for sx, sy in iterator:
            # Check occupancy at seed
            if occupancy_grid[sy, sx] > 0:
                continue
                
            # Integrate with collision check
            # Use smaller step_size for precision
            # Increase max_steps to ensure lines can traverse the entire image
            pts = self.integrate_streamline(sx, sy, step_size=0.7, max_steps=4000, collision_mask=occupancy_grid)
            
            if len(pts) < min_len:
                continue
            
            # If line is valid, mark occupancy
            # We use polylines to draw the exclusion zone
            pts_i32 = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(occupancy_grid, [pts_i32], False, 255, thickness=collision_thickness)
            
            lines.append(pts)
            
        return lines

    def render_from_lines(self, lines, line_width, taper_sharpness, canvas=None):
        """
        Render already computed (and smoothed) lines onto a canvas.
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
                # Use Ribbon Rendering for Tapered Lines
                self.draw_tapered_line(canvas, line_pts, line_width, taper_sharpness)
        return canvas

    def render_image(self, density=20, line_width=2, bg_image=None, show_progress=False, mask=None, taper_sharpness=0.0):
        self.mask = None
        if isinstance(mask, np.ndarray):
            if mask.shape[:2] == (self.h, self.w):
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                self.mask = mask
        
        # Use new generation method
        # Note: generate_streamlines handles seeds, mask, and progress
        # But we need to pass show_progress
        
        # Min length should be proportional to density to avoid "debris"
        # If a line is shorter than 1.5x spacing, it's likely a fragment between two other lines.
        # By rejecting it, we leave the space open for a potentially better seed that connects the gap.
        auto_min_len = int(max(15, density * 1.5))
        
        raw_lines = self.generate_streamlines(density, min_len=auto_min_len, show_progress=show_progress)
        
        smoothed_lines = []
        iterator = raw_lines
        if show_progress and tqdm is not None:
             iterator = tqdm(raw_lines, desc="Rendering lines", leave=True)
        
        for line_pts in iterator:
            # Smooth the line using Spline
            smoothed_lines.append(self.smooth_line(line_pts))
            
        return self.render_from_lines(smoothed_lines, line_width, taper_sharpness)
