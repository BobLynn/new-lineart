# core/tensor_solver.py
import numpy as np
import scipy.sparse as sp
import cv2
from scipy.sparse.linalg import spsolve
from typing import List, Tuple
import time

def log_solver(msg):
    """
    記錄求解器日誌到文件

    參數:
        msg: 日誌訊息字符串
    """
    with open("solver_debug.log", "a", encoding='utf-8') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

class TensorFieldGenerator:
    """
    張量場生成器類
    負責根據用戶約束和邊界條件求解拉普拉斯方程，生成平滑的張量場
    """
    def __init__(self, height: int, width: int):
        """
        初始化張量場生成器

        參數:
            height: 網格高度
            width: 網格寬度
        """
        self.h, self.w = height, width
        self.N = self.h * self.w
        log_solver(f"Initialized TensorFieldGenerator: {width}x{height} (N={self.N})")
        
    def _xy_to_idx(self, x, y):
        """
        將二維坐標轉換為一維索引

        參數:
            x: X 坐標
            y: Y 坐標
        
        返回:
            一維索引值
        """
        return y * self.w + x

    def solve_field(self, constraints: List[Tuple[int, int, float, float]]):
        """
        求解拉普拉斯方程：L * T = 0 s.t. 硬約束
        
        參數:
            constraints: (x, y, vector_x, vector_y) 的列表，代表硬約束點
        
        返回:
            求解後的張量場 (H, W, 3)
        """
        log_solver(f"solve_field called with {len(constraints)} constraints")
        print(f"Building system for {self.w}x{self.h} grid...")
        
        # 1. 構建拉普拉斯算子 (離散拉普拉斯算子)
        # 這裡使用 5-point stencil (上下左右中)
        diagonals = []
        offsets = []
        
        # 主對角線 (-4)
        diagonals.append(np.full(self.N, -4.0))
        offsets.append(0)
        
        # 鄰居 (+1)
        # 正確處理邊界以避免環繞
        # 右鄰居：如果 x < w-1 則有效。所以遮罩掉 x == w-1 的索引
        # (i+1) % w == 0 的索引 i 是右邊緣。
        ones = np.ones(self.N - 1)
        # 遮罩右環繞
        # i 從 0 到 N-2。
        # 如果 (i+1) % w == 0，那麼連接 i -> i+1 是從右邊緣跨越到下一行的左邊緣。
        # 我們想要斷開 (i) -- (i+1) 的連接。
        # 但是 offsets=1 意味著 A[i, i+1] = 1。
        # 所以我們需要將 (i+1)%w == 0 的 ones[i] 設為 0
        for i in range(self.N - 1):
            if (i + 1) % self.w == 0:
                ones[i] = 0.0
                
        diagonals.append(ones)
        offsets.append(1) # 右
        
        diagonals.append(ones) # 對稱
        offsets.append(-1) # 左
        
        diagonals.append(np.full(self.N - self.w, 1.0))
        offsets.append(self.w) # 下
        diagonals.append(np.full(self.N - self.w, 1.0))
        offsets.append(-self.w) # 上
        
        # 構建稀疏矩陣 A
        A = sp.diags(diagonals, offsets, shape=(self.N, self.N), format='lil')
        
        # RHS 向量 (T11, T12, T22 的目標值)
        b_t11 = np.zeros(self.N)
        b_t12 = np.zeros(self.N)
        b_t22 = np.zeros(self.N)
        
        # 3. 應用用戶約束 (硬約束)
        # 用戶畫線的地方，強制定義 Tensor 值
        constraint_indices = set()
        
        for i, (x, y, vx, vy) in enumerate(constraints):
            if not (0 <= x < self.w and 0 <= y < self.h): 
                continue
            
            idx = self._xy_to_idx(x, y)
            if idx in constraint_indices:
                # 記錄覆蓋？
                pass
            constraint_indices.add(idx)
            
            # 歸一化向量 u
            norm = np.sqrt(vx**2 + vy**2)
            if norm > 1e-6:
                ux, uy = vx/norm, vy/norm
            else:
                ux, uy = 1.0, 0.0 # 默認水平
                
            # 構造結構張量 T = u * u.T
            # T11 = ux*ux, T12 = ux*uy, T22 = uy*uy
            
            # 修改矩陣行：設為 Identity，強制 x_i = target
            A[idx, :] = 0
            A[idx, idx] = 1.0
            
            b_t11[idx] = ux * ux
            b_t12[idx] = ux * uy
            b_t22[idx] = uy * uy
            
            if i == 0:
                log_solver(f"Applied first constraint at ({x},{y}) idx={idx}: vec=({ux:.2f},{uy:.2f})")
            
        log_solver(f"Applied {len(constraint_indices)} unique hard constraints")
            
        # 轉換為 CSR 格式加速求解
        A_csr = A.tocsr()
        
        print("Solving sparse linear systems...")
        # 分別解三個分量
        t11_field = spsolve(A_csr, b_t11).reshape(self.h, self.w)
        t12_field = spsolve(A_csr, b_t12).reshape(self.h, self.w)
        t22_field = spsolve(A_csr, b_t22).reshape(self.h, self.w)
        
        return np.stack([t11_field, t12_field, t22_field], axis=-1)

    def solve_field_with_mask(self, stroke_constraints: List[Tuple[int, int, float, float]], mask: np.ndarray):
        """
        合併遮罩邊界約束和用戶筆觸約束進行求解。
        會自動處理邊界約束與用戶筆觸的衝突（抑制附近的邊界約束）。

        參數:
            stroke_constraints: 用戶筆觸約束列表 [(x, y, vx, vy), ...]
            mask: 二值遮罩 (uint8)，255 或 1 表示前景區域

        返回:
            求解後的張量場
        """
        log_solver(f"solve_field_with_mask called. Stroke constraints: {len(stroke_constraints)}")
        
        # 1. 準備用戶筆觸點以進行鄰近檢查
        user_pts = []
        if stroke_constraints:
            user_pts = np.array([[c[0], c[1]] for c in stroke_constraints])
            
        suppression_radius = 60.0 # 忽略用戶筆觸附近的邊界約束的像素半徑
        
        # 首先從邊界約束開始
        constraints = []
        
        # 調試可視化
        debug_viz = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        if mask is not None:
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # 在調試可視化上繪製遮罩 (灰色)
            debug_viz[mask > 0] = [50, 50, 50]
                
            # 提取邊緣
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            log_solver(f"Found {len(contours)} contours in mask")
            
            added_boundary_constraints = 0
            suppressed_boundary_constraints = 0
            
            for contour in contours:
                # 繪製輪廓 (深藍色)
                cv2.drawContours(debug_viz, [contour], -1, (100, 0, 0), 1)
                
                pts = contour.squeeze()
                if len(pts) < 3: continue
                
                for i in range(len(pts)):
                    # 下採樣邊界約束
                    if i % 2 != 0: continue
                    
                    p_curr = pts[i]
                    x, y = int(p_curr[0]), int(p_curr[1])
                    
                    # --- 鄰近檢查 ---
                    # 如果此邊界點距離任何用戶筆觸太近，請忽略它。
                    if len(user_pts) > 0:
                        # 計算到所有用戶點的距離
                        # 優化：先檢查邊界框？不，N 足夠小。
                        dists = np.sqrt(np.sum((user_pts - np.array([x, y]))**2, axis=1))
                        min_dist = np.min(dists)
                        
                        if min_dist < suppression_radius:
                            suppressed_boundary_constraints += 1
                            # 可視化被抑制的約束 (青色)
                            cv2.circle(debug_viz, (x, y), 1, (255, 255, 0), -1)
                            continue
                    
                    p_prev = pts[i-1]
                    p_next = pts[(i+1) % len(pts)]
                    
                    vx = p_next[0] - p_prev[0]
                    vy = p_next[1] - p_prev[1]
                    
                    constraints.append((x, y, vx, vy))
                    added_boundary_constraints += 1
                    # 可視化活動邊界 (藍色)
                    cv2.circle(debug_viz, (x, y), 1, (255, 0, 0), -1)
            
            log_solver(f"Added {added_boundary_constraints} boundary constraints")
            log_solver(f"Suppressed {suppressed_boundary_constraints} boundary constraints due to user stroke proximity")
            
        # 最後添加用戶筆觸約束
        # 應用膨脹
        dilation_radius = 2 
        user_constraint_indices = set()
        
        for (x, y, vx, vy) in stroke_constraints:
            for dy in range(-dilation_radius, dilation_radius + 1):
                for dx in range(-dilation_radius, dilation_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h:
                        constraints.append((nx, ny, vx, vy))
                        
        log_solver(f"Appended {len(stroke_constraints)} user constraints (expanded)")
            
        # 可視化筆觸約束（紅箭頭）
        for (x, y, vx, vy) in stroke_constraints:
            if 0 <= x < self.w and 0 <= y < self.h:
                norm = np.sqrt(vx**2 + vy**2)
                if norm > 0:
                    vx, vy = vx/norm * 10, vy/norm * 10
                    cv2.arrowedLine(debug_viz, (int(x), int(y)), (int(x+vx), int(y+vy)), (0, 0, 255), 1, tipLength=0.3)
                    
        cv2.imwrite("debug_constraints.png", debug_viz)
        log_solver("Saved debug_constraints.png")
                        
        return self.solve_field(constraints)