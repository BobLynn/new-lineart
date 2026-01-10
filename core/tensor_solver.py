# core/tensor_solver.py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import List, Tuple

class TensorFieldGenerator:
    def __init__(self, height: int, width: int):
        self.h, self.w = height, width
        self.N = self.h * self.w
        
    def _xy_to_idx(self, x, y):
        return y * self.w + x

    def solve_field(self, constraints: List[Tuple[int, int, float, float]]):
        """
        constraints: List of (x, y, vector_x, vector_y)
        Solving Laplacian equation: L * T = 0 s.t. Hard Constraints
        """
        print(f"Building system for {self.w}x{self.h} grid...")
        
        # 1. 構建拉普拉斯算子 (Discrete Laplacian Operator)
        # 這裡使用 5-point stencil (上下左右中)
        diagonals = []
        offsets = []
        
        # Main diagonal (-4)
        diagonals.append(np.full(self.N, -4.0))
        offsets.append(0)
        
        # Neighbors (+1)
        diagonals.append(np.full(self.N - 1, 1.0))
        offsets.append(1) # Right
        diagonals.append(np.full(self.N - 1, 1.0))
        offsets.append(-1) # Left
        diagonals.append(np.full(self.N - self.w, 1.0))
        offsets.append(self.w) # Down
        diagonals.append(np.full(self.N - self.w, 1.0))
        offsets.append(-self.w) # Up
        
        # 構建稀疏矩陣 A
        A = sp.diags(diagonals, offsets, shape=(self.N, self.N), format='lil')
        
        # RHS vectors (Target values for T11, T12, T22)
        b_t11 = np.zeros(self.N)
        b_t12 = np.zeros(self.N)
        b_t22 = np.zeros(self.N)
        
        # 2. 應用 Neumann 邊界條件 (論文提及 [cite: 202])
        # 這裡為了簡單，我們假設圖像邊緣是自由的，或者用 padding 處理。
        # 實作上，如果不處理邊界，稀疏矩陣求解可能會不收斂。
        # 簡單 hack: 固定四個角的值為 0 或 Identity。
        
        # 3. 應用用戶約束 (Hard Constraints)
        # 用戶畫線的地方，強制定義 Tensor 值
        for x, y, vx, vy in constraints:
            if not (0 <= x < self.w and 0 <= y < self.h): continue
            
            idx = self._xy_to_idx(x, y)
            
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
            
        # 轉換為 CSR 格式加速求解
        A_csr = A.tocsr()
        
        print("Solving sparse linear systems...")
        # 分別解三個分量
        t11_field = spsolve(A_csr, b_t11).reshape(self.h, self.w)
        t12_field = spsolve(A_csr, b_t12).reshape(self.h, self.w)
        t22_field = spsolve(A_csr, b_t22).reshape(self.h, self.w)
        
        return np.stack([t11_field, t12_field, t22_field], axis=-1)