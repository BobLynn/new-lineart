import numpy as np
from structure_tensor import structure_tensor_2d, eig_special_2d

# 1. 準備數據：這裡我們先隨機生成一個 100x100 的圖像作為範例
# 在實際應用中，你會用 imageio 或 cv2 讀取你的圖片
image = np.random.rand(100, 100)

# 2. 設定參數
sigma = 1.0  # 噪聲尺度 (Noise scale)：過濾掉小於這個尺寸的細節
rho = 5.0    # 積分尺度 (Integration scale)：決定要看多大範圍的鄰域

# 3. 計算結構張量 (Structure Tensor)
# S 包含了每個像素點的梯度訊息
S = structure_tensor_2d(image, sigma, rho)

# 4. 進行特徵分解 (Eigendecomposition)
# val: 特徵值 (Eigenvalues)，告訴你這個區域的「特徵強度」（是邊緣、角落還是平坦區域）
# vec: 特徵向量 (Eigenvectors)，告訴你這個區域的「主要方向」
val, vec = eig_special_2d(S)

print("計算完成！")
print("特徵值的形狀:", val.shape)  # 應該是 (2, 100, 100)
print("特徵向量的形狀:", vec.shape) # 應該是 (2, 100, 100)