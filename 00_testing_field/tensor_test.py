import numpy as np
import matplotlib.pyplot as plt

def generate_hypnotic_field(height, width):
    # 1. 建立座標網格
    what_is_this = 20
    x = np.linspace(0, what_is_this, width)
    y = np.linspace(0, what_is_this, height)
    X, Y = np.meshgrid(x, y)

    # 2. 隨機生成一個純量場 (這裡簡單用 sin/cos 模擬 Perlin Noise 的效果)
    # 在實際專案中，這裡建議用 noise library
    scalar_field = np.sin(X) + np.cos(Y) + np.sin(X*0.5 + Y*0.5)

    # 3. 計算梯度 (Gradient) -> 向量
    grad_y, grad_x = np.gradient(scalar_field)

    # 4. 建構結構張量 (Structure Tensor) 的分量
    # T = [ Jxx  Jxy ]
    #     [ Jxy  Jyy ]
    Jxx = grad_x ** 2
    Jxy = grad_x * grad_y
    Jyy = grad_y ** 2

    # 5. 平滑化張量場 (這是產生迷幻感的關鍵！)
    # 利用高斯濾波器讓方向具有一致性
    from scipy.ndimage import gaussian_filter
    sigma = 20 # 模糊程度越大，線條越滑順
    Jxx = gaussian_filter(Jxx, sigma)
    Jxy = gaussian_filter(Jxy, sigma)
    Jyy = gaussian_filter(Jyy, sigma)

    # 6. 算出特徵向量的角度 (Principal Direction)
    # 這會給出與梯度垂直的方向 (即沿著紋理的方向)
    angles = 0.5 * np.arctan2(2 * Jxy, Jyy - Jxx)
    
    # 轉個 90 度，讓它沿著紋理流動，而不是切斷紋理
    angles += np.pi / 2 

    return angles

# 視覺化 (使用 Streamplot 模擬粒子流動)
h, w = 500, 500
field_angles = generate_hypnotic_field(h, w)

# 將角度轉回向量分量以便繪圖 (U, V)
U = np.cos(field_angles)
V = np.sin(field_angles)
patch_size = 7
plt.figure(figsize=(patch_size, patch_size))
# 使用流線圖 (Streamplot) 來呈現張量場的軌跡
plt.streamplot(np.arange(w), np.arange(h), U, V, density=2, color='purple', linewidth=0.5)
plt.axis('on')
plt.title("Tensor Field Art: Smoothed Structure Tensor")
plt.show()