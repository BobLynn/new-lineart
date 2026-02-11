import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def get_noise(h, w, scale=10):
    """
    生成平滑的隨機噪音 (模擬 Perlin Noise)
    原理：先生成白雜訊，再用高斯模糊把它塗抹開，形成連續的起伏。
    """
    noise = np.random.randn(h, w)
    return gaussian_filter(noise, sigma=scale)

def get_tensor_field_angles(scalar_field, sigma=2.0):
    """
    【核心演算法】
    從純量場 (Scalar Field) 計算結構張量 (Structure Tensor)，並提取流線角度。
    """
    # 1. 計算梯度 (Gradient vector)
    grad_y, grad_x = np.gradient(scalar_field)
    
    # 2. 建構結構張量 (Structure Tensor) 的分量
    # T = [ Jxx  Jxy ]
    #     [ Jxy  Jyy ]
    # 這裡相當於外積：grad * grad.T
    Jxx = grad_x ** 2
    Jxy = grad_x * grad_y
    Jyy = grad_y ** 2
    
    # 3. 張量平滑化 (Tensor Smoothing)
    # 這是消除雜訊、產生「絲綢感」的關鍵
    Jxx = gaussian_filter(Jxx, sigma)
    Jxy = gaussian_filter(Jxy, sigma)
    Jyy = gaussian_filter(Jyy, sigma)
    
    # 4. 特徵分解 (Eigendecomposition) 的快速公式
    # 我們想要的是「最小特徵值」對應的方向 (Minor Eigenvector)
    # 因為梯度方向(Major)是變化最劇烈的方向，而我們想沿著紋理(變化最小)的方向畫線。
    # 公式：theta = 0.5 * atan2(2*Jxy, Jyy - Jxx)
    angles = 0.5 * np.arctan2(2 * Jxy, Jyy - Jxx)
    
    # 加上 90 度 (pi/2)，因為我們要垂直於梯度走 (沿著等高線)
    return angles + np.pi / 2

def generate_art():
    h, w = 600, 600
    x = np.linspace(-5, 5, w)
    y = np.linspace(-5, 5, h)
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ==========================================
    # 案例 1: 仿木紋 (Wood Grain)
    # 數學模型: 距離場 (Distance Field) + 噪音
    # ==========================================
    
    # 基礎：同心圓 (距離場)
    distance = np.sqrt(X**2 + Y**2)
    # 擾動：加一點噪音讓圓不規則
    noise = get_noise(h, w, scale=30)
    # 合成：sin 函數創造年輪的明暗條紋感
    wood_scalar = np.sin(distance * 10 + noise * 2)
    
    # 計算張量場角度
    wood_angles = get_tensor_field_angles(wood_scalar, sigma=3.0)
    
    # 繪圖
    axes[0].streamplot(np.arange(w), np.arange(h), 
                       np.cos(wood_angles), np.sin(wood_angles), 
                       density=2.5, color='#8B4513', linewidth=0.6, arrowsize=0)
    axes[0].set_title("Tensor Field: Wood Grain\n(Perturbed Distance Field)", fontsize=15)
    axes[0].invert_yaxis()
    axes[0].axis('off')

    # ==========================================
    # 案例 2: 大馬士革鋼 (Damascus Steel)
    # 數學模型: 域扭曲 (Domain Warping)
    # ==========================================
    
    # 基礎噪音
    base_noise = get_noise(h, w, scale=40)
    
    # 扭曲場：用另一個噪音去推擠座標
    warp_x = get_noise(h, w, scale=40) * 5.0
    warp_y = get_noise(h, w, scale=40) * 5.0
    
    # 這裡雖然沒顯式寫出複合函數，但邏輯上是 f(x + dx, y + dy)
    # 我們直接疊加產生這種「液態金屬」的流動感
    damascus_scalar = base_noise + 0.5 * np.sin(X * 2 + warp_x) + 0.5 * np.cos(Y * 2 + warp_y)
    
    # 計算張量場角度
    damascus_angles = get_tensor_field_angles(damascus_scalar, sigma=5.0)
    
    # 繪圖 (使用灰黑色調)
    axes[1].streamplot(np.arange(w), np.arange(h), 
                       np.cos(damascus_angles), np.sin(damascus_angles), 
                       density=3, color='#2F4F4F', linewidth=0.7, arrowsize=0)
    axes[1].set_title("Tensor Field: Damascus Steel\n(Domain Warping)", fontsize=15)
    axes[1].invert_yaxis()
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# 執行
if __name__ == "__main__":
    generate_art()