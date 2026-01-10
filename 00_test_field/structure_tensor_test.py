import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d

sigma = 1.5
rho = 5.5

# Load 2D data.
image = np.random.random((128, 128))

S = structure_tensor_2d(image, sigma, rho)
val, vec = eig_special_2d(S)