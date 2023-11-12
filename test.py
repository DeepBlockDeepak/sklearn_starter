import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Example 1: 2D array with a singleton dimension
arr_2d = np.array([[1], [2], [3], [4]])  # Shape (4, 1)
squeezed_2d = arr_2d.squeeze()

# Example 2: 3D array with two singleton dimensions
arr_3d = np.array([[[1], [2], [3], [4]]])  # Shape (1, 4, 1)
squeezed_3d = arr_3d.squeeze()

# Shapes and arrays before and after squeezing
shapes_and_arrays = {
    "arr_2d_shape": arr_2d.shape,
    "squeezed_2d_shape": squeezed_2d.shape,
    "arr_2d": arr_2d,
    "squeezed_2d": squeezed_2d,
    "arr_3d_shape": arr_3d.shape,
    "squeezed_3d_shape": squeezed_3d.shape,
    "arr_3d": arr_3d,
    "squeezed_3d": squeezed_3d
}

for k,v in shapes_and_arrays.items():
    print(f"***___*** {k} : {v}\n")
