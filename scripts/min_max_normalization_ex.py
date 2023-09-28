# Min_Max_Normalization_Example.py

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Create a synthetic dataset with varying scales
X = np.array([[1., -1.,  2.],
              [2.,  0.,  0.],
              [0.,  1., -1.]])

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to the data and transform the dataset
X_scaled = scaler.fit_transform(X)

print(f"Original Data:\n{X}")
print(f"Scaled Data:\n{X_scaled}")
