"""
Linear Regression is used to predict a continuous value based 
on one or more input features. 
The simplest form is Simple Linear Regression, 
which relies on just one input feature.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)
# Generate synthetic data
X = np.random.rand(100, 1)  # Feature
y = 3 * X.squeeze() + 2 + np.random.randn(100)  # Labels with some noise

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.legend()
plt.show()

# Print model parameters
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Calculate score
score = model.score(X_test, y_test)
print(f"Model Score: {score}")
