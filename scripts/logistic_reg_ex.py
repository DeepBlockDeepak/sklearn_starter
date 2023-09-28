"""
Logistic Regression is used for binary classification problems. 
Despite the name "regression," it's actually used to 
categorize items into one of two classes, usually labeled as 0 and 1.
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate synthetic classification data
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_samples=1000
)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a Logistic Regression model and fit it to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Plotting the decision boundary
xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="white", linewidth=1)
plt.contour(xx, yy, probs, levels=[0.5], cmap="Greys", vmin=0, vmax=0.6)
plt.show()
