# Model_Selection_Example.py

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define two models for comparison: K-NN and Logistic Regression
knn_model = KNeighborsClassifier(n_neighbors=3)
log_model = LogisticRegression()

# Train and evaluate K-NN model
knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.score(X_test, y_test)
print(f"K-NN Model Accuracy: {knn_accuracy}")

# Train and evaluate Logistic Regression model
log_model.fit(X_train, y_train)
log_accuracy = log_model.score(X_test, y_test)
print(f"Logistic Regression Model Accuracy: {log_accuracy}")

# Use cross-validation to evaluate both models
knn_scores = cross_val_score(knn_model, X, y, cv=5)
log_scores = cross_val_score(log_model, X, y, cv=5)

print(f"K-NN Cross-Validation Scores: {knn_scores}")
print(f"K-NN Average Score: {np.mean(knn_scores)}")
print(f"Logistic Regression Cross-Validation Scores: {log_scores}")
print(f"Logistic Regression Average Score: {np.mean(log_scores)}")
