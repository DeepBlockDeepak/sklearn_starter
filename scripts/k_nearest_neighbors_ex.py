# K_Nearest_Neighbors_Example.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate synthetic classification data
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, n_samples=1000)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a K-NN model with k=3 and fit it to the training data
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Plot the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="white", linewidth=1)
plt.title("K-Nearest Neighbors")
plt.show()
