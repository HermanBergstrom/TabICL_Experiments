"""
Boilerplate code for KNN identification using sklearn breast_cancer dataset.
For each test sample, identifies the K nearest neighbors in the training set.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Standardize features (important for distance-based methods like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize NearestNeighbors
k = 5  # Number of nearest neighbors to find
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(X_train_scaled)

# Find K nearest neighbors for each test sample
distances, indices = knn.kneighbors(X_test_scaled)

print(f"\n\nFinding {k} nearest neighbors for each test sample...")
print("=" * 60)

# Display results for first 5 test samples as examples
for test_idx in range(min(5, len(X_test))):
    print(f"\nTest sample {test_idx}:")
    print(f"  True label: {y_test[test_idx]}")
    print(f"  KNN indices in training set: {indices[test_idx]}")
    print(f"  KNN labels: {y_train[indices[test_idx]]}")
    print(f"  Distances: {distances[test_idx]}")

# Optional: Store all results for further analysis
print("\n" + "=" * 60)
print(f"All KNN indices shape: {indices.shape}")
print(f"All distances shape: {distances.shape}")

# Example: Get majority vote from KNN labels for each test sample
majority_votes = np.zeros(len(X_test), dtype=int)
for test_idx in range(len(X_test)):
    knn_labels = y_train[indices[test_idx]]
    majority_votes[test_idx] = np.bincount(knn_labels).argmax()

print(f"\nMajority vote predictions from {k}NN:")
print(f"Accuracy: {np.mean(majority_votes == y_test):.4f}")
