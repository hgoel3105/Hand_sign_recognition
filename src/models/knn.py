"""
src/models/knn.py

Pure Classical ML Ensemble - K-Nearest Neighbors Classifier
Author: Senior Machine Learning Systems Architect

Strict Engineering Rules:
- NO SCIKIT-LEARN imported or used.
- Pure NumPy mathematical operations.
- Pure NumPy serialization (save/load using savez_compressed).
"""

import os
from typing import Dict, Tuple
import numpy as np


class KNearestNeighbors:
    """
    From-scratch K-Nearest Neighbors (KNN) Classifier implemented in pure NumPy.
    Utilizes exact vectorized Euclidean distance calculation.
    """

    def __init__(self, k: int = 5):
        if k <= 0:
            raise ValueError("Parameter k must be a strictly positive integer.")
        self.k = k
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.num_classes: int = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNearestNeighbors":
        """
        Stores training dataset references for lazy evaluation during inference.
        """
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D matrix.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train sample counts must match.")

        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.num_classes = int(np.max(self.y_train)) + 1
        return self

    def _compute_euclidean_distance_matrix(self, X_test: np.ndarray) -> np.ndarray:
        """
        Manually computes the Euclidean distance matrix between query samples (X_test)
        and stored training reference samples (X_train).
        """
        test_sq = np.sum(np.square(X_test), axis=1, keepdims=True)
        train_sq = np.sum(np.square(self.X_train), axis=1, keepdims=True).T
        dot_product = np.dot(X_test, self.X_train.T)
        sq_dist = test_sq - 2.0 * dot_product + train_sq
        sq_dist = np.maximum(sq_dist, 0.0)
        return np.sqrt(sq_dist)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Computes soft voting class probabilities across the k nearest neighbors.
        """
        if self.X_train.size == 0:
            raise RuntimeError("Model must be fitted with training data before predicting.")

        is_1d = (X_test.ndim == 1)
        queries = X_test.reshape(1, -1) if is_1d else np.asarray(X_test, dtype=np.float64)
        num_queries = queries.shape[0]

        dist_matrix = self._compute_euclidean_distance_matrix(queries)
        effective_k = min(self.k, self.X_train.shape[0])
        knn_indices = np.argpartition(dist_matrix, effective_k - 1, axis=1)[:, :effective_k]
        neighbor_labels = self.y_train[knn_indices]

        probabilities = np.zeros((num_queries, self.num_classes), dtype=np.float64)
        for i in range(num_queries):
            counts = np.bincount(neighbor_labels[i], minlength=self.num_classes)
            probabilities[i] = counts / float(effective_k)

        if is_1d:
            return probabilities[0]
        return probabilities

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)
        if probs.ndim == 1:
            return int(np.argmax(probs))
        return np.argmax(probs, axis=1)

    def save(self, filepath: str) -> None:
        """
        Serializes KNN training matrices to disk using pure NumPy savez_compressed.
        """
        if self.X_train.size == 0:
            raise RuntimeError("Cannot save untrained KNN model.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            k=np.array([self.k], dtype=np.int64),
            X_train=self.X_train,
            y_train=self.y_train,
            num_classes=np.array([self.num_classes], dtype=np.int64)
        )

    def load(self, filepath: str) -> "KNearestNeighbors":
        """
        Loads serialized KNN model matrices from disk without invoking fit().
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Serialized KNN model file not found: {filepath}")
        data = np.load(filepath)
        self.k = int(data["k"][0])
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.num_classes = int(data["num_classes"][0])
        return self
