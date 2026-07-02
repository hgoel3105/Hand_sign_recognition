import os
import numpy as np


class KNearestNeighbors:
    """K-Nearest Neighbors classifier implemented with vectorized distance matrices."""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.num_classes = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Stores reference training data for distance evaluation."""
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.num_classes = int(np.max(self.y_train)) + 1
        return self

    def _compute_euclidean_distances(self, X_test: np.ndarray) -> np.ndarray:
        # ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2(X * Y^T)
        test_sq = np.sum(np.square(X_test), axis=1, keepdims=True)
        train_sq = np.sum(np.square(self.X_train), axis=1, keepdims=True).T
        dot = np.dot(X_test, self.X_train.T)
        dist_sq = np.maximum(test_sq - 2.0 * dot + train_sq, 0.0)
        return np.sqrt(dist_sq)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Estimates class probability distributions across k nearest neighbors."""
        if self.X_train.size == 0:
            raise RuntimeError("Model has not been fitted.")

        is_1d = (X_test.ndim == 1)
        queries = X_test.reshape(1, -1) if is_1d else np.asarray(X_test, dtype=np.float64)
        n_queries = queries.shape[0]

        dists = self._compute_euclidean_distances(queries)
        eff_k = min(self.k, self.X_train.shape[0])
        
        # O(N) partial sort to extract k nearest indices
        idx = np.argpartition(dists, eff_k - 1, axis=1)[:, :eff_k]
        labels = self.y_train[idx]

        probs = np.zeros((n_queries, self.num_classes), dtype=np.float64)
        for i in range(n_queries):
            counts = np.bincount(labels[i], minlength=self.num_classes)
            probs[i] = counts / float(eff_k)

        return probs[0] if is_1d else probs

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)
        return int(np.argmax(probs)) if probs.ndim == 1 else np.argmax(probs, axis=1)

    def save(self, filepath: str) -> None:
        """Saves reference arrays to compressed archive."""
        if self.X_train.size == 0:
            raise RuntimeError("Cannot save unfitted model.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            k=np.array([self.k], dtype=np.int64),
            X_train=self.X_train,
            y_train=self.y_train,
            num_classes=np.array([self.num_classes], dtype=np.int64)
        )

    def load(self, filepath: str):
        """Loads training matrices directly from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing model binary: {filepath}")
        data = np.load(filepath)
        self.k = int(data["k"][0])
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.num_classes = int(data["num_classes"][0])
        return self
