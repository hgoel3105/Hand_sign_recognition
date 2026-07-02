import os
import numpy as np
from src.models.decision_tree import DecisionTree


class RandomForestClassifier:
    """Ensemble classifier utilizing bootstrap aggregation (bagging) over CART decision trees."""

    def __init__(self, n_estimators=30, max_depth=12, min_samples_split=2, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.num_classes = 0
        self.rng = np.random.default_rng(seed=self.random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = False):
        """Fits individual trees on bootstrapped data subsamples."""
        n_samples = X_train.shape[0]
        self.num_classes = int(np.max(y_train)) + 1
        self.trees = []

        for t in range(self.n_estimators):
            idx = self.rng.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X_train[idx], y_train[idx]

            seed = None if self.random_state is None else (self.random_state + t * 997)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=seed
            )
            tree.fit(X_boot, y_boot, num_classes=self.num_classes)
            self.trees.append(tree)

            if verbose and (t + 1) % 10 == 0:
                print(f"Fitted tree {t + 1:3d}/{self.n_estimators}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Averages predicted probability distributions across all trees."""
        if not self.trees:
            raise RuntimeError("Forest has not been fitted.")

        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else X
        m = X_mat.shape[0]

        acc = np.zeros((m, self.num_classes), dtype=np.float64)
        for tree in self.trees:
            acc += tree.predict_proba(X_mat)

        probs = acc / float(len(self.trees))
        return probs[0] if is_1d else probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return int(np.argmax(probs)) if probs.ndim == 1 else np.argmax(probs, axis=1)

    def save(self, filepath: str) -> None:
        """Serializes base tree arrays into a compressed archive."""
        if not self.trees:
            raise RuntimeError("Cannot save unfitted forest.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        data = {
            "n_estimators": np.array([len(self.trees)], dtype=np.int64),
            "num_classes": np.array([self.num_classes], dtype=np.int64)
        }

        for i, tree in enumerate(self.trees):
            arrs = tree.to_arrays()
            for k, v in arrs.items():
                data[f"tree_{i}_{k}"] = v

        np.savez_compressed(filepath, **data)

    def load(self, filepath: str):
        """Reconstructs forest trees from contiguous compressed arrays."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing forest binary: {filepath}")
        data = np.load(filepath)
        n_trees = int(data["n_estimators"][0])
        self.num_classes = int(data["num_classes"][0])
        self.trees = []

        for i in range(n_trees):
            arrs = {
                "features": data[f"tree_{i}_features"],
                "thresholds": data[f"tree_{i}_thresholds"],
                "lefts": data[f"tree_{i}_lefts"],
                "rights": data[f"tree_{i}_rights"],
                "values": data[f"tree_{i}_values"],
                "probs": data[f"tree_{i}_probs"]
            }
            self.trees.append(DecisionTree.from_arrays(arrs, num_classes=self.num_classes))

        self.n_estimators = len(self.trees)
        return self
