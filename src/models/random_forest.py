"""
src/models/random_forest.py

Pure Classical ML Ensemble - Random Forest Classifier
Author: Senior Machine Learning Systems Architect

Strict Engineering Rules:
- NO SCIKIT-LEARN imported or used.
- Pure NumPy mathematical operations.
- Implements Bootstrap Aggregation (Bagging) across multiple DecisionTree models.
- Pure NumPy serialization (save/load using savez_compressed without pickle/joblib).
"""

import os
from typing import List, Optional, Union
import numpy as np
from src.models.decision_tree import DecisionTree


class RandomForestClassifier:
    """
    From-scratch Random Forest Classifier implementing Bootstrap Aggregation (Bagging)
    and Random Subspace feature selection.
    """

    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 2,
        max_features: Union[int, float, str, None] = "sqrt",
        random_state: Optional[int] = 42
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be strictly positive.")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        
        self.trees: List[DecisionTree] = []
        self.num_classes: int = 0
        self.rng = np.random.default_rng(seed=self.random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = False) -> "RandomForestClassifier":
        N = X_train.shape[0]
        self.num_classes = int(np.max(y_train)) + 1
        self.trees = []

        for t in range(self.n_estimators):
            bootstrap_indices = self.rng.choice(N, size=N, replace=True)
            X_bootstrap = X_train[bootstrap_indices]
            y_bootstrap = y_train[bootstrap_indices]

            tree_seed = None if self.random_state is None else (self.random_state + t * 997)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed
            )
            tree.fit(X_bootstrap, y_bootstrap, num_classes=self.num_classes)
            self.trees.append(tree)

            if verbose and (t + 1) % 10 == 0:
                print(f"[Random Forest] Fitted tree {t + 1:3d}/{self.n_estimators}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("RandomForestClassifier is untrained. Call fit() first.")

        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else np.asarray(X, dtype=np.float64)
        M = X_mat.shape[0]

        accumulated_probs = np.zeros((M, self.num_classes), dtype=np.float64)
        for tree in self.trees:
            accumulated_probs += tree.predict_proba(X_mat)

        ensemble_probs = accumulated_probs / float(len(self.trees))

        if is_1d:
            return ensemble_probs[0]
        return ensemble_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return int(np.argmax(probs))
        return np.argmax(probs, axis=1)

    def save(self, filepath: str) -> None:
        """
        Serializes all base DecisionTree arrays into a single compressed NPZ archive.
        """
        if not self.trees:
            raise RuntimeError("Cannot save untrained RandomForestClassifier.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        save_dict = {
            "n_estimators": np.array([len(self.trees)], dtype=np.int64),
            "num_classes": np.array([self.num_classes], dtype=np.int64)
        }

        for t_idx, tree in enumerate(self.trees):
            arrs = tree.to_arrays()
            for key, val in arrs.items():
                save_dict[f"tree_{t_idx}_{key}"] = val

        np.savez_compressed(filepath, **save_dict)

    def load(self, filepath: str) -> "RandomForestClassifier":
        """
        Reconstructs the forest of DecisionTrees directly from contiguous NPZ arrays.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Serialized Random Forest file not found: {filepath}")
        data = np.load(filepath)
        n_trees = int(data["n_estimators"][0])
        self.num_classes = int(data["num_classes"][0])
        self.trees = []

        for t_idx in range(n_trees):
            arrs = {
                "features": data[f"tree_{t_idx}_features"],
                "thresholds": data[f"tree_{t_idx}_thresholds"],
                "lefts": data[f"tree_{t_idx}_lefts"],
                "rights": data[f"tree_{t_idx}_rights"],
                "values": data[f"tree_{t_idx}_values"],
                "probs": data[f"tree_{t_idx}_probs"]
            }
            tree = DecisionTree.from_arrays(arrs, num_classes=self.num_classes)
            self.trees.append(tree)

        self.n_estimators = len(self.trees)
        return self
