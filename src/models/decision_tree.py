"""
src/models/decision_tree.py

Pure Classical ML Ensemble - Decision Tree Classifier (CART)
Author: Senior Machine Learning Systems Architect

Strict Engineering Rules:
- NO SCIKIT-LEARN imported or used.
- Pure NumPy mathematical operations.
- Recursive CART tree splitting logic implemented from scratch.
- Exact Gini Impurity evaluation: G = 1 - sum(p_i^2).
- Pure NumPy contiguous array serialization support.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class TreeNode:
    """
    Represents a single node in the Decision Tree.
    Can be an internal split node or a terminal leaf node.
    """

    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[int] = None,
        probabilities: Optional[np.ndarray] = None
    ):
        self.feature_index: Optional[int] = feature_index
        self.threshold: Optional[float] = threshold
        self.left: Optional["TreeNode"] = left
        self.right: Optional["TreeNode"] = right
        self.value: Optional[int] = value
        self.probabilities: Optional[np.ndarray] = probabilities

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    """
    From-scratch Classification and Regression Tree (CART) implementation.
    """

    def __init__(
        self,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 1e-7,
        max_features: Union[int, float, str, None] = None,
        random_state: Optional[int] = None
    ):
        self.max_depth = max_depth if max_depth is not None else 9999
        self.min_samples_split = max(2, min_samples_split)
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        
        self.root: Optional[TreeNode] = None
        self.num_classes: int = 0
        self.rng = np.random.default_rng(seed=self.random_state)

    @staticmethod
    def _compute_gini_impurity(y: np.ndarray) -> float:
        N = y.size
        if N == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / float(N)
        return float(1.0 - np.sum(np.square(probabilities)))

    def _determine_feature_subset_size(self, total_features: int) -> int:
        if self.max_features is None:
            return total_features
        if isinstance(self.max_features, str):
            if self.max_features.lower() == "sqrt":
                return max(1, int(np.sqrt(total_features)))
            elif self.max_features.lower() == "log2":
                return max(1, int(np.log2(total_features)))
            else:
                raise ValueError(f"Unknown max_features string: {self.max_features}")
        elif isinstance(self.max_features, float):
            return max(1, min(total_features, int(self.max_features * total_features)))
        elif isinstance(self.max_features, int):
            return max(1, min(total_features, self.max_features))
        return total_features

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        N, D = X.shape
        parent_gini = self._compute_gini_impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        num_features_to_sample = self._determine_feature_subset_size(D)
        feature_indices = self.rng.choice(D, size=num_features_to_sample, replace=False)

        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            if unique_values.size <= 1:
                continue

            if unique_values.size > 25:
                thresholds = np.unique(np.percentile(feature_values, np.linspace(4, 96, 25)))
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                n_left = int(np.sum(left_mask))
                n_right = N - n_left

                if n_left == 0 or n_right == 0:
                    continue

                gini_left = self._compute_gini_impurity(y[left_mask])
                gini_right = self._compute_gini_impurity(y[right_mask])
                weighted_gini = (n_left / float(N)) * gini_left + (n_right / float(N)) * gini_right

                impurity_gain = parent_gini - weighted_gini

                if impurity_gain > best_gain:
                    best_gain = impurity_gain
                    best_feature = feature_idx
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        N = y.size
        counts = np.bincount(y, minlength=self.num_classes)
        leaf_probs = counts / float(N) if N > 0 else np.zeros((self.num_classes,), dtype=np.float64)
        leaf_val = int(np.argmax(counts))

        if (
            depth >= self.max_depth or
            N < self.min_samples_split or
            np.unique(y).size == 1
        ):
            return TreeNode(value=leaf_val, probabilities=leaf_probs)

        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        if best_feature is None or best_gain < self.min_impurity_decrease:
            return TreeNode(value=leaf_val, probabilities=leaf_probs)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, num_classes: Optional[int] = None) -> "DecisionTree":
        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D matrix.")
        
        self.num_classes = num_classes if num_classes is not None else int(np.max(y_train)) + 1
        self.root = self._build_tree(X_train, y_train, depth=0)
        return self

    def _predict_single_sample(self, node: TreeNode, x: np.ndarray) -> Tuple[int, np.ndarray]:
        if node.is_leaf:
            return node.value, node.probabilities
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_single_sample(node.left, x)
        else:
            return self._predict_single_sample(node.right, x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("DecisionTree is untrained. Call fit() first.")

        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else np.asarray(X, dtype=np.float64)
        M = X_mat.shape[0]

        probs = np.zeros((M, self.num_classes), dtype=np.float64)
        for i in range(M):
            _, sample_probs = self._predict_single_sample(self.root, X_mat[i])
            probs[i] = sample_probs

        return probs[0] if is_1d else probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return int(np.argmax(probs))
        return np.argmax(probs, axis=1)

    def _flatten_node(self, node: TreeNode, nodes_list: List[Dict]) -> int:
        """Helper to recursively flatten pointer tree into parallel node dictionaries."""
        node_id = len(nodes_list)
        nodes_list.append({})  # Reserve slot
        if node.is_leaf:
            nodes_list[node_id] = {
                "feature": -1,
                "threshold": 0.0,
                "left": -1,
                "right": -1,
                "value": node.value,
                "probs": node.probabilities if node.probabilities is not None else np.zeros((self.num_classes,))
            }
        else:
            left_id = self._flatten_node(node.left, nodes_list)
            right_id = self._flatten_node(node.right, nodes_list)
            nodes_list[node_id] = {
                "feature": node.feature_index,
                "threshold": node.threshold,
                "left": left_id,
                "right": right_id,
                "value": -1,
                "probs": np.zeros((self.num_classes,))
            }
        return node_id

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """
        Converts the binary tree structure into contiguous 1D NumPy arrays for serialization.
        """
        if self.root is None:
            raise RuntimeError("Cannot serialize untrained DecisionTree.")
        nodes_list: List[Dict] = []
        self._flatten_node(self.root, nodes_list)

        features = np.array([n["feature"] for n in nodes_list], dtype=np.int64)
        thresholds = np.array([n["threshold"] for n in nodes_list], dtype=np.float64)
        lefts = np.array([n["left"] for n in nodes_list], dtype=np.int64)
        rights = np.array([n["right"] for n in nodes_list], dtype=np.int64)
        values = np.array([n["value"] for n in nodes_list], dtype=np.int64)
        probs = np.array([n["probs"] for n in nodes_list], dtype=np.float64)

        return {
            "features": features,
            "thresholds": thresholds,
            "lefts": lefts,
            "rights": rights,
            "values": values,
            "probs": probs
        }

    @classmethod
    def from_arrays(cls, arrays: Dict[str, np.ndarray], num_classes: int) -> "DecisionTree":
        """
        Reconstructs a DecisionTree hierarchy from serialized contiguous 1D arrays.
        """
        tree = cls()
        tree.num_classes = num_classes

        features = arrays["features"]
        thresholds = arrays["thresholds"]
        lefts = arrays["lefts"]
        rights = arrays["rights"]
        values = arrays["values"]
        probs = arrays["probs"]

        def _build(node_id: int) -> TreeNode:
            if features[node_id] == -1:
                return TreeNode(value=int(values[node_id]), probabilities=probs[node_id])
            left_node = _build(int(lefts[node_id]))
            right_node = _build(int(rights[node_id]))
            return TreeNode(
                feature_index=int(features[node_id]),
                threshold=float(thresholds[node_id]),
                left=left_node,
                right=right_node
            )

        tree.root = _build(0)
        return tree
