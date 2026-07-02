import numpy as np


class TreeNode:
    """Represents a single decision node or terminal leaf within a CART tree."""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, probabilities=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probabilities = probabilities

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    """Binary decision tree classifier utilizing Gini impurity reduction."""

    def __init__(self, max_depth=12, min_samples_split=2, min_impurity_decrease=1e-7, max_features=None, random_state=None):
        self.max_depth = max_depth if max_depth is not None else 9999
        self.min_samples_split = max(2, min_samples_split)
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.num_classes = 0
        self.rng = np.random.default_rng(seed=self.random_state)

    @staticmethod
    def _compute_gini(y: np.ndarray) -> float:
        # G = 1 - sum(p_k^2)
        n = y.size
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / float(n)
        return float(1.0 - np.sum(np.square(probs)))

    def _get_subspace_size(self, total_features: int) -> int:
        if self.max_features is None:
            return total_features
        if isinstance(self.max_features, str):
            if self.max_features.lower() == "sqrt":
                return max(1, int(np.sqrt(total_features)))
            elif self.max_features.lower() == "log2":
                return max(1, int(np.log2(total_features)))
        elif isinstance(self.max_features, float):
            return max(1, min(total_features, int(self.max_features * total_features)))
        elif isinstance(self.max_features, int):
            return max(1, min(total_features, self.max_features))
        return total_features

    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        parent_gini = self._compute_gini(y)
        best_gain = 0.0
        best_feat, best_thresh = None, None

        n_sub = self._get_subspace_size(n_features)
        feat_indices = self.rng.choice(n_features, size=n_sub, replace=False)

        for feat_idx in feat_indices:
            values = X[:, feat_idx]
            uniques = np.unique(values)
            if uniques.size <= 1:
                continue

            # Quantile discretization to keep tree induction fast on continuous data
            if uniques.size > 25:
                thresholds = np.unique(np.percentile(values, np.linspace(4, 96, 25)))
            else:
                thresholds = (uniques[:-1] + uniques[1:]) / 2.0

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                n_l, n_r = int(np.sum(left_mask)), n_samples - int(np.sum(left_mask))
                if n_l == 0 or n_r == 0:
                    continue

                g_left = self._compute_gini(y[left_mask])
                g_right = self._compute_gini(y[right_mask])
                weighted_gini = (n_l / float(n_samples)) * g_left + (n_r / float(n_samples)) * g_right

                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = float(thresh)

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n = y.size
        counts = np.bincount(y, minlength=self.num_classes)
        leaf_probs = counts / float(n) if n > 0 else np.zeros((self.num_classes,), dtype=np.float64)
        leaf_val = int(np.argmax(counts))

        if depth >= self.max_depth or n < self.min_samples_split or np.unique(y).size == 1:
            return TreeNode(value=leaf_val, probabilities=leaf_probs)

        best_feat, best_thresh, best_gain = self._find_best_split(X, y)

        if best_feat is None or best_gain < self.min_impurity_decrease:
            return TreeNode(value=leaf_val, probabilities=leaf_probs)

        left_mask = X[:, best_feat] <= best_thresh
        left_sub = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_sub = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return TreeNode(feature_index=best_feat, threshold=best_thresh, left=left_sub, right=right_sub)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, num_classes=None):
        """Recursively induces the binary decision tree hierarchy."""
        self.num_classes = num_classes if num_classes is not None else int(np.max(y_train)) + 1
        self.root = self._build_tree(X_train, y_train, depth=0)
        return self

    def _predict_row(self, node: TreeNode, x: np.ndarray):
        if node.is_leaf:
            return node.value, node.probabilities
        if x[node.feature_index] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Decision tree not fitted.")

        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else X
        m = X_mat.shape[0]

        probs = np.zeros((m, self.num_classes), dtype=np.float64)
        for i in range(m):
            _, probs[i] = self._predict_row(self.root, X_mat[i])

        return probs[0] if is_1d else probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return int(np.argmax(probs)) if probs.ndim == 1 else np.argmax(probs, axis=1)

    def _flatten_node(self, node: TreeNode, nodes: list) -> int:
        node_id = len(nodes)
        nodes.append({})
        if node.is_leaf:
            nodes[node_id] = {
                "feat": -1, "thresh": 0.0, "left": -1, "right": -1,
                "val": node.value,
                "probs": node.probabilities if node.probabilities is not None else np.zeros((self.num_classes,))
            }
        else:
            left_id = self._flatten_node(node.left, nodes)
            right_id = self._flatten_node(node.right, nodes)
            nodes[node_id] = {
                "feat": node.feature_index, "thresh": node.threshold,
                "left": left_id, "right": right_id,
                "val": -1, "probs": np.zeros((self.num_classes,))
            }
        return node_id

    def to_arrays(self) -> dict:
        """Flattens the tree into contiguous 1D arrays for serialization."""
        if self.root is None:
            raise RuntimeError("Cannot serialize unfitted tree.")
        nodes = []
        self._flatten_node(self.root, nodes)

        return {
            "features": np.array([n["feat"] for n in nodes], dtype=np.int64),
            "thresholds": np.array([n["thresh"] for n in nodes], dtype=np.float64),
            "lefts": np.array([n["left"] for n in nodes], dtype=np.int64),
            "rights": np.array([n["right"] for n in nodes], dtype=np.int64),
            "values": np.array([n["val"] for n in nodes], dtype=np.int64),
            "probs": np.array([n["probs"] for n in nodes], dtype=np.float64)
        }

    @classmethod
    def from_arrays(cls, arrays: dict, num_classes: int):
        """Reconstructs tree structure from contiguous arrays."""
        tree = cls()
        tree.num_classes = num_classes

        feats = arrays["features"]
        threshs = arrays["thresholds"]
        lefts = arrays["lefts"]
        rights = arrays["rights"]
        vals = arrays["values"]
        probs = arrays["probs"]

        def _build(nid: int) -> TreeNode:
            if feats[nid] == -1:
                return TreeNode(value=int(vals[nid]), probabilities=probs[nid])
            return TreeNode(
                feature_index=int(feats[nid]),
                threshold=float(threshs[nid]),
                left=_build(int(lefts[nid])),
                right=_build(int(rights[nid]))
            )

        tree.root = _build(0)
        return tree
