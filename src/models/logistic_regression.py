import os
import numpy as np


class MultinomialLogisticRegression:
    """Multi-class logistic regression trained via analytical cross-entropy gradient descent."""

    def __init__(self, num_classes: int = 26, learning_rate: float = 0.05, num_iterations: int = 1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = np.array([])
        self.bias = np.array([])
        self.loss_history = []

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability inside exponent
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
        n = y.shape[0]
        oh = np.zeros((n, num_classes), dtype=np.float64)
        oh[np.arange(n), y] = 1.0
        return oh

    def _cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
        clipped = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.sum(y_true * np.log(clipped)) / y_true.shape[0])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int = None, verbose: bool = False):
        """Optimizes weights and bias using gradient descent."""
        n_samples, n_features = X_train.shape
        rng = np.random.default_rng(seed=42)
        
        # Xavier-style initialization
        limit = np.sqrt(6.0 / (n_features + self.num_classes))
        self.weights = rng.uniform(-limit, limit, size=(n_features, self.num_classes))
        self.bias = np.zeros((self.num_classes,), dtype=np.float64)
        
        y_oh = self._one_hot(y_train, self.num_classes)
        self.loss_history = []
        bs = n_samples if batch_size is None else min(batch_size, n_samples)

        for epoch in range(self.num_iterations):
            if batch_size is None:
                X_b, y_b = X_train, y_oh
            else:
                idx = rng.choice(n_samples, size=bs, replace=False)
                X_b, y_b = X_train[idx], y_oh[idx]

            m = X_b.shape[0]
            logits = np.dot(X_b, self.weights) + self.bias
            probs = self._softmax(logits)
            
            # dL/dW = X^T * (probs - Y_onehot) / m
            err = probs - y_b
            grad_w = np.dot(X_b.T, err) / float(m)
            grad_b = np.sum(err, axis=0) / float(m)
            
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            
            if epoch % 50 == 0 or epoch == self.num_iterations - 1:
                full_probs = self._softmax(np.dot(X_train, self.weights) + self.bias)
                loss = self._cross_entropy_loss(y_oh, full_probs)
                self.loss_history.append(loss)
                if verbose and (epoch % 200 == 0 or epoch == self.num_iterations - 1):
                    print(f"Epoch {epoch:4d}/{self.num_iterations} | Loss: {loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights.size == 0:
            raise RuntimeError("Model parameters not fitted.")

        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else X
        probs = self._softmax(np.dot(X_mat, self.weights) + self.bias)
        return probs[0] if is_1d else probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return int(np.argmax(probs)) if probs.ndim == 1 else np.argmax(probs, axis=1)

    def save(self, filepath: str) -> None:
        """Saves model weights and bias vectors to disk."""
        if self.weights.size == 0:
            raise RuntimeError("Cannot save unfitted model.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            weights=self.weights,
            bias=self.bias,
            num_classes=np.array([self.num_classes], dtype=np.int64)
        )

    def load(self, filepath: str):
        """Loads pre-trained parameter matrices from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing model binary: {filepath}")
        data = np.load(filepath)
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.num_classes = int(data["num_classes"][0])
        return self
