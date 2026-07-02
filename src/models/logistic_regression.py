"""
src/models/logistic_regression.py

Pure Classical ML Ensemble - Multinomial (Softmax) Logistic Regression
Author: Senior Machine Learning Systems Architect

Strict Engineering Rules:
- NO SCIKIT-LEARN imported or used.
- Pure NumPy mathematical operations.
- Pure NumPy serialization (save/load using savez_compressed).
"""

import os
from typing import Dict, List, Tuple
import numpy as np


class MultinomialLogisticRegression:
    """
    From-scratch Multinomial Logistic Regression (Softmax Regression) for multi-class classification.
    """

    def __init__(self, num_classes: int = 26, learning_rate: float = 0.05, num_iterations: int = 1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        
        self.weights: np.ndarray = np.array([])
        self.bias: np.ndarray = np.array([])
        self.loss_history: List[float] = []

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        max_logits = np.max(z, axis=1, keepdims=True)
        exp_shifted = np.exp(z - max_logits)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / sum_exp

    @staticmethod
    def _one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
        N = y.shape[0]
        one_hot = np.zeros((N, num_classes), dtype=np.float64)
        one_hot[np.arange(N), y] = 1.0
        return one_hot

    def _compute_cross_entropy_loss(self, y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, eps: float = 1e-15) -> float:
        N = y_true_onehot.shape[0]
        clipped_probs = np.clip(y_pred_probs, eps, 1.0 - eps)
        log_probs = np.log(clipped_probs)
        loss = -np.sum(y_true_onehot * log_probs) / float(N)
        return float(loss)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int = None, verbose: bool = False) -> "MultinomialLogisticRegression":
        N, D = X_train.shape
        rng = np.random.default_rng(seed=42)
        limit = np.sqrt(6.0 / (D + self.num_classes))
        self.weights = rng.uniform(-limit, limit, size=(D, self.num_classes))
        self.bias = np.zeros((self.num_classes,), dtype=np.float64)
        
        Y_onehot = self._one_hot_encode(y_train, self.num_classes)
        self.loss_history = []
        effective_batch_size = N if batch_size is None else min(batch_size, N)

        for epoch in range(self.num_iterations):
            if batch_size is None:
                X_batch = X_train
                Y_batch = Y_onehot
            else:
                indices = rng.choice(N, size=effective_batch_size, replace=False)
                X_batch = X_train[indices]
                Y_batch = Y_onehot[indices]

            M = X_batch.shape[0]
            logits = np.dot(X_batch, self.weights) + self.bias
            probs = self._softmax(logits)
            error = probs - Y_batch
            
            grad_weights = np.dot(X_batch.T, error) / float(M)
            grad_bias = np.sum(error, axis=0) / float(M)
            
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
            
            if epoch % 50 == 0 or epoch == self.num_iterations - 1:
                full_logits = np.dot(X_train, self.weights) + self.bias
                full_probs = self._softmax(full_logits)
                current_loss = self._compute_cross_entropy_loss(Y_onehot, full_probs)
                self.loss_history.append(current_loss)
                if verbose and (epoch % 200 == 0 or epoch == self.num_iterations - 1):
                    print(f"[Softmax Regression] Epoch {epoch:4d}/{self.num_iterations} | Loss: {current_loss:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights.size == 0:
            raise RuntimeError("Model parameters untrained. Call fit() before predicting.")

        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else np.asarray(X, dtype=np.float64)
        logits = np.dot(X_mat, self.weights) + self.bias
        probs = self._softmax(logits)

        return probs[0] if is_1d else probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return int(np.argmax(probs))
        return np.argmax(probs, axis=1)

    def save(self, filepath: str) -> None:
        """
        Serializes Softmax regression weights and bias to disk using pure NumPy savez_compressed.
        """
        if self.weights.size == 0:
            raise RuntimeError("Cannot save untrained Logistic Regression model.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            weights=self.weights,
            bias=self.bias,
            num_classes=np.array([self.num_classes], dtype=np.int64)
        )

    def load(self, filepath: str) -> "MultinomialLogisticRegression":
        """
        Loads serialized weights and bias directly from disk without invoking fit().
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Serialized Logistic Regression file not found: {filepath}")
        data = np.load(filepath)
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.num_classes = int(data["num_classes"][0])
        return self
