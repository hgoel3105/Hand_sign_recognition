import csv
import os
import numpy as np


class HandLandmarkPreprocessor:
    """Handles CSV data loading and spatial standardization for hand coordinates."""

    def __init__(self):
        self.label_to_index = {chr(65 + i): i for i in range(26)}
        self.index_to_label = {i: chr(65 + i) for i in range(26)}
        self.feature_mean = None
        self.feature_std = None

    def load_csv_dataset(self, filepath: str):
        """Reads 63-dimensional coordinate vectors and integer class labels from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        features, labels = [], []

        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # Handle files that might not have a text header row
            if header and header[0].replace(".", "").replace("-", "").isdigit():
                f.seek(0)
                reader = csv.reader(f)

            for row in reader:
                if not row or len(row) < 64:
                    continue
                try:
                    coords = [float(x) for x in row[:63]]
                    label = row[63].strip().upper()
                    if label in self.label_to_index:
                        features.append(coords)
                        labels.append(self.label_to_index[label])
                except ValueError:
                    continue

        return np.array(features, dtype=np.float64), np.array(labels, dtype=np.int64)

    @staticmethod
    def normalize_sample_spatial(X: np.ndarray, anchor_mode: str = "wrist", eps: float = 1e-8) -> np.ndarray:
        """Translates joints relative to wrist anchor and normalizes scale."""
        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else X.copy()
        n_samples = X_mat.shape[0]
        
        coords = X_mat.reshape(n_samples, 21, 3)

        if anchor_mode == "wrist":
            anchor = coords[:, 0:1, :]
        elif anchor_mode == "min_x":
            idx = np.argmin(coords[:, :, 0], axis=1)
            anchor = coords[np.arange(n_samples), idx, :][:, np.newaxis, :]
        else:
            raise ValueError(f"Invalid anchor mode: {anchor_mode}")

        centered = (coords - anchor).reshape(n_samples, 63)
        mean = np.mean(centered, axis=1, keepdims=True)
        std = np.std(centered, axis=1, keepdims=True)

        normed = (centered - mean) / (std + eps)
        return normed.flatten() if is_1d else normed

    def fit_transform_dataset_scaler(self, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Computes feature means/stds across dataset and standardizes inputs."""
        self.feature_mean = np.mean(X, axis=0, keepdims=True)
        self.feature_std = np.std(X, axis=0, keepdims=True)
        return (X - self.feature_mean) / (self.feature_std + eps)

    def transform_dataset_scaler(self, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Applies pre-computed feature standardization to new inference frames."""
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Scaler not fitted yet.")
        
        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else X
        scaled = (X_mat - self.feature_mean) / (self.feature_std + eps)
        return scaled.flatten() if is_1d else scaled

    def save(self, filepath: str) -> None:
        """Saves fitted scaler statistics to disk."""
        if self.feature_mean is None:
            raise RuntimeError("Cannot save unfitted preprocessor.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(filepath, feature_mean=self.feature_mean, feature_std=self.feature_std)

    def load(self, filepath: str):
        """Loads scaler statistics from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing preprocessor file: {filepath}")
        data = np.load(filepath)
        self.feature_mean = data["feature_mean"]
        self.feature_std = data["feature_std"]
        return self
