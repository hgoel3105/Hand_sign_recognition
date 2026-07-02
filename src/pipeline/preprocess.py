"""
src/pipeline/preprocess.py

Pure Classical ML Pipeline - Preprocessing Module
Author: Senior Machine Learning Systems Architect

Strict Engineering Rules:
- NO SCIKIT-LEARN or PANDAS imported or used.
- Pure NumPy mathematical operations for data ingestion, normalization, and splits.
- Pure NumPy serialization (save/load) for feature scaler statistics.
"""

import csv
import os
from typing import Dict, List, Tuple, Union
import numpy as np


class HandLandmarkPreprocessor:
    """
    Handles CSV ingestion, label encoding/decoding, spatial anchor normalization,
    and Z-score feature scaling using pure NumPy.
    """

    def __init__(self):
        # Mapping from categorical string labels ('A'-'Z') to integer IDs (0-25)
        self.label_to_index: Dict[str, int] = {chr(65 + i): i for i in range(26)}
        self.index_to_label: Dict[int, str] = {i: chr(65 + i) for i in range(26)}
        
        # Dataset-level statistics for feature standardization across training set
        self.feature_mean: Union[np.ndarray, None] = None
        self.feature_std: Union[np.ndarray, None] = None

    def load_csv_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the landmark CSV dataset into pure NumPy arrays without Pandas.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at path: {filepath}")

        features_list: List[List[float]] = []
        labels_list: List[int] = []

        with open(filepath, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header and not header[0].replace(".", "").replace("-", "").isdigit():
                pass
            else:
                file.seek(0)
                reader = csv.reader(file)

            for row in reader:
                if not row or len(row) < 64:
                    continue
                try:
                    coords = [float(val) for val in row[:63]]
                    raw_label = row[63].strip().upper()
                    if raw_label not in self.label_to_index:
                        continue
                    features_list.append(coords)
                    labels_list.append(self.label_to_index[raw_label])
                except ValueError:
                    continue

        X = np.array(features_list, dtype=np.float64)
        y = np.array(labels_list, dtype=np.int64)
        return X, y

    @staticmethod
    def normalize_sample_spatial(X: np.ndarray, anchor_mode: str = "wrist", eps: float = 1e-8) -> np.ndarray:
        """
        Performs per-frame spatial anchoring and sample-wise Z-score normalization.
        """
        is_1d = (X.ndim == 1)
        if is_1d:
            X_mat = X.reshape(1, -1)
        else:
            X_mat = X.copy()

        N = X_mat.shape[0]
        coords_3d = X_mat.reshape(N, 21, 3)

        if anchor_mode == "wrist":
            anchor = coords_3d[:, 0:1, :]
        elif anchor_mode == "min_x":
            min_x_indices = np.argmin(coords_3d[:, :, 0], axis=1)
            anchor = coords_3d[np.arange(N), min_x_indices, :][:, np.newaxis, :]
        else:
            raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")

        coords_centered = coords_3d - anchor
        X_centered = coords_centered.reshape(N, 63)

        sample_mean = np.mean(X_centered, axis=1, keepdims=True)
        sample_std = np.std(X_centered, axis=1, keepdims=True)

        X_norm = (X_centered - sample_mean) / (sample_std + eps)

        if is_1d:
            return X_norm.flatten()
        return X_norm

    def fit_transform_dataset_scaler(self, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Fits dataset-level feature standardization and transforms the training set.
        """
        self.feature_mean = np.mean(X, axis=0, keepdims=True)
        self.feature_std = np.std(X, axis=0, keepdims=True)
        return (X - self.feature_mean) / (self.feature_std + eps)

    def transform_dataset_scaler(self, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Transforms test or live inference features using fitted dataset statistics.
        """
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Dataset scaler statistics not fitted. Call fit_transform_dataset_scaler first.")
        
        is_1d = (X.ndim == 1)
        X_mat = X.reshape(1, -1) if is_1d else X
        scaled = (X_mat - self.feature_mean) / (self.feature_std + eps)
        return scaled.flatten() if is_1d else scaled

    def save(self, filepath: str) -> None:
        """
        Serializes fitted preprocessor scaler statistics to disk using pure NumPy savez_compressed.
        """
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Cannot save preprocessor: statistics have not been fitted.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(filepath, feature_mean=self.feature_mean, feature_std=self.feature_std)

    def load(self, filepath: str) -> "HandLandmarkPreprocessor":
        """
        Loads serialized preprocessor scaler statistics from disk into memory.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Serialized preprocessor file not found: {filepath}")
        data = np.load(filepath)
        self.feature_mean = data["feature_mean"]
        self.feature_std = data["feature_std"]
        return self
