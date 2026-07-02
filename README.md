# Real-Time Hand Gesture Recognition: From-Scratch Classical ML Ensemble

A real-time American Sign Language (ASL) alphabet recognition system built using pure Python and NumPy. This project implements classical machine learning classifiers from scratch without relying on higher-level libraries like scikit-learn or pandas, focusing on algorithmic implementation and low-latency execution.

## System Overview

The pipeline extracts 3D hand landmark coordinates from live video frames using MediaPipe and classifies them across 26 discrete gesture classes (A-Z). To optimize real-time inference speed and memory footprint, the classification engine uses a custom majority voting ensemble consisting of K-Nearest Neighbors, Multinomial Logistic Regression, and a Random Forest classifier.

### Implementation Details

- **K-Nearest Neighbors (KNN):** Implements vectorized pairwise Euclidean distance matrices and uses `np.argpartition` for $O(N)$ partial sorting of neighbor distances.
- **Multinomial Logistic Regression:** Implements softmax activation and full cross-entropy gradient descent optimization from scratch.
- **Random Forest (CART):** Built from custom CART decision trees evaluated using Gini impurity ($G = 1 - \sum p_k^2$). Includes quantile discretization for continuous threshold search and array-based serialization.
- **Data Pipeline:** Standardizes 3D spatial coordinates relative to the wrist anchor and applies Z-score normalization directly on NumPy arrays.

## Project Structure

```text
Hand_sign_recognition/
├── src/
│   ├── main.py                     # CLI router
│   ├── pipeline/
│   │   ├── preprocess.py           # Coordinate normalization and dataset loading
│   │   └── inference_loop.py       # OpenCV capture and ensemble voting logic
│   └── models/
│       ├── knn.py                  # Vectorized K-Nearest Neighbors
│       ├── logistic_regression.py  # Softmax Regression via gradient descent
│       ├── decision_tree.py        # CART binary tree splitter
│       └── random_forest.py        # Bagging ensemble wrapper
├── models_bin/                     # Serialized model arrays (.npz)
├── hand_landmarks_final.csv        # MediaPipe landmark dataset (1212 samples)
└── run.py                          # Environment and execution launcher
```

## Mathematical Formulations

### 1. Softmax Gradient Descent
The parameter updates for the $K$-class logistic regression model are computed via the analytical gradient of the cross-entropy loss function:
$$\nabla_{W} L = \frac{1}{M} X^T (\hat{Y} - Y_{onehot})$$

### 2. Decision Tree Split Criterion
At each internal node, candidate thresholds are evaluated to maximize the reduction in Gini impurity across child partitions:
$$G = 1 - \sum_{k=0}^{K-1} p_k^2$$

### 3. Euclidean Distance Calculation
Pairwise distances between query samples and stored training vectors are computed without explicit Python loops using the identity:
$$\|X - Y\|^2 = \|X\|^2 + \|Y\|^2 - 2(X \cdot Y^T)$$

## Setup and Usage

### Requirements
Ensure Python 3.9+ is installed along with the primary dependencies:
```bash
pip install numpy opencv-python mediapipe
```

Alternatively, `run.py` can be executed directly to verify local packages automatically.

### Training the Models
To ingest `hand_landmarks_final.csv`, fit the ensemble models, and serialize weight matrices to `models_bin/`:
```bash
python -m src.main --mode train
```

### Running Real-Time Inference
To run live classification from the webcam using pre-trained binary weights:
```bash
python -m src.main --mode infer --camera 0
```

## Performance Benchmarks

Evaluated on the 1,212 sample dataset and standard desktop CPU hardware:

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Average Latency** | ~1.8 ms / frame | Measured per frame inside the OpenCV loop via `time.perf_counter()`. |
| **Model Size** | 667 KB | Combined `.npz` archive sizes across all three models. |
| **Load Time** | 85 ms | Disk deserialization time for ensemble arrays. |
| **Holdout Accuracy** | 98.36% | Evaluated on stratified test partitions. |
