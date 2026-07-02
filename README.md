# Real-Time Hand Gesture Recognition: Pure Classical ML Ensemble

An edge-optimized, real-time hand sign recognition pipeline built entirely from scratch in pure Python and NumPy. This project eschews heavy black-box libraries like `scikit-learn` and `pandas` in favor of custom, mathematically rigorous implementations of classical Machine Learning algorithms. 

Designed for ultra-low latency inference, the system achieves **sub-2ms processing time per frame** and compresses the entire multi-model ensemble state to **< 700 KB** for embedded/edge deployment.

## 🚀 Architectural Highlights

* **Dependency-Free Core ML:** Custom implementations of K-Nearest Neighbors, Multinomial Logistic Regression, and a Random Forest (CART) Classifier using pure vectorized NumPy.
* **Zero-Allocation Inference Loop:** Completely eliminated Pandas DataFrames from the OpenCV real-time capture loop, utilizing lightweight 2D NumPy arrays to prevent memory fragmentation and garbage collection spikes.
* **$O(N)$ Partial Sorting:** Optimized the K-Nearest Neighbors distance matrix evaluation using `np.argpartition` for $O(N)$ time complexity, bypassing the standard $O(N \log N)$ sorting bottlenecks.
* **Contiguous Array Serialization:** Engineered a custom tree-flattening algorithm to serialize pointer-based Random Forest hierarchies into parallel 1D arrays, resulting in a microscopic footprint (`np.savez_compressed`).

## 🗂️ Directory Structure

```text
Hand_sign_recognition/
├── src/
│   ├── main.py                     # CLI Router (argparse)
│   ├── pipeline/
│   │   ├── preprocess.py           # Z-score & spatial anchor normalization
│   │   └── inference_loop.py       # Live OpenCV webcam & majority voting engine
│   └── models/
│       ├── knn.py                  # Custom K-Nearest Neighbors
│       ├── logistic_regression.py  # Custom Softmax Regression (Gradient Descent)
│       ├── decision_tree.py        # Custom CART recursive splitter
│       └── random_forest.py        # Bootstrap Aggregation (Bagging) wrapper
├── models_bin/                     # Serialized .npz binary weights (< 700 KB total)
└── hand_landmarks_final.csv        # Extracted MediaPipe 3D coordinate dataset
```

## 🧮 Mathematical Foundations

To ensure maximum performance and demonstrate algorithmic competency, all core ML mechanics were manually derived and implemented:

### 1. Multinomial Logistic Regression (Softmax)
Optimized via explicit Gradient Descent, utilizing numerical stability shifts and analytical Cross-Entropy gradients:
$$\nabla_{W} L = \frac{1}{M} X^T (\hat{Y} - Y_{onehot})$$

### 2. Random Forest & CART Splitting
Binary recursive splits are evaluated exhaustively across random feature subspaces to maximize Gini Impurity reduction:
$$G = 1 - \sum_{k=0}^{K-1} p_k^2$$

### 3. Vectorized Euclidean Distance Matrices (KNN)
Pairwise distances are computed without nested loops using algebraic expansion:
$$\|X - Y\|^2 = \|X\|^2 + \|Y\|^2 - 2(X \cdot Y^T)$$

## ⚙️ Installation & Usage

### Prerequisites
* `numpy`
* `opencv-python`
* `mediapipe`

### 1. Offline Training Mode
Ingests the raw 63-dimensional coordinate CSV, normalizes the spatial anatomy relative to the wrist anchor, fits all three models, and serializes the state to the `models_bin/` directory.

```bash
python -m src.main --mode train
```

### 2. Real-Time Inference Mode
Bypasses training, instantly loads the highly compressed binary weights (~85ms cold start), and launches the OpenCV webcam loop. Features on-screen latency benchmarking and soft-probability Majority Voting.

```bash
python -m src.main --mode infer --camera 0
```

## 📊 Performance Benchmarks

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Inference Latency** | ~1.8 ms / frame | Measured via `time.perf_counter()` post-capture. |
| **Ensemble Disk Footprint** | 667 KB | KNN, LogReg, and 30-Tree Forest combined. |
| **Cold Start Boot Time** | 85 ms | Total time to load `np.savez_compressed` binaries. |
| **Classification Accuracy** | 98.36% | Evaluated on stratified holdout set. |
