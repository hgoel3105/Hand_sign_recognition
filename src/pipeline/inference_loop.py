"""
src/pipeline/inference_loop.py

Pure Classical ML Ensemble - Real-Time OpenCV Inference Loop
Author: Senior Machine Learning Systems Architect

Strict Engineering Rules:
- NO PANDAS or SCIKIT-LEARN imported or used anywhere in this file.
- Pure NumPy 1D/2D array allocations for real-time camera processing.
- Universal MediaPipe compatibility (Supports both legacy solutions and modern Tasks API).
- Majority Voting Ensemble across K-Nearest Neighbors, Logistic Regression, and Random Forest models.
- Exact millisecond latency benchmarking using time.perf_counter().
"""

import os
import sys
import time
import urllib.request
from typing import Dict, Tuple
import cv2
import mediapipe as mp
import numpy as np

# Ensure parent root is in Python search path to import custom src packages
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.preprocess import HandLandmarkPreprocessor
from src.models.knn import KNearestNeighbors
from src.models.logistic_regression import MultinomialLogisticRegression
from src.models.random_forest import RandomForestClassifier


class RealTimeMajorityVotingEnsemble:
    """
    Orchestrates live webcam data extraction via MediaPipe, pure NumPy spatial normalization,
    simultaneous multi-model inference, and ensemble majority vote rendering.
    """

    def __init__(self, dataset_path: str = "hand_landmarks_final.csv"):
        self.dataset_path = dataset_path
        self.preprocessor = HandLandmarkPreprocessor()
        
        # Instantiate from-scratch classical models
        self.knn = KNearestNeighbors(k=5)
        self.logreg = MultinomialLogisticRegression(num_classes=26, learning_rate=0.08, num_iterations=800)
        self.rf = RandomForestClassifier(n_estimators=30, max_depth=12, min_samples_split=2, max_features="sqrt", random_state=42)
        
        self.is_trained = False
        self.models_dir = os.path.join(project_root, "models_bin")

    def save_ensemble_to_disk(self) -> None:
        """Serializes all fitted models and scaler statistics into pure NumPy .npz archives."""
        if not self.is_trained:
            raise RuntimeError("Cannot save ensemble: models have not been trained.")
        print(f"[Serialization] Saving models to {self.models_dir}...")
        self.preprocessor.save(os.path.join(self.models_dir, "preprocessor.npz"))
        self.knn.save(os.path.join(self.models_dir, "knn_weights.npz"))
        self.logreg.save(os.path.join(self.models_dir, "logreg_weights.npz"))
        self.rf.save(os.path.join(self.models_dir, "rf_weights.npz"))
        print("[Serialization] All ensemble binaries written successfully.")

    def load_ensemble_from_disk(self) -> None:
        """Loads serialized model weights from disk directly into memory, bypassing fit()."""
        print(f"[Serialization] Loading pre-trained binaries from {self.models_dir}...")
        self.preprocessor.load(os.path.join(self.models_dir, "preprocessor.npz"))
        self.knn.load(os.path.join(self.models_dir, "knn_weights.npz"))
        self.logreg.load(os.path.join(self.models_dir, "logreg_weights.npz"))
        self.rf.load(os.path.join(self.models_dir, "rf_weights.npz"))
        self.is_trained = True
        print("[Serialization] Ensemble loaded from disk. Ready for inference.")

    def train_models_from_csv(self) -> None:
        """
        Loads dataset, applies pure NumPy normalization, and trains all three models.
        """
        print(f"[Ensemble Trainer] Ingesting CSV dataset from: {self.dataset_path}...")
        X_raw, y_raw = self.preprocessor.load_csv_dataset(self.dataset_path)
        print(f"[Ensemble Trainer] Loaded {X_raw.shape[0]} samples across {X_raw.shape[1]} features.")

        # 1. Spatial & Z-score sample normalization
        X_norm = self.preprocessor.normalize_sample_spatial(X_raw, anchor_mode="wrist")
        
        # 2. Dataset feature scaling
        X_scaled = self.preprocessor.fit_transform_dataset_scaler(X_norm)

        # Train KNN
        print("[Ensemble Trainer] Fitting K-Nearest Neighbors (k=5)...")
        t0 = time.time()
        self.knn.fit(X_scaled, y_raw)
        print(f"[Ensemble Trainer] KNN fitted in {time.time() - t0:.3f}s.")

        # Train Multinomial Logistic Regression
        print("[Ensemble Trainer] Fitting Softmax Logistic Regression...")
        t0 = time.time()
        self.logreg.fit(X_scaled, y_raw, verbose=True)
        print(f"[Ensemble Trainer] Softmax Regression fitted in {time.time() - t0:.3f}s.")

        # Train Random Forest
        print("[Ensemble Trainer] Fitting Random Forest Classifier (30 trees)...")
        t0 = time.time()
        self.rf.fit(X_scaled, y_raw, verbose=True)
        print(f"[Ensemble Trainer] Random Forest fitted in {time.time() - t0:.3f}s.")

        self.is_trained = True
        print("[Ensemble Trainer] All ensemble models operational.\n")

    def _perform_majority_vote(
        self,
        prob_knn: np.ndarray,
        prob_logreg: np.ndarray,
        prob_rf: np.ndarray
    ) -> Tuple[int, str, float, Dict[str, str]]:
        """
        Executes Majority Voting across the three models:
        1. Hard Vote: Counts discrete predictions.
        2. Soft Vote Tie-Breaker: Averages probability distributions if hard votes tie.
        """
        pred_knn = int(np.argmax(prob_knn))
        pred_logreg = int(np.argmax(prob_logreg))
        pred_rf = int(np.argmax(prob_rf))

        model_preds = {
            "KNN": self.preprocessor.index_to_label[pred_knn],
            "LogReg": self.preprocessor.index_to_label[pred_logreg],
            "RF": self.preprocessor.index_to_label[pred_rf]
        }

        votes = np.bincount([pred_knn, pred_logreg, pred_rf], minlength=26)
        max_votes = np.max(votes)

        if max_votes >= 2:
            winning_class_id = int(np.argmax(votes))
            confidence = float(max_votes) / 3.0
        else:
            avg_probs = (prob_knn + prob_logreg + prob_rf) / 3.0
            winning_class_id = int(np.argmax(avg_probs))
            confidence = float(avg_probs[winning_class_id])

        winning_label = self.preprocessor.index_to_label[winning_class_id]
        return winning_class_id, winning_label, confidence, model_preds

    def run_live_camera_loop(self, camera_index: int = 0) -> None:
        """
        Launches real-time OpenCV webcam capture, MediaPipe tracking, and ensemble inference.
        STRICT: Absolutely zero Pandas objects created in this loop.
        Includes time.perf_counter() precision latency benchmarking.
        """
        if not self.is_trained:
            # If binaries exist on disk, prefer loading them directly
            if os.path.exists(os.path.join(self.models_dir, "rf_weights.npz")):
                self.load_ensemble_from_disk()
            else:
                self.train_models_from_csv()

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[Error] Could not initialize webcam at camera index {camera_index}.")
            return

        print("[Inference Engine] Webcam active. Press 'q' inside video window to terminate.")

        has_legacy_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

        if has_legacy_solutions:
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
            use_modern_tasks = False
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            
            model_path = os.path.join(project_root, "hand_landmarker.task")
            if not os.path.exists(model_path):
                print("[MediaPipe] Fetching lightweight hand tracking asset (hand_landmarker.task)...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                print("[MediaPipe] Asset retrieved successfully.")

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
            detector = vision.HandLandmarker.create_from_options(options)
            mp_drawing = vision.drawing_utils
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
            use_modern_tasks = True

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[Warning] Frame capture failed.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                landmarks_list = []
                if not use_modern_tasks:
                    results = detector.process(frame_rgb)
                    if results.multi_hand_landmarks:
                        for hl in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                            landmarks_list.append(hl.landmark)
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    results = detector.detect(mp_image)
                    if results.hand_landmarks:
                        for hl in results.hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hl, connections)
                            landmarks_list.append(hl)

                if landmarks_list:
                    for hand_lm in landmarks_list:
                        # Start precision inference latency benchmark
                        t_start = time.perf_counter()

                        # PURE NUMPY ALLOCATION: Fast list comprehension to 1D flat float array (63,)
                        raw_coords = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand_lm],
                            dtype=np.float64
                        ).flatten()

                        # Spatial normalization & scaling
                        norm_coords = self.preprocessor.normalize_sample_spatial(raw_coords, anchor_mode="wrist")
                        scaled_coords = self.preprocessor.transform_dataset_scaler(norm_coords)

                        # Simultaneous multi-model inference
                        prob_knn = self.knn.predict_proba(scaled_coords)
                        prob_logreg = self.logreg.predict_proba(scaled_coords)
                        prob_rf = self.rf.predict_proba(scaled_coords)

                        # Majority Voting Ensemble
                        _, win_label, conf, ind_preds = self._perform_majority_vote(
                            prob_knn, prob_logreg, prob_rf
                        )

                        # End precision inference latency benchmark
                        t_end = time.perf_counter()
                        latency_ms = (t_end - t_start) * 1000.0

                        # Render dashboard HUD overlay with precise latency
                        overlay_text = f"VOTE: {win_label} ({conf*100:.1f}%) | Latency: {latency_ms:.2f}ms"
                        detail_text = f"KNN:{ind_preds['KNN']} | LogReg:{ind_preds['LogReg']} | RF:{ind_preds['RF']}"

                        cv2.rectangle(frame, (10, 10), (460, 95), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame, overlay_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, detail_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow("Pure Classical ML Ensemble - Hand Sign Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            if use_modern_tasks and hasattr(detector, "close"):
                detector.close()

        cap.release()
        cv2.destroyAllWindows()
        print("[Inference Engine] Camera loop terminated safely.")


if __name__ == "__main__":
    csv_file = os.path.join(project_root, "hand_landmarks_final.csv")
    ensemble = RealTimeMajorityVotingEnsemble(dataset_path=csv_file)
    ensemble.run_live_camera_loop(camera_index=0)
