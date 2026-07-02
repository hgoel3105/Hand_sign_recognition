import os
import sys
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

# Ensure root path is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.preprocess import HandLandmarkPreprocessor
from src.models.knn import KNearestNeighbors
from src.models.logistic_regression import MultinomialLogisticRegression
from src.models.random_forest import RandomForestClassifier


class RealTimeMajorityVotingEnsemble:
    """Manages webcam capture, landmark normalization, and ensemble inference."""

    def __init__(self, dataset_path: str = "hand_landmarks_final.csv"):
        self.dataset_path = dataset_path
        self.preprocessor = HandLandmarkPreprocessor()
        self.knn = KNearestNeighbors(k=5)
        self.logreg = MultinomialLogisticRegression(num_classes=26, learning_rate=0.08, num_iterations=800)
        self.rf = RandomForestClassifier(n_estimators=30, max_depth=12, min_samples_split=2, max_features="sqrt", random_state=42)
        self.is_trained = False
        self.models_dir = os.path.join(project_root, "models_bin")

    def save_ensemble_to_disk(self):
        """Writes trained model parameters to compressed binary archives."""
        if not self.is_trained:
            raise RuntimeError("Models must be trained before saving.")
        print(f"Saving model binaries to {self.models_dir}...")
        self.preprocessor.save(os.path.join(self.models_dir, "preprocessor.npz"))
        self.knn.save(os.path.join(self.models_dir, "knn_weights.npz"))
        self.logreg.save(os.path.join(self.models_dir, "logreg_weights.npz"))
        self.rf.save(os.path.join(self.models_dir, "rf_weights.npz"))
        print("Binaries written successfully.")

    def load_ensemble_from_disk(self):
        """Loads pre-trained model weights from disk."""
        print(f"Loading pre-trained binaries from {self.models_dir}...")
        self.preprocessor.load(os.path.join(self.models_dir, "preprocessor.npz"))
        self.knn.load(os.path.join(self.models_dir, "knn_weights.npz"))
        self.logreg.load(os.path.join(self.models_dir, "logreg_weights.npz"))
        self.rf.load(os.path.join(self.models_dir, "rf_weights.npz"))
        self.is_trained = True
        print("Ensemble loaded.")

    def train_models_from_csv(self):
        """Fits all three classifiers on the raw coordinate dataset."""
        print(f"Loading dataset: {self.dataset_path}...")
        X_raw, y_raw = self.preprocessor.load_csv_dataset(self.dataset_path)
        print(f"Loaded {X_raw.shape[0]} samples across {X_raw.shape[1]} features.")

        X_norm = self.preprocessor.normalize_sample_spatial(X_raw, anchor_mode="wrist")
        X_scaled = self.preprocessor.fit_transform_dataset_scaler(X_norm)

        print("Training KNN...")
        t0 = time.time()
        self.knn.fit(X_scaled, y_raw)
        print(f"KNN fitted in {time.time() - t0:.3f}s.")

        print("Training Logistic Regression...")
        t0 = time.time()
        self.logreg.fit(X_scaled, y_raw, verbose=True)
        print(f"Logistic Regression fitted in {time.time() - t0:.3f}s.")

        print("Training Random Forest...")
        t0 = time.time()
        self.rf.fit(X_scaled, y_raw, verbose=True)
        print(f"Random Forest fitted in {time.time() - t0:.3f}s.")

        self.is_trained = True
        print("Training complete.\n")

    def _perform_majority_vote(self, prob_knn: np.ndarray, prob_logreg: np.ndarray, prob_rf: np.ndarray):
        """Combines model outputs using hard voting with soft probability tie-breaking."""
        pred_knn = int(np.argmax(prob_knn))
        pred_logreg = int(np.argmax(prob_logreg))
        pred_rf = int(np.argmax(prob_rf))

        preds = {
            "KNN": self.preprocessor.index_to_label[pred_knn],
            "LogReg": self.preprocessor.index_to_label[pred_logreg],
            "RF": self.preprocessor.index_to_label[pred_rf]
        }

        votes = np.bincount([pred_knn, pred_logreg, pred_rf], minlength=26)
        max_v = np.max(votes)

        if max_v >= 2:
            win_id = int(np.argmax(votes))
            conf = float(max_v) / 3.0
        else:
            avg_probs = (prob_knn + prob_logreg + prob_rf) / 3.0
            win_id = int(np.argmax(avg_probs))
            conf = float(avg_probs[win_id])

        return win_id, self.preprocessor.index_to_label[win_id], conf, preds

    def run_live_camera_loop(self, camera_index: int = 0):
        """Runs video capture and frame classification loop."""
        if not self.is_trained:
            if os.path.exists(os.path.join(self.models_dir, "rf_weights.npz")):
                self.load_ensemble_from_disk()
            else:
                self.train_models_from_csv()

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera at index {camera_index}.")
            return

        print("Webcam active. Press 'q' to exit.")

        # Check for legacy MediaPipe python API vs modern Tasks API
        has_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

        if has_legacy:
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            )
            use_tasks = False
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            
            model_path = os.path.join(project_root, "hand_landmarker.task")
            if not os.path.exists(model_path):
                print("Downloading hand_landmarker.task model asset...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)

            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = vision.HandLandmarkerOptions(base_options=base_opts, num_hands=1)
            detector = vision.HandLandmarker.create_from_options(opts)
            mp_drawing = vision.drawing_utils
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
            use_tasks = True

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                landmarks = []
                if not use_tasks:
                    res = detector.process(rgb)
                    if res.multi_hand_landmarks:
                        for hl in res.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                            landmarks.append(hl.landmark)
                else:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = detector.detect(mp_img)
                    if res.hand_landmarks:
                        for hl in res.hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hl, connections)
                            landmarks.append(hl)

                if landmarks:
                    for lm in landmarks:
                        t_start = time.perf_counter()

                        raw = np.array([[point.x, point.y, point.z] for point in lm], dtype=np.float64).flatten()
                        normed = self.preprocessor.normalize_sample_spatial(raw, anchor_mode="wrist")
                        scaled = self.preprocessor.transform_dataset_scaler(normed)

                        p_knn = self.knn.predict_proba(scaled)
                        p_log = self.logreg.predict_proba(scaled)
                        p_rf = self.rf.predict_proba(scaled)

                        _, label, conf, preds = self._perform_majority_vote(p_knn, p_log, p_rf)

                        latency_ms = (time.perf_counter() - t_start) * 1000.0

                        overlay = f"VOTE: {label} ({conf*100:.1f}%) | Latency: {latency_ms:.2f}ms"
                        details = f"KNN:{preds['KNN']} | LogReg:{preds['LogReg']} | RF:{preds['RF']}"

                        cv2.rectangle(frame, (10, 10), (460, 95), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame, overlay, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, details, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow("ASL Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            if use_tasks and hasattr(detector, "close"):
                detector.close()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    csv_file = os.path.join(project_root, "hand_landmarks_final.csv")
    ensemble = RealTimeMajorityVotingEnsemble(dataset_path=csv_file)
    ensemble.run_live_camera_loop(camera_index=0)
