"""
src/main.py

Pure Classical ML Ensemble - Command Line Router
Author: Senior Machine Learning Systems Architect

Usage:
  python -m src.main --mode train
  python -m src.main --mode infer [--camera 0]
"""

import argparse
import os
import sys

# Ensure project root is on Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.inference_loop import RealTimeMajorityVotingEnsemble


def main():
    parser = argparse.ArgumentParser(
        description="Pure Classical ML Ensemble for Real-Time Hand Sign Recognition"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer"],
        required=True,
        help="'train' fits models on CSV and saves to models_bin/. 'infer' loads disk binaries and launches camera loop."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera device index for live inference (default: 0)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hand_landmarks_final.csv",
        help="Path to training CSV file (default: hand_landmarks_final.csv)."
    )

    args = parser.parse_args()
    dataset_path = os.path.join(project_root, args.dataset)
    ensemble = RealTimeMajorityVotingEnsemble(dataset_path=dataset_path)

    if args.mode == "train":
        print("[CLI Router] Entering OFFLINE TRAINING phase...")
        ensemble.train_models_from_csv()
        ensemble.save_ensemble_to_disk()
        print("[CLI Router] Offline training and serialization complete.")
    elif args.mode == "infer":
        print("[CLI Router] Entering REAL-TIME INFERENCE phase...")
        models_bin = os.path.join(project_root, "models_bin")
        if not os.path.exists(os.path.join(models_bin, "rf_weights.npz")):
            print("[Warning] Serialized model binaries not found. Triggering offline training first...")
            ensemble.train_models_from_csv()
            ensemble.save_ensemble_to_disk()
        else:
            ensemble.load_ensemble_from_disk()
        
        ensemble.run_live_camera_loop(camera_index=args.camera)


if __name__ == "__main__":
    main()
