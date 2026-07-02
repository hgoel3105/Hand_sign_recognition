import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline.inference_loop import RealTimeMajorityVotingEnsemble


def main():
    parser = argparse.ArgumentParser(description="ASL Recognition CLI")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True, help="Mode of operation")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for video capture")
    parser.add_argument("--dataset", type=str, default="hand_landmarks_final.csv", help="Path to landmark CSV")

    args = parser.parse_args()
    dataset_path = os.path.join(project_root, args.dataset)
    ensemble = RealTimeMajorityVotingEnsemble(dataset_path=dataset_path)

    if args.mode == "train":
        print("Starting offline model training...")
        ensemble.train_models_from_csv()
        ensemble.save_ensemble_to_disk()
    elif args.mode == "infer":
        models_bin = os.path.join(project_root, "models_bin")
        if not os.path.exists(os.path.join(models_bin, "rf_weights.npz")):
            print("Pre-trained binaries not found. Training models first...")
            ensemble.train_models_from_csv()
            ensemble.save_ensemble_to_disk()
        else:
            ensemble.load_ensemble_from_disk()
        
        ensemble.run_live_camera_loop(camera_index=args.camera)


if __name__ == "__main__":
    main()
