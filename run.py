"""
run.py

Cross-Platform Universal Bootstrap Launcher
Author: Senior Machine Learning Systems Architect

Automatically diagnoses and establishes virtual environments, verifies required dependencies,
and executes the clean CLI router without requiring manual environment activation.
"""

import os
import subprocess
import sys

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "mediapipe": "mediapipe"
}


def check_and_bootstrap_environment() -> str:
    """
    Verifies runtime environment and establishes isolated venv if necessary.
    Returns path to valid Python executable inside venv.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(project_root, "venv")
    
    if os.name == "nt":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    missing = []
    for module_name in REQUIRED_PACKAGES.keys():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if not missing and sys.prefix != sys.base_prefix:
        return sys.executable

    if not os.path.exists(venv_python):
        print("[Bootstrap] Creating isolated `venv` environment inside project directory...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)

    print("[Bootstrap] Verifying and installing dependencies into `venv` (this runs once)...")
    packages_to_install = list(REQUIRED_PACKAGES.values())
    subprocess.run(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip"] + packages_to_install,
        check=True
    )
    print("[Bootstrap] All packages installed and verified.\n")
    return venv_python


def main():
    venv_python = check_and_bootstrap_environment()

    # Pass CLI arguments forward to src/main.py, defaulting to `--mode infer`
    cli_args = sys.argv[1:]
    if not cli_args:
        models_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_bin")
        if not os.path.exists(os.path.join(models_bin, "rf_weights.npz")):
            cli_args = ["--mode", "train"]
        else:
            cli_args = ["--mode", "infer"]

    print(f"[Bootstrap] Re-spawning router inside isolated `venv`: `python -m src.main {' '.join(cli_args)}`...\n")
    cmd = [venv_python, "-m", "src.main"] + cli_args
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
