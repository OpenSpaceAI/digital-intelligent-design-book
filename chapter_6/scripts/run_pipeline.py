import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "01_generate_data.py",
    "02_preprocess_data.py",
    "03_explore_data.py",
    "04_train_model.py",
    "05_evaluate_model.py",
    "06_sensitivity_analysis.py",
]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        path = script_dir / script
        print(f"\n=== Running {script} ===")
        subprocess.run([sys.executable, str(path)], check=True)


if __name__ == "__main__":
    main()
