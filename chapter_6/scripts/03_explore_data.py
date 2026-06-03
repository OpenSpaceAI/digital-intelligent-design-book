import _bootstrap  # noqa: F401
import pandas as pd

from deepspace_temperature.config import FIGURE_DIR, RAW_DATA_PATH, ensure_project_dirs
from deepspace_temperature.evaluation import plot_correlation_heatmap


def main() -> None:
    ensure_project_dirs()
    data = pd.read_csv(RAW_DATA_PATH)
    output_path = FIGURE_DIR / "correlation_heatmap.png"
    plot_correlation_heatmap(data.dropna(), output_path)
    print(f"Saved correlation heatmap: {output_path}")


if __name__ == "__main__":
    main()
