import _bootstrap  # noqa: F401
import joblib
import pandas as pd

from deepspace_temperature.config import FIGURE_DIR, METRICS_PATH, MODEL_PATH, TEST_DATA_PATH
from deepspace_temperature.evaluation import (
    plot_actual_vs_predicted,
    regression_metrics,
    save_metrics,
)
from deepspace_temperature.modeling import split_features_target


def main() -> None:
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    x_test, y_test = split_features_target(test_df)
    predictions = model.predict(x_test)

    metrics = regression_metrics(y_test, predictions)
    save_metrics(metrics, METRICS_PATH)
    plot_actual_vs_predicted(y_test, predictions, FIGURE_DIR / "actual_vs_predicted.png")
    print(metrics)


if __name__ == "__main__":
    main()
