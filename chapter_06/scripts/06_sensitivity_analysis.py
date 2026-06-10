import _bootstrap  # noqa: F401
import json

import joblib
import numpy as np
import pandas as pd

from deepspace_temperature.config import FEATURE_COLUMNS, MODEL_PATH, REPORT_DIR, TEST_DATA_PATH


def main() -> None:
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    baseline = test_df[FEATURE_COLUMNS].median().to_frame().T

    results = {}
    for feature in FEATURE_COLUMNS:
        grid = np.linspace(test_df[feature].quantile(0.05), test_df[feature].quantile(0.95), 30)
        samples = pd.concat([baseline] * len(grid), ignore_index=True)
        samples[feature] = grid
        predictions = model.predict(samples)
        results[feature] = {
            "min_prediction": float(np.min(predictions)),
            "max_prediction": float(np.max(predictions)),
            "range": float(np.max(predictions) - np.min(predictions)),
        }

    output_path = REPORT_DIR / "sensitivity.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved sensitivity report: {output_path}")


if __name__ == "__main__":
    main()
