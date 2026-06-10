import _bootstrap  # noqa: F401
import json

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from deepspace_temperature.config import FEATURE_COLUMNS, REPORT_DIR, TARGET_COLUMN, TRAIN_DATA_PATH


def main() -> None:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    search = GridSearchCV(
        estimator=MLPRegressor(
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
        param_grid={
            "hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [5e-4, 1e-3],
        },
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)

    result = {
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
    }
    output_path = REPORT_DIR / "hyperparameter_search.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
