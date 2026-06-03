import _bootstrap  # noqa: F401
import joblib
import pandas as pd

from deepspace_temperature.config import MODEL_PATH, TRAIN_DATA_PATH, ensure_project_dirs
from deepspace_temperature.modeling import train_model


def main() -> None:
    ensure_project_dirs()
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    model = train_model(train_df)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
