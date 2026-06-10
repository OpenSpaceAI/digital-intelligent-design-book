import _bootstrap  # noqa: F401
import joblib
import pandas as pd

from deepspace_temperature.config import (
    PREPROCESS_BUNDLE_PATH,
    RAW_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    ensure_project_dirs,
)
from deepspace_temperature.preprocessing import preprocess_dataframe


def main() -> None:
    ensure_project_dirs()
    data = pd.read_csv(RAW_DATA_PATH)
    train_df, test_df, bundle = preprocess_dataframe(data)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    joblib.dump(bundle, PREPROCESS_BUNDLE_PATH)
    print(f"Saved train data: {TRAIN_DATA_PATH} ({train_df.shape[0]} rows)")
    print(f"Saved test data: {TEST_DATA_PATH} ({test_df.shape[0]} rows)")


if __name__ == "__main__":
    main()
