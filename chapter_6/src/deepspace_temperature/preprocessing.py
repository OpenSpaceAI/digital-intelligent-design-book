from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class PreprocessBundle:
    imputer: SimpleImputer
    scaler: StandardScaler
    feature_columns: list[str]
    target_column: str


def remove_outliers_3sigma(
    data: pd.DataFrame,
    columns: list[str],
    sigma: float = 3.0,
) -> pd.DataFrame:
    numeric = data[columns]
    mean = numeric.mean()
    std = numeric.std(ddof=0).replace(0, 1)
    mask = ((numeric - mean).abs() <= sigma * std).all(axis=1)
    return data.loc[mask].reset_index(drop=True)


def preprocess_dataframe(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessBundle]:
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[required].copy()

    imputer = SimpleImputer(strategy="mean")
    data[FEATURE_COLUMNS] = imputer.fit_transform(data[FEATURE_COLUMNS])
    data = remove_outliers_3sigma(data, required)

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    bundle = PreprocessBundle(
        imputer=imputer,
        scaler=scaler,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), bundle
