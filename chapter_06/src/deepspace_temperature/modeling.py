import pandas as pd
from sklearn.neural_network import MLPRegressor

from .config import FEATURE_COLUMNS, TARGET_COLUMN


def build_mlp_model(random_state: int = 42) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=random_state,
    )


def split_features_target(data: pd.DataFrame):
    return data[FEATURE_COLUMNS], data[TARGET_COLUMN]


def train_model(train_df: pd.DataFrame, random_state: int = 42) -> MLPRegressor:
    x_train, y_train = split_features_target(train_df)
    model = build_mlp_model(random_state=random_state)
    model.fit(x_train, y_train)
    return model
