import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMN


def generate_temperature_data(
    n_samples: int = 1200,
    random_state: int = 42,
    missing_rate: float = 0.03,
) -> pd.DataFrame:
    """Generate a reproducible synthetic deep-space temperature dataset."""
    rng = np.random.default_rng(random_state)

    distance_au = rng.uniform(0.7, 5.2, n_samples)
    solar_flux = 1361.0 / np.square(distance_au) + rng.normal(0, 12, n_samples)
    orbital_radius = rng.uniform(1.0e5, 9.0e5, n_samples)
    albedo = rng.uniform(0.05, 0.65, n_samples)
    heater_power = rng.uniform(0, 120, n_samples)
    radiator_area = rng.uniform(0.5, 8.0, n_samples)
    instrument_load = rng.uniform(10, 260, n_samples)
    attitude_angle = rng.uniform(0, 180, n_samples)

    temperature_c = (
        -95
        + 0.055 * solar_flux * (1 - albedo)
        + 0.24 * heater_power
        + 0.075 * instrument_load
        - 4.8 * radiator_area
        - 7.5 * np.log1p(distance_au)
        + 6.0 * np.cos(np.deg2rad(attitude_angle))
        + 3.5 * np.sin(orbital_radius / 1.8e5)
        + rng.normal(0, 4.5, n_samples)
    )

    data = pd.DataFrame(
        {
            "solar_flux": solar_flux,
            "distance_au": distance_au,
            "orbital_radius": orbital_radius,
            "albedo": albedo,
            "heater_power": heater_power,
            "radiator_area": radiator_area,
            "instrument_load": instrument_load,
            "attitude_angle": attitude_angle,
            TARGET_COLUMN: temperature_c,
        }
    )

    for column in FEATURE_COLUMNS:
        mask = rng.random(n_samples) < missing_rate
        data.loc[mask, column] = np.nan

    outlier_count = max(1, n_samples // 80)
    outlier_index = rng.choice(n_samples, size=outlier_count, replace=False)
    data.loc[outlier_index, TARGET_COLUMN] += rng.normal(55, 12, outlier_count)
    return data
