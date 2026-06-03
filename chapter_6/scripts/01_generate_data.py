import _bootstrap  # noqa: F401

from deepspace_temperature.config import RAW_DATA_PATH, ensure_project_dirs
from deepspace_temperature.data import generate_temperature_data


def main() -> None:
    ensure_project_dirs()
    data = generate_temperature_data()
    data.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw data: {RAW_DATA_PATH} ({data.shape[0]} rows)")


if __name__ == "__main__":
    main()
