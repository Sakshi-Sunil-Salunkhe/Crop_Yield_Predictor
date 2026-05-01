import argparse
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "crop_yield_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crop yield.")
    parser.add_argument("--rainfall", type=float, required=True, help="Rainfall in mm")
    parser.add_argument(
        "--temperature", type=float, required=True, help="Temperature in degree Celsius"
    )
    parser.add_argument("--soil", type=str, required=True, help="Soil type")
    parser.add_argument(
        "--fertilizer",
        type=float,
        required=True,
        help="Fertilizer usage in kg per hectare",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python src/train.py' before prediction."
        )

    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "rainfall_mm": args.rainfall,
                "temperature_c": args.temperature,
                "soil_type": args.soil,
                "fertilizer_kg_per_hectare": args.fertilizer,
            }
        ]
    )

    prediction = model.predict(sample)[0]
    print(f"Predicted Yield: {prediction:.2f} tons/hectare")


if __name__ == "__main__":
    main()
