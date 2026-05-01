from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "crop_yield_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "crop_yield_model.joblib"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[[
        "rainfall_mm",
        "temperature_c",
        "soil_type",
        "fertilizer_kg_per_hectare",
    ]]
    y = df["yield_tons_per_hectare"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "soil",
                OneHotEncoder(handle_unknown="ignore"),
                ["soil_type"],
            )
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(n_estimators=200, random_state=42),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model training completed.")
    print(f"Dataset rows: {len(df)}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R2 Score: {r2:.3f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
