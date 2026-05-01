from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_yield_model.joblib"

SOIL_TYPES = ["Black", "Clay", "Loamy", "Red", "Sandy", "Silt"]

app = FastAPI(title="Crop Yield Predictor")


def render_page(
    rainfall: float | None = None,
    temperature: float | None = None,
    soil: str = "Loamy",
    fertilizer: float | None = None,
    prediction: float | None = None,
    error: str | None = None,
) -> str:
    options = []
    for item in SOIL_TYPES:
        selected = "selected" if item == soil else ""
        options.append(f'<option value="{item}" {selected}>{item}</option>')

    result_markup = ""
    if prediction is not None:
        result_markup = f"""
        <section class="result-card success">
            <p class="eyebrow">Prediction Result</p>
            <h2>{prediction:.2f} tons/hectare</h2>
            <p>The model estimates this crop yield from the values you entered.</p>
        </section>
        """
    elif error:
        result_markup = f"""
        <section class="result-card error">
            <p class="eyebrow">Unable to Predict</p>
            <h2>Please check the setup</h2>
            <p>{error}</p>
        </section>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Crop Yield Predictor</title>
        <style>
            :root {{
                --bg: #f5efe3;
                --panel: #fffaf2;
                --ink: #203227;
                --muted: #5f6f63;
                --accent: #1f7a4c;
                --accent-2: #d89b3c;
                --border: #d9ccb8;
                --error: #a53c2f;
                --shadow: 0 20px 40px rgba(32, 50, 39, 0.10);
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                font-family: Georgia, "Times New Roman", serif;
                color: var(--ink);
                background:
                    radial-gradient(circle at top right, rgba(216, 155, 60, 0.22), transparent 26%),
                    radial-gradient(circle at bottom left, rgba(31, 122, 76, 0.18), transparent 28%),
                    var(--bg);
                min-height: 100vh;
            }}

            .shell {{
                width: min(1080px, calc(100% - 32px));
                margin: 32px auto;
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 24px;
                align-items: start;
            }}

            .hero,
            .panel {{
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 24px;
                box-shadow: var(--shadow);
            }}

            .hero {{
                padding: 32px;
                position: relative;
                overflow: hidden;
            }}

            .hero::after {{
                content: "";
                position: absolute;
                right: -40px;
                top: -40px;
                width: 180px;
                height: 180px;
                background: linear-gradient(135deg, rgba(216, 155, 60, 0.18), rgba(31, 122, 76, 0.12));
                border-radius: 50%;
            }}

            .eyebrow {{
                margin: 0 0 12px;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.78rem;
                color: var(--accent);
                font-weight: 700;
            }}

            h1 {{
                margin: 0 0 16px;
                font-size: clamp(2.2rem, 4vw, 4rem);
                line-height: 0.95;
            }}

            .hero p {{
                margin: 0;
                max-width: 36rem;
                color: var(--muted);
                font-size: 1.05rem;
                line-height: 1.7;
            }}

            .tag-row {{
                margin-top: 24px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}

            .tag {{
                padding: 10px 14px;
                border-radius: 999px;
                background: rgba(31, 122, 76, 0.08);
                color: var(--ink);
                font-size: 0.92rem;
            }}

            .panel {{
                padding: 28px;
            }}

            form {{
                display: grid;
                gap: 16px;
            }}

            label {{
                display: grid;
                gap: 8px;
                color: var(--ink);
                font-weight: 700;
            }}

            input,
            select {{
                width: 100%;
                padding: 14px 16px;
                border-radius: 14px;
                border: 1px solid var(--border);
                background: #fff;
                color: var(--ink);
                font-size: 1rem;
            }}

            input:focus,
            select:focus {{
                outline: 2px solid rgba(31, 122, 76, 0.24);
                border-color: var(--accent);
            }}

            button {{
                border: 0;
                border-radius: 14px;
                padding: 15px 18px;
                font-size: 1rem;
                font-weight: 700;
                color: #fff;
                background: linear-gradient(135deg, var(--accent), #165d39);
                cursor: pointer;
                transition: transform 120ms ease, box-shadow 120ms ease;
                box-shadow: 0 12px 24px rgba(31, 122, 76, 0.24);
            }}

            button:hover {{
                transform: translateY(-1px);
            }}

            .result-card {{
                margin-top: 22px;
                padding: 22px;
                border-radius: 18px;
            }}

            .result-card.success {{
                background: linear-gradient(135deg, rgba(31, 122, 76, 0.10), rgba(216, 155, 60, 0.12));
                border: 1px solid rgba(31, 122, 76, 0.18);
            }}

            .result-card.error {{
                background: rgba(165, 60, 47, 0.08);
                border: 1px solid rgba(165, 60, 47, 0.18);
            }}

            .result-card h2 {{
                margin: 0 0 8px;
                font-size: clamp(1.8rem, 3vw, 2.6rem);
            }}

            .result-card p:last-child {{
                margin: 0;
                color: var(--muted);
                line-height: 1.6;
            }}

            .note {{
                margin-top: 14px;
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.6;
            }}

            @media (max-width: 860px) {{
                .shell {{
                    grid-template-columns: 1fr;
                }}

                .hero,
                .panel {{
                    padding: 24px;
                }}
            }}
        </style>
    </head>
    <body>
        <main class="shell">
            <section class="hero">
                <p class="eyebrow">Machine Learning for Agriculture</p>
                <h1>Crop Yield Predictor</h1>
                <p>
                    Enter rainfall, temperature, soil type, and fertilizer usage to estimate
                    crop yield in tons per hectare using a trained regression model.
                </p>
            </section>

            <section class="panel">
                <form method="get" action="/predict">
                    <label>
                        Rainfall (mm)
                        <input type="number" step="0.1" name="rainfall" value="{'' if rainfall is None else rainfall}" required />
                    </label>
                    <label>
                        Temperature (C)
                        <input type="number" step="0.1" name="temperature" value="{'' if temperature is None else temperature}" required />
                    </label>
                    <label>
                        Soil Type
                        <select name="soil">
                            {''.join(options)}
                        </select>
                    </label>
                    <label>
                        Fertilizer (kg/hectare)
                        <input type="number" step="0.1" name="fertilizer" value="{'' if fertilizer is None else fertilizer}" required />
                    </label>
                    <button type="submit">Predict Yield</button>
                </form>
                {result_markup}
                <p class="note">
                    Example values: Rainfall 800 mm, Temperature 28 C, Soil Loamy, Fertilizer 140 kg/hectare.
                </p>
            </section>
        </main>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return render_page()


@app.get("/predict", response_class=HTMLResponse)
def predict(
    rainfall: float = Query(..., gt=0),
    temperature: float = Query(...),
    soil: str = Query(...),
    fertilizer: float = Query(..., gt=0),
) -> str:
    if not MODEL_PATH.exists():
        return render_page(
            rainfall=rainfall,
            temperature=temperature,
            soil=soil,
            fertilizer=fertilizer,
            error="The trained model is missing. Run 'python src/train.py' first.",
        )

    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "rainfall_mm": rainfall,
                "temperature_c": temperature,
                "soil_type": soil,
                "fertilizer_kg_per_hectare": fertilizer,
            }
        ]
    )
    prediction = float(model.predict(sample)[0])
    return render_page(
        rainfall=rainfall,
        temperature=temperature,
        soil=soil,
        fertilizer=fertilizer,
        prediction=prediction,
    )
