# Crop Yield Detection Using Machine Learning

This project predicts crop yield in **tons per hectare** using machine learning. It uses agricultural factors such as rainfall, temperature, soil type, and fertilizer usage to estimate production in advance.

## 1. Introduction

Agriculture is a major part of India's economy, but farmers often face uncertainty in crop production because of:

- Unpredictable rainfall
- Changing temperature
- Soil quality variations
- Improper fertilizer usage

Because of these factors, farmers cannot accurately estimate crop yield in advance.

## 2. Problem Statement

Many farmers still rely on experience and guesswork to estimate production. This can lead to:

- Low productivity
- Poor crop planning
- Financial loss

So, there is a need for a data-driven system that can predict crop yield more accurately.

## 3. Objective

To develop a machine learning model that predicts crop yield based on:

- Rainfall
- Temperature
- Soil type
- Fertilizer usage

## 4. Why Machine Learning?

Traditional methods struggle to handle many factors at the same time. Machine learning can:

- Learn patterns from historical data
- Capture complex relationships
- Provide better predictions

## 5. Real-World Use

This system can help:

- Farmers plan crop selection and fertilizer use
- Government estimate food production
- Businesses manage supply chains

Example:
If the predicted yield is low, the farmer can adjust fertilizer usage or choose a different crop strategy.

## 6. Type of Machine Learning Problem

This is a **regression problem** because the output is a continuous value:

`Yield in tons/hectare`

## 7. Example Input and Output

Input:

- Rainfall = 800 mm
- Temperature = 28 C
- Soil = Loamy
- Fertilizer = 140 kg/hectare

Output:

- Predicted Yield = 4.3 tons/hectare

## 8. Project Structure

```text
.
|-- app.py
|-- data/
|   `-- crop_yield_dataset.csv
|-- model/
|-- src/
|   |-- predict.py
|   `-- train.py
|-- requirements.txt
`-- README.md
```

## 9. Model Used

The project uses:

- `RandomForestRegressor` for prediction
- `OneHotEncoder` for soil type conversion
- `Pipeline` and `ColumnTransformer` for preprocessing and modeling

## 10. How to Run

Create and activate a virtual environment if you want:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train the model:

```powershell
python src/train.py
```

Start the web UI:

```powershell
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

You can also predict from the command line:

```powershell
python src/predict.py --rainfall 800 --temperature 28 --soil Loamy --fertilizer 140
```

## 11. Conclusion

This project demonstrates how machine learning can improve agricultural decision-making, reduce risk, and increase productivity.

It is a practical example of using AI in agriculture for smarter planning and better outcomes.
