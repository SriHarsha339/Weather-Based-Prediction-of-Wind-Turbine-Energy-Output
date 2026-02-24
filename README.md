# Weather-Based Prediction of Wind Turbine Energy Output: A Next-Generation Approach to Renewable Energy Management

Modern wind farms collect a continuous stream of atmospheric telemetry. This repository stitches those feeds into an end-to-end workflow that cleans the data, trains a RandomForestRegressor, generates diagnostics, and exposes real-time predictions through a Flask web application.

## Repository Map

```
Smartinternz Project/
├── README.md                         # You are here
├── requirements.txt                  # Python dependencies
├── train_model.py                    # Data prep + training + EDA
├── windApp.py                        # Flask web server
├── config/
│   └── api_config.json               # Optional API key fallback
├── data/
│   └── wind_dataset.csv              # Working dataset (replace with your own)
├── model/
│   ├── wind_power_model.sav          # Serialized model
│   └── metadata.json                 # Feature list + metrics + provenance
├── static/
│   ├── style.css                     # UI theme
│   └── plots/                        # Auto-generated EDA charts
└── templates/
    └── index.html                    # Flask UI template
```

## Key Capabilities

- **Weather-aware preprocessing:** Column-name normalization plus automatic detection of the power target, wind speed, direction, and theoretical curve columns (with guardrails when absent).
- **Exploratory Data Analysis:** Correlation heatmap, scatter plots, and target distribution saved to `static/plots/` for quick inspection or dashboards.
- **Robust Modeling:** A scikit-learn pipeline (imputers + scalers + RandomForestRegressor) with metrics logged to `model/metadata.json` for traceability.
- **Live Flask Dashboard:** Dual-tab UI for (A) fetching city-level weather via OpenWeather and (B) manual/auto-filled prediction inputs.
- **Model Persistence:** `joblib` artifacts enable reproducible deployments and simple CI/CD packaging.

## Data Requirements

| Requirement | Description |
|-------------|-------------|
| Target column | Must contain actual turbine power values. The auto-detector prefers any column containing "power" but not "theoretical". |
| Feature columns | Wind speed, direction, theoretical power curve, air density, ambient temperature, etc. are auto-detected when possible. |
| File format | CSV with headers. Empty rows are dropped automatically. |

> ⚠️ If your dataset lacks a clear power column, `train_model.py` will raise a descriptive error. Rename or add the target column before retraining.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## OpenWeather API Key

Set the key for both the CLI training workflow (if scraping weather data) and the Flask app:

- **Linux/macOS**
  ```bash
  export OPENWEATHER_API_KEY="YOUR_KEY"
  ```
- **Windows PowerShell**
  ```powershell
  setx OPENWEATHER_API_KEY "YOUR_KEY"
  ```

As a fallback, copy `config/api_config.json.example` to `config/api_config.json` and update with your key—just avoid committing secrets to Git history.

## Training Workflow

```bash
python train_model.py --data data/wind_dataset.csv \
                      --model-dir model \
                      --plot-dir static/plots
```

Outputs:

- `model/wind_power_model.sav` – Trained pipeline
- `model/metadata.json` – Feature names, metrics (MAE/RMSE/R²), training timestamp, and data lineage
- `static/plots/*.png` – Correlation, scatter, and distribution figures for reports

## Running the Flask App

```bash
python windApp.py
# Navigate to http://127.0.0.1:5000
```

Features:
- Fetch current weather by city (OpenWeather) and auto-populate the predictor form.
- Manual overrides for every feature required by the model (including theoretical curve inputs that APIs don’t expose).
- Friendly error states for missing model artifacts, invalid API keys, or malformed inputs.

## Deployment Checklist (GitHub)

1. **Create the repository on GitHub** named `Weather-Based Prediction of Wind Turbine Energy Output: A Next-Generation Approach to Renewable Energy Management`.
2. Initialize locally if you haven’t already:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: wind turbine power prediction"
   git remote add origin https://github.com/<you>/Weather-Based...git
   git push -u origin main
   ```
3. Configure branch protection / CI (optional) and store secrets (e.g., `OPENWEATHER_API_KEY`) using GitHub Actions secrets or environment variables.

## Use Cases

- **Short-term forecasting:** Estimate next-hour turbine output for demand-response planning.
- **Predictive maintenance:** Compare theoretical vs. actual power to flag derating turbines.
- **Grid harmonization:** Share predicted kW with utilities to manage spinning reserve commitments.

## Limitations & Future Enhancements

- Current model assumes a power column is present; feature engineering for derived power should be handled upstream.
- RandomForestRegressor is resilient but may underfit long-term temporal trends; consider Gradient Boosting or hybrid physics-ML models for production fleets.
- Weather fetch covers current conditions only—extend with forecast endpoints for multi-hour lead times.
- Add CI workflows (pytest/lint) and containerization for smoother cloud deployments.

## License

Specify your license of choice (MIT, Apache-2.0, etc.) before publishing. Update this section accordingly.
