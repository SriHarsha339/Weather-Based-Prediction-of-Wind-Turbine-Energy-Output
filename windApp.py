"""Flask application for weather-driven wind turbine power prediction."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "wind_power_model.sav"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"
CONFIG_PATH = BASE_DIR / "config" / "api_config.json"
DEFAULT_SECRET = "change-me-in-production"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", DEFAULT_SECRET)

model_pipeline = None
metadata: Dict[str, object] | None = None
model_error: Optional[str] = None


def load_artifacts() -> None:
    global model_pipeline, metadata, model_error
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model_error = None
    except Exception as exc:  # pragma: no cover - startup diagnostics
        model_pipeline = None
        metadata = None
        model_error = (
            f"Failed to load model or metadata. Ensure training is complete. Details: {exc}"
        )


def get_feature_columns() -> list[str]:
    if metadata and "feature_columns" in metadata:
        return metadata["feature_columns"]
    return []


def get_api_key() -> Optional[str]:
    env_key = os.getenv("OPENWEATHER_API_KEY")
    if env_key:
        return env_key
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("openweather_api_key") or None
        except json.JSONDecodeError:
            return None
    return None


def fetch_weather_by_city(city: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    api_key = get_api_key()
    if not api_key:
        return None, "Missing OpenWeather API key. Set OPENWEATHER_API_KEY before fetching weather data."
    if not city:
        return None, "City name is required."

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
            timeout=6,
        )
    except requests.RequestException as exc:
        return None, f"Weather service error: {exc}"

    if response.status_code != 200:
        message = response.json().get("message", "Unknown error") if response.content else "Unknown error"
        return None, f"Weather API responded with status {response.status_code}: {message}"

    payload = response.json()
    wind = payload.get("wind", {})
    main = payload.get("main", {})

    weather_data = {
        "city": payload.get("name", city.title()),
        "description": payload.get("weather", [{}])[0].get("description", "N/A").title(),
        "wind_speed": wind.get("speed"),
        "wind_direction": wind.get("deg"),
        "temperature": main.get("temp"),
        "pressure": main.get("pressure"),
        "humidity": main.get("humidity"),
    }

    return weather_data, None


def map_weather_to_features(weather_data: Dict[str, object]) -> Dict[str, object]:
    feature_defaults: Dict[str, object] = {}
    for feature in get_feature_columns():
        lower = feature.lower()
        if "wind" in lower and "speed" in lower:
            feature_defaults[feature] = weather_data.get("wind_speed")
        elif "direction" in lower:
            feature_defaults[feature] = weather_data.get("wind_direction")
        elif "temp" in lower:
            feature_defaults[feature] = weather_data.get("temperature")
        elif "humid" in lower:
            feature_defaults[feature] = weather_data.get("humidity")
        elif "pressure" in lower:
            feature_defaults[feature] = weather_data.get("pressure")
    return feature_defaults


def validate_inputs(form_data: Dict[str, str]) -> Tuple[Dict[str, float], list[str]]:
    cleaned: Dict[str, float] = {}
    errors: list[str] = []
    for feature in get_feature_columns():
        value = form_data.get(feature)
        if value in (None, ""):
            errors.append(f"Missing value for {feature}.")
            continue
        try:
            cleaned[feature] = float(value)
        except ValueError:
            errors.append(f"{feature} must be numeric.")
    return cleaned, errors


def predict_power(feature_values: Dict[str, float]) -> Tuple[Optional[float], Optional[str]]:
    if model_error:
        return None, model_error
    if not model_pipeline:
        return None, "Model is not loaded. Train the model first."

    missing_features = [col for col in get_feature_columns() if col not in feature_values]
    if missing_features:
        return None, f"Missing inputs for: {', '.join(missing_features)}"

    df = pd.DataFrame([feature_values])
    try:
        prediction = model_pipeline.predict(df)[0]
        return float(prediction), None
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return None, f"Prediction failed: {exc}"


def render_home(**context):
    defaults = {
        "weather_data": None,
        "weather_error": None,
        "prediction": None,
        "prediction_error": None,
        "feature_defaults": {},
        "feature_columns": get_feature_columns(),
    }
    defaults.update(context)
    return render_template("index.html", **defaults)


@app.route("/", methods=["GET"])
def home():
    return render_home()


@app.route("/weather", methods=["POST"])
def weather():
    city = request.form.get("city", "").strip()
    weather_data, error = fetch_weather_by_city(city)
    feature_defaults = map_weather_to_features(weather_data) if weather_data else {}
    return render_home(
        weather_data=weather_data,
        weather_error=error,
        feature_defaults=feature_defaults,
    )


@app.route("/predict", methods=["POST"])
def predict():
    cleaned_inputs, validation_errors = validate_inputs(request.form)
    if validation_errors:
        return render_home(
            prediction_error="; ".join(validation_errors),
            feature_defaults=cleaned_inputs,
        )

    prediction, error = predict_power(cleaned_inputs)
    return render_home(
        prediction=prediction,
        prediction_error=error,
        feature_defaults=cleaned_inputs,
    )


@app.errorhandler(404)
def not_found(_):
    return redirect(url_for("home"))


load_artifacts()

if __name__ == "__main__":
    app.run(debug=True)
