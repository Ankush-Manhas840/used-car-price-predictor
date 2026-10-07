# Small Flask server: serves the page and answers predictions from the saved model.
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).parent
if not (HERE / "car_model.joblib").exists():
    raise SystemExit("car_model.joblib not found. Run `python train.py` first.")
saved = joblib.load(HERE / "car_model.joblib")
rf, columns, specs = saved['model'], saved['columns'], saved['specs']

app = Flask(__name__, static_folder=str(HERE / "static"))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/options")
def options():
    cars = {}
    for _, r in specs.iterrows():
        cars.setdefault(r['brand'], {})[r['model']] = {
            'mileage': round(float(r['mileage']), 1), 'engine': int(r['engine']),
            'max_power': round(float(r['max_power']), 1), 'seats': int(r['seats'])}
    return jsonify(cars=cars, fuel_types=saved['fuel_types'], seller_types=saved['seller_types'])


@app.post("/api/predict")
def predict():
    car = pd.DataFrame([request.get_json()])
    # same trick as X_test in the notebook: line the columns up with training
    car = pd.get_dummies(car).reindex(columns=columns, fill_value=0)
    return jsonify(price=float(rf.predict(car)[0]))


if __name__ == "__main__":
    app.run(port=8502)
