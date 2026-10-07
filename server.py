# Small Flask server: serves the page and answers predictions from the saved model.
import joblib
from flask import Flask, jsonify, request, send_from_directory
from carprice import HERE, options, predict

if not (HERE / "car_model.joblib").exists():
    raise SystemExit("car_model.joblib not found. Run `python train.py` first.")
bundle = joblib.load(HERE / "car_model.joblib")
opts = options(bundle)

app = Flask(__name__, static_folder=str(HERE / "static"))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/options")
def get_options():
    return jsonify(opts)


@app.post("/api/predict")
def get_price():
    return jsonify(price=predict(bundle, request.get_json()))


if __name__ == "__main__":
    app.run(port=8502)
