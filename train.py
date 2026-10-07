# Trains the final model and saves it for server.py.
# Usage: python train.py
import joblib
from carprice import HERE, train

bundle = train()
joblib.dump(bundle, HERE / "car_model.joblib", compress=3)
print("saved car_model.joblib", len(bundle['columns']), "columns")
