# Trains the random forest with the same cleaning as the analysis and saves it for the app.
# Usage: put cardekho_dataset.csv in data/, then run `python train.py`
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "data" / "cardekho_dataset.csv")

# same cleaning steps as notebook/car_price_analysis.py
df = df[df['seats'] != 0]
df['brand'] = df['brand'].str.lower()
df = df[df['km_driven'] <= 950000]
counts = df['brand'].value_counts()
rare_cars = counts[counts < 10]
df = df[~df['brand'].isin(rare_cars.index)]

feature = ['brand','model','vehicle_age','km_driven','seller_type','fuel_type',
           'transmission_type','mileage','engine','max_power','seats']
X = pd.get_dummies(df[feature])
y = df['selling_price']

# final model is trained on all the cleaned data (5-fold CV already gave R2 ~0.93)
rf = RandomForestRegressor(n_estimators=100, random_state=34, n_jobs=-1)
rf.fit(X, y)

# typical specs per car model, used to pre-fill the form
specs = df.groupby(['brand', 'model'])[['mileage','engine','max_power','seats']].median().reset_index()

joblib.dump({'model': rf, 'columns': X.columns, 'specs': specs,
             'fuel_types': sorted(df['fuel_type'].unique()),
             'seller_types': sorted(df['seller_type'].unique())},
            HERE / "car_model.joblib", compress=3)
print("saved car_model.joblib", X.shape)
