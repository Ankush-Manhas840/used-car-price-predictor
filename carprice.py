# Shared model code used by train.py, server.py and streamlit_app.py
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HERE = Path(__file__).parent
FEATURES = ['brand','model','vehicle_age','km_driven','seller_type','fuel_type',
            'transmission_type','mileage','engine','max_power','seats']


def load_clean_data():
    df = pd.read_csv(HERE / "data" / "cardekho_dataset.csv")
    # same cleaning steps as notebook/car_price_analysis.py
    df = df[df['seats'] != 0]
    df['brand'] = df['brand'].str.lower()
    df = df[df['km_driven'] <= 950000]
    counts = df['brand'].value_counts()
    rare_cars = counts[counts < 10]
    return df[~df['brand'].isin(rare_cars.index)]


def train():
    df = load_clean_data()
    X = pd.get_dummies(df[FEATURES])
    y = df['selling_price']
    # final model is trained on all the cleaned data (5-fold CV already gave R2 ~0.93)
    rf = RandomForestRegressor(n_estimators=100, random_state=34, n_jobs=-1)
    rf.fit(X, y)
    # typical specs per car model, used to pre-fill the form
    specs = df.groupby(['brand', 'model'])[['mileage','engine','max_power','seats']].median().reset_index()
    return {'model': rf, 'columns': X.columns, 'specs': specs,
            'fuel_types': sorted(df['fuel_type'].unique()),
            'seller_types': sorted(df['seller_type'].unique())}


def options(bundle):
    cars = {}
    for _, r in bundle['specs'].iterrows():
        cars.setdefault(r['brand'], {})[r['model']] = {
            'mileage': round(float(r['mileage']), 1), 'engine': int(r['engine']),
            'max_power': round(float(r['max_power']), 1), 'seats': int(r['seats'])}
    return {'cars': cars, 'fuel_types': bundle['fuel_types'], 'seller_types': bundle['seller_types']}


def predict(bundle, car):
    car = pd.DataFrame([{f: car[f] for f in FEATURES}])
    # same trick as X_test in the notebook: line the columns up with training
    car = pd.get_dummies(car).reindex(columns=bundle['columns'], fill_value=0)
    return float(bundle['model'].predict(car)[0])
