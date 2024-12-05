import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

# Daten laden
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'training_data.csv')
training_data = pd.read_csv(file_path)

training_data['Datum'] = pd.to_datetime(training_data['Datum'])

# Feature Engineering
training_data['Jahr'] = training_data['Datum'].dt.year
training_data['Monat'] = training_data['Datum'].dt.month
training_data['Wochentag'] = training_data['Datum'].dt.dayofweek

# Kategorische und numerische Features vorbereiten
kategorische_features = ['Warengruppe', 'Feiertag']
numerische_features = ['Temperatur', 'Windgeschwindigkeit', 'Bewoelkung', 'KielerWoche', 
                       'Jahr', 'Monat', 'Wochentag']

# Preprocessor für kategorische und numerische Daten
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerische_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ]), kategorische_features),
    ]
)

# Pipeline für Gradient Boosting
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor())
])

# Vorbereitung der Features und Zielvariable
X = training_data[kategorische_features + numerische_features]
y = training_data['Umsatz']

# Modell fitten
pipeline.fit(X, y)

# Vorhersagen und R² berechnen
y_pred = pipeline.predict(X)
r2 = r2_score(y, y_pred)

# Adjusted R² berechnen
n = X.shape[0]  # Anzahl der Beobachtungen
p = X.shape[1]  # Anzahl der Prädiktoren
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"R²: {r2:.4f}")
print(f"Adjustiertes R²: {adjusted_r2:.4f}")
