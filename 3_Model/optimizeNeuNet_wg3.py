import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Erstelle Verzeichnis für Pickle-Dateien
subdirectory = "pickle_data"
if not os.path.exists(subdirectory):
    os.makedirs(subdirectory)

# Import Data
data = pd.read_csv("merged_data_full_with_weekdays.csv")

# Filtere für gewünschte Warengruppe (z.B. Warengruppe 1)
selected_group = 3  # Hier die gewünschte Warengruppe eintragen (1, 2 oder 3)
data = data[data['Warengruppe'] == selected_group]

# Definiere kategorische Features
categorical_features = ['Warengruppe', 'Wettercode']

# Konvertiere kategorische Spalten in Kategorie-Datentyp
for col in categorical_features:
    data[col] = data[col].astype('category')

# One-Hot-Encoding für kategorische Variablen
features = pd.get_dummies(data[categorical_features], drop_first=True, dtype=int)

# Füge numerische Spalten hinzu
numeric_features = ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'KielerWoche', 'Feiertag',
                   'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
for feature in numeric_features:
    features[feature] = data[feature]

# Erstelle den vorbereiteten Datensatz mit der Zielvariable ('Umsatz')
prepared_data = pd.concat([data[['Datum', 'Umsatz']], features], axis=1)

# Setze Datum als Index
prepared_data['Datum'] = pd.to_datetime(prepared_data['Datum'])
prepared_data.set_index('Datum', inplace=True)

# Behandle übrige fehlende Werte
#for feature in ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit']:
#    prepared_data[feature].fillna(prepared_data[feature].mean(), inplace=True)

# Setze Zufallsseed für Reproduzierbarkeit
np.random.seed(42)

# Mische die Daten
prepared_data = prepared_data.sample(frac=1)

# Berechne die Anzahl der Zeilen für jeden Datensatz
n_total = len(prepared_data)
n_training = int(0.7 * n_total)
n_validation = int(0.20 * n_total)

# Teile die Daten in Training, Validierung und Test
training_data = prepared_data.iloc[:n_training]
validation_data = prepared_data.iloc[n_training:n_training+n_validation]
test_data = prepared_data.iloc[n_training+n_validation:]

# Trenne Features und Labels
training_features = training_data.drop('Umsatz', axis=1)
validation_features = validation_data.drop('Umsatz', axis=1)
test_features = test_data.drop('Umsatz', axis=1)

training_labels = training_data[['Umsatz']]
validation_labels = validation_data[['Umsatz']]
test_labels = test_data[['Umsatz']]

# Speichere die aufbereiteten Datensätze als Pickle
training_features.to_pickle(f"{subdirectory}/training_features_group_{selected_group}.pkl")
training_labels.to_pickle(f"{subdirectory}/training_labels_group_{selected_group}.pkl")
validation_features.to_pickle(f"{subdirectory}/validation_features_group_{selected_group}.pkl")
validation_labels.to_pickle(f"{subdirectory}/validation_labels_group_{selected_group}.pkl")
test_features.to_pickle(f"{subdirectory}/test_features_group_{selected_group}.pkl")
test_labels.to_pickle(f"{subdirectory}/test_labels_group_{selected_group}.pkl")

# Passe die Modellarchitektur an die reduzierte Datenmenge an
model = Sequential([
    InputLayer(shape=(training_features.shape[1],)),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(8, activation='relu'),
    Dropout(0.1),
    Dense(4, activation='relu'),
    Dense(1)
])

model.summary()

# Kompilierung
model.compile(loss="mse",
             optimizer=Adam(learning_rate=0.0005))

# Training
history = model.fit(training_features,
                   training_labels,
                   epochs=50,
                   batch_size=32,
                   validation_data=(validation_features, validation_labels))

# Speichere das Modell
model.save(f"umsatz_model_group_{selected_group}.h5")

# Plotte den Trainingsverlauf
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Modellverlust während des Trainings - Warengruppe {selected_group}')
plt.xlabel('Epochen')
plt.ylabel('Verlust')
plt.legend()
plt.show()

# MAPE Funktion
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# Berechne Vorhersagen
training_predictions = model.predict(training_features)
validation_predictions = model.predict(validation_features)

# Berechne und zeige MAPE
print(f"MAPE auf den Trainingsdaten (Warengruppe {selected_group}): {mape(training_labels, training_predictions):.2f}%")
print(f"MAPE auf den Validierungsdaten (Warengruppe {selected_group}): {mape(validation_labels, validation_predictions):.2f}%")

# Plot-Funktion
def plot_predictions(data, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data['actual'], label='Tatsächliche Werte', color='red')
    plt.plot(data['prediction'], label='Vorhergesagte Werte', color='blue')
    plt.title(title)
    plt.xlabel('Fallnummer')
    plt.ylabel('Umsatz')
    plt.legend()
    plt.show()

# Bereite Daten für das Plotting vor
training_predictions = np.array(training_predictions).flatten()
validation_predictions = np.array(validation_predictions).flatten()
training_labels = np.array(training_labels).flatten()
validation_labels = np.array(validation_labels).flatten()

# Erstelle DataFrames für das Plotting
data_train = pd.DataFrame({'prediction': training_predictions, 'actual': training_labels})
data_validation = pd.DataFrame({'prediction': validation_predictions, 'actual': validation_labels})

# Plotte die Vorhersagen
plot_predictions(data_train.head(100), f'Vorhergesagte und tatsächliche Werte für die Trainingsdaten - Warengruppe {selected_group}')
plot_predictions(data_validation.head(100), f'Vorhergesagte und tatsächliche Werte für die Validierungsdaten - Warengruppe {selected_group}')