import pandas as pd

# Daten laden
merged_data = pd.read_csv('merged_data.csv')

# Datum-Spalte in datetime konvertieren
merged_data['Datum'] = pd.to_datetime(merged_data['Datum'])

# Definieren der Zeiträume
training_start = '2013-07-01'
training_end = '2017-07-31'
validation_start = '2017-08-01'
validation_end = '2018-07-31'

# Trainingsdatensatz erstellen
training_data = merged_data[
    (merged_data['Datum'] >= training_start) & 
    (merged_data['Datum'] <= training_end)
]

# Validierungsdatensatz erstellen
validation_data = merged_data[
    (merged_data['Datum'] >= validation_start) & 
    (merged_data['Datum'] <= validation_end)
]

# Speichern der Datensätze
training_data.to_csv('training_data.csv', index=False)
validation_data.to_csv('validation_data.csv', index=False)

# Überprüfung der Datensätze
print("Trainingsdatensatz:")
print("Zeitraum:", training_start, "bis", training_end)
print("Anzahl der Zeilen:", len(training_data))
print("Erste Zeile:", training_data['Datum'].min())
print("Letzte Zeile:", training_data['Datum'].max())

print("\nValidierungsdatensatz:")
print("Zeitraum:", validation_start, "bis", validation_end)
print("Anzahl der Zeilen:", len(validation_data))
print("Erste Zeile:", validation_data['Datum'].min())
print("Letzte Zeile:", validation_data['Datum'].max())
