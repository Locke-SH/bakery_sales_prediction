import pandas as pd

# CSV-Dateien einlesen
umsatzdaten = pd.read_csv('umsatzdaten_gekuerzt.csv')
kiwo = pd.read_csv('kiwo.csv')
wetter = pd.read_csv('wetter.csv')

# Merge der Dataframes basierend auf der Spalte 'Datum'
merged_data = umsatzdaten.merge(kiwo, on='Datum', how='left')
merged_data = merged_data.merge(wetter, on='Datum', how='left')

# Überprüfung des zusammengeführten Dataframes
print(merged_data.info())
print("\nErste Zeilen des zusammengeführten Dataframes:")
print(merged_data.head())

# Speichern des zusammengeführten Dataframes
merged_data.to_csv('merged_data.csv', index=False)