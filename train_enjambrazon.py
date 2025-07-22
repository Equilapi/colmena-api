import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Datos de ejemplo (debes poner más reales si tienes)
data = pd.DataFrame([
    {"cuadros_abejas": 9, "cuadros_cria": 6, "presencia_celdas": 1, "reina_vieja": 1,
     "produccion_baja": 1, "temperatura_alta": 1, "hacinamiento": 1, "enjambrazon": 1},
    {"cuadros_abejas": 6, "cuadros_cria": 3, "presencia_celdas": 0, "reina_vieja": 0,
     "produccion_baja": 0, "temperatura_alta": 0, "hacinamiento": 0, "enjambrazon": 0},
    {"cuadros_abejas": 8, "cuadros_cria": 5, "presencia_celdas": 1, "reina_vieja": 1,
     "produccion_baja": 1, "temperatura_alta": 0, "hacinamiento": 1, "enjambrazon": 1},
    {"cuadros_abejas": 7, "cuadros_cria": 4, "presencia_celdas": 0, "reina_vieja": 0,
     "produccion_baja": 0, "temperatura_alta": 1, "hacinamiento": 0, "enjambrazon": 0},
])

X = data.drop("enjambrazon", axis=1)
y = data["enjambrazon"]

modelo = RandomForestClassifier()
modelo.fit(X, y)

with open("modelo_enjambrazon.pkl", "wb") as f:
    pickle.dump(modelo, f)