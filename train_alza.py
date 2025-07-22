import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Datos de ejemplo (puedes ampliarlos)
data = pd.DataFrame([
    {"cuadros_abejas": 9, "cuadros_cria": 6, "cuadros_miel": 5, "espacio_ocupado_pct": 90,
     "floracion_abundante": 1, "reina_activa": 1, "actividad_piquera_alta": 1, "temperatura_alta": 1, "necesita_alza": 1},

    {"cuadros_abejas": 6, "cuadros_cria": 3, "cuadros_miel": 2, "espacio_ocupado_pct": 50,
     "floracion_abundante": 0, "reina_activa": 1, "actividad_piquera_alta": 0, "temperatura_alta": 0, "necesita_alza": 0},

    {"cuadros_abejas": 8, "cuadros_cria": 5, "cuadros_miel": 4, "espacio_ocupado_pct": 85,
     "floracion_abundante": 1, "reina_activa": 1, "actividad_piquera_alta": 1, "temperatura_alta": 1, "necesita_alza": 1},

    {"cuadros_abejas": 5, "cuadros_cria": 2, "cuadros_miel": 1, "espacio_ocupado_pct": 40,
     "floracion_abundante": 0, "reina_activa": 0, "actividad_piquera_alta": 0, "temperatura_alta": 0, "necesita_alza": 0}
])

X = data.drop("necesita_alza", axis=1)
y = data["necesita_alza"]

modelo_alza = RandomForestClassifier()
modelo_alza.fit(X, y)

with open("modelo_alza_espacio.pkl", "wb") as f:
    pickle.dump(modelo_alza, f)
