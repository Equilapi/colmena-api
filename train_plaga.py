import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.DataFrame([
    {"restos_miel_expuestos": 1, "colmena_sellada": 0, "debilidad_colmena": 1,
     "alimentacion_azucar": 1, "frecuencia_visitas": 5, "humedad_alta": 0, "presencia_plaga": 1},
    
    {"restos_miel_expuestos": 0, "colmena_sellada": 1, "debilidad_colmena": 0,
     "alimentacion_azucar": 0, "frecuencia_visitas": 0, "humedad_alta": 1, "presencia_plaga": 0},

    {"restos_miel_expuestos": 1, "colmena_sellada": 0, "debilidad_colmena": 0,
     "alimentacion_azucar": 1, "frecuencia_visitas": 3, "humedad_alta": 1, "presencia_plaga": 1},

    {"restos_miel_expuestos": 0, "colmena_sellada": 1, "debilidad_colmena": 0,
     "alimentacion_azucar": 0, "frecuencia_visitas": 1, "humedad_alta": 0, "presencia_plaga": 0},
])

X = data.drop("presencia_plaga", axis=1)
y = data["presencia_plaga"]

modelo_plaga = RandomForestClassifier()
modelo_plaga.fit(X, y)

with open("modelo_avispa.pkl", "wb") as f:
    pickle.dump(modelo_plaga, f)
