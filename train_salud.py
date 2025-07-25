import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.DataFrame([
    {"cuadros_abejas": 9, "cuadros_cria": 6, "alimentada": 1, "tratamiento": 0, "enfermedades": 0, "salud": 0},
    {"cuadros_abejas": 4, "cuadros_cria": 2, "alimentada": 0, "tratamiento": 1, "enfermedades": 1, "salud": 2},
    {"cuadros_abejas": 6, "cuadros_cria": 3, "alimentada": 1, "tratamiento": 1, "enfermedades": 1, "salud": 1},
])

X = data.drop("salud", axis=1)
y = data["salud"]

model = RandomForestClassifier()
model.fit(X, y)

with open("modelo_salud.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modelo entrenado y guardado como model_salud.pkl")
