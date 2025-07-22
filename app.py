from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Cargar el modelo entrenado
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

estado_colmena = {0: "Saludable", 1: "En observación", 2: "Crítica"}

@app.route("/")
def home():
    return "API de Predicción de Salud de Colmenas funcionando"

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return jsonify({"estado": estado_colmena[pred]})
