from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Cargar el modelo entrenado
with open("model_salud.pkl", "rb") as f:
    model = pickle.load(f)

with open("modelo_enjambrazon.pkl", "rb") as f:
    modelo = pickle.load(f)

estado_colmena = {0: "Saludable", 1: "En observación", 2: "Crítica"}

@app.route("/")
def home():
    return "API de Predicción de Salud de Colmenas funcionando"

@app.route("/predecir", methods=["POST"])
def predecir_salud():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return jsonify({"estado": estado_colmena[pred]})

@app.route("/predecir-enjambrazon", methods=["POST"])
def predecir_enjambrazon():
    datos = request.get_json()
    df = pd.DataFrame([datos])
    pred = modelo.predict(df)[0]
    prob = modelo.predict_proba(df)[0][1]

    return jsonify({
        "enjambrazon_probable": bool(pred),
        "confianza": round(prob, 2)
    })