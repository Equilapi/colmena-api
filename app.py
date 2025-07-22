from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Definimos el peso promedio por cuadro o paleta
PESO_PROMEDIO_POR_CUADRO = 2.5  # Ajusta este valor según tus datos

# Cargar el modelo entrenado
with open("modelo_salud.pkl", "rb") as f:
    modelo_salud = pickle.load(f)

with open("modelo_enjambrazon.pkl", "rb") as f:
    modelo_enjambrazon = pickle.load(f)

with open("modelo_avispa.pkl", "rb") as f:
    modelo_avispa = pickle.load(f)  

with open("modelo_alza_espacio.pkl", "rb") as f:
    modelo_alza_espacio = pickle.load(f)

estado_colmena = {0: "Saludable", 1: "En observación", 2: "Crítica"}

@app.route("/")
def home():
    return "API de Predicción de Salud de Colmenas funcionando"

@app.route("/predecir-salud", methods=["POST"])
def predecir_salud():
    data = request.json
    df = pd.DataFrame([data])
    pred = modelo_salud.predict(df)[0]
    return jsonify({"estado": estado_colmena[pred]})

@app.route("/predecir-enjambrazon", methods=["POST"])
def predecir_enjambrazon():
    datos = request.get_json()
    df = pd.DataFrame([datos])
    pred = modelo_enjambrazon.predict(df)[0]
    prob = modelo_enjambrazon.predict_proba(df)[0][1]

    return jsonify({
        "enjambrazon_probable": bool(pred),
        "confianza": round(prob, 2)
    })

@app.route("/predecir-avispa", methods=["POST"])
def predecir_avispa():
    datos = request.get_json()
    df = pd.DataFrame([datos])
    pred = modelo_avispa.predict(df)[0]
    prob = modelo_avispa.predict_proba(df)[0][1]

    return jsonify({
        "riesgo_avispa": bool(pred),
        "confianza": round(prob, 2)
    })

@app.route("/predecir-alza-espacio", methods=["POST"])
def predecir_alza():
    datos = request.get_json()
    df = pd.DataFrame([datos])
    pred = modelo_alza_espacio.predict(df)[0]
    prob = modelo_alza_espacio.predict_proba(df)[0][1]

    return jsonify({
        "lista_para_alza": bool(pred),
        "confianza": round(prob, 2)
    })

@app.route("/predecir-miel", methods=["POST"])
def predecir_miel():
    datos = request.get_json()
    
    # Validar que venga la cantidad de cuadros
    if "cantidad_cuadros" not in datos:
        return jsonify({"error": "Falta el campo 'cantidad_cuadros'"}), 400
    
    try:
        cantidad = float(datos["cantidad_cuadros"])
    except ValueError:
        return jsonify({"error": "'cantidad_cuadros' debe ser un número"}), 400
    
    kilos_estimados = cantidad * PESO_PROMEDIO_POR_CUADRO
    
    return jsonify({
        "cantidad_cuadros": cantidad,
        "kilos_estimados": round(kilos_estimados, 2)
})