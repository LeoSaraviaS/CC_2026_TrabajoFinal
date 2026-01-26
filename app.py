from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import os

# Cargar el modelo y el scaler
# Intentamos cargar desde el directorio actual
path_modelo = 'modelo_regresion_logistica.pkl'
path_scaler = 'scaler.pkl'

# Validar existencia (opcional, ayuda a depurar)
if not os.path.exists(path_modelo):
    # Si no esta en actual, probar en ../lab/ si es que estamos en trabajo_final
    potential_path = '../lab/modelo_regresion_logistica.pkl'
    if os.path.exists(potential_path):
        path_modelo = potential_path
        path_scaler = '../lab/scaler.pkl'

try:
    with open(path_modelo, 'rb') as archivo_modelo:
        modelo = pickle.load(archivo_modelo)

    with open(path_scaler, 'rb') as archivo_scaler:
        scaler = pickle.load(archivo_scaler)
except FileNotFoundError:
    # Esto pasará si no ejecutan Model_02 antes o no mueven los archivos
    print("ADVERTENCIA: No se encontraron los archivos .pkl. La API fallará al predecir.")
    modelo = None
    scaler = None

# Definir las características esperadas por el modelo (Orden Importante)
columnas_modelo = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Crear la aplicación FastAPI
app = FastAPI(title="Predicción de Supervivencia Titanic")

# Definir el modelo de datos de entrada utilizando Pydantic
class Pasajero(BaseModel):
    Pclass: int       # 1, 2, 3
    Sex: str          # male, female
    Age: float        # ej. 22.5
    SibSp: int        # Hermanos/Cónyuges (ej. 1)
    Parch: int        # Padres/Hijos (ej. 0)
    Fare: float       # Tarifa (ej. 50.0)
    Embarked: str     # C, Q, S

# Definir el endpoint para predicción
@app.post("/predict/")
async def predecir_supervivencia(pasajero: Pasajero):
    if modelo is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Ejecuta Model_02.py primero.")

    try:
        # Preprocesamiento de variables categóricas
        sex_map = {'male': 0, 'female': 1, 'hombre': 0, 'mujer': 1}
        embarked_map = {'S': 0, 'C': 1, 'Q': 2}
        
        sex_val = sex_map.get(pasajero.Sex.lower())
        if sex_val is None:
             raise HTTPException(status_code=400, detail="Sex debe ser 'male' o 'female'")
             
        embarked_val = embarked_map.get(pasajero.Embarked.upper())
        if embarked_val is None:
             raise HTTPException(status_code=400, detail="Embarked debe ser 'S', 'C' o 'Q'")

        # Crear lista de valores en el orden correcto
        datos_lista = [[
            pasajero.Pclass,
            sex_val,
            pasajero.Age,
            pasajero.SibSp,
            pasajero.Parch,
            pasajero.Fare,
            embarked_val
        ]]

        # Convertir a DataFrame
        datos_entrada = pd.DataFrame(datos_lista, columns=columnas_modelo)
        
        # Escalar
        datos_entrada_scaled = scaler.transform(datos_entrada)
        
        # Predecir
        prediccion = modelo.predict(datos_entrada_scaled)
        probabilidad = modelo.predict_proba(datos_entrada_scaled)[:, 1]
        
        # Respuesta
        resultado = {
            "Sobrevive": bool(prediccion[0] == 1),
            "ProbabilidadSupervivencia": float(probabilidad[0]),
            "Mensaje": "El pasajero SOBREVIVE" if prediccion[0] == 1 else "El pasajero NO sobrevive"
        }
        
        return resultado
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

