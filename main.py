from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 

# --- Rutas de Archivos ---
MODEL_PATH = 'modelo_bebidas.pkl'
COLUMNS_PATH = 'columnas_entrenamiento.pkl' 

app = FastAPI()

try:
    model = joblib.load(MODEL_PATH)
    COLUMNS = joblib.load(COLUMNS_PATH)
    print(f"Modelo cargado. Esperando {len(COLUMNS)} features.")
except Exception as e:
    print(f"ERROR: Fallo al cargar el modelo o la lista de columnas: {e}")
    model = None
    COLUMNS = None
CATEGORICAL_COLS = [
    'hotel', 'meal', 'market_segment', 'distribution_channel', 
    'reserved_room_type', 'assigned_room_type', 'deposit_type', 
    'customer_type'
]


@app.get("/")
def home():
    """Endpoint de prueba para verificar que la API está viva."""
    return {"message": "API de predicción de bebidas lista."}


@app.post("/predict_beverage/")
async def predict_beverage(data: dict):
    """
    Recibe los datos originales de la reserva (ej. 13 features) y predice 
    la categoría de la bebida (o si cancelará, dependiendo del modelo).
    """
    if model is None or COLUMNS is None:
        return {"Error": "El modelo no se cargó correctamente al inicio."}, 500

    try:
        input_df = pd.DataFrame([data])
        input_ohe = pd.get_dummies(
            input_df, 
            columns=CATEGORICAL_COLS, 
            drop_first=True
        )
        final_input = pd.DataFrame(0, index=[0], columns=COLUMNS)

        for col in input_ohe.columns:
            if col in final_input.columns:
                final_input[col] = input_ohe[col]
        
        final_input = final_input[COLUMNS]

        prediction = model.predict(final_input.values) 
        
        return {
            "predicted_beverage": str(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        return {"Error": f"Fallo en la predicción: {str(e)}"}, 400