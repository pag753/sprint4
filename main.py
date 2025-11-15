from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
# Importa la clase del modelo usado
from sklearn.ensemble import RandomForestClassifier 

# --- Rutas de Archivos ---
MODEL_PATH = 'modelo_bebidas.pkl'
COLUMNS_PATH = 'columnas_entrenamiento.pkl' 

app = FastAPI()

# --- 1. Carga de Recursos (Se ejecuta una sola vez al iniciar la API) ---
try:
    model = joblib.load(MODEL_PATH)
    COLUMNS = joblib.load(COLUMNS_PATH) # Lista de las 223 columnas de entrenamiento
    print(f"Modelo cargado. Esperando {len(COLUMNS)} features.")
except Exception as e:
    print(f"ERROR: Fallo al cargar el modelo o la lista de columnas: {e}")
    model = None
    COLUMNS = None

# --- 2. Lista de Columnas Categóricas Originales ---
# CRÍTICO: Debes listar aquí TODAS las columnas que fueron convertidas a OHE
# durante el entrenamiento (ej. deposit_type, market_segment, meal, customer_type, etc.)
# Si tu modelo predice 'bebidas' con el dataset de hoteles, estas son las columnas originales:
CATEGORICAL_COLS = [
    'hotel', 'meal', 'market_segment', 'distribution_channel', 
    'reserved_room_type', 'assigned_room_type', 'deposit_type', 
    'customer_type'
    # NOTA: Si usaste 'country' y tiene muchas categorías, el modelo lo espera. 
    # Asegúrate de incluirla si la entrenaste.
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
        # 1. Convertir JSON a DataFrame (13 features originales)
        input_df = pd.DataFrame([data])
        
        # 2. Preprocesamiento: Aplicar OHE a las categóricas
        # Usamos drop_first=True si lo usaste en el entrenamiento
        input_ohe = pd.get_dummies(
            input_df, 
            columns=CATEGORICAL_COLS, 
            drop_first=True
        )

        # 3. Alineación de Columnas (LA CLAVE para solucionar el error 13 vs 223)
        # Creamos un DataFrame con 0 en todas las 223 columnas esperadas
        final_input = pd.DataFrame(0, index=[0], columns=COLUMNS)
        
        # Copiamos los datos (numéricos y OHE) del input_ohe al final_input
        # Esto rellena las columnas existentes y deja el resto en 0.
        for col in input_ohe.columns:
            if col in final_input.columns:
                final_input[col] = input_ohe[col]
        
        # 4. Asegurar el orden (aunque el loop lo mantiene, es buena práctica)
        final_input = final_input[COLUMNS]

        # 5. Predicción (final_input ahora tiene las 223 features correctas)
        prediction = model.predict(final_input.values) 
        
        return {
            "predicted_beverage": str(prediction[0]), # Retorna la categoría
            "status": "success"
        }
    except Exception as e:
        # Devolvemos el error en caso de fallo en el preprocesamiento o predicción
        return {"Error": f"Fallo en la predicción: {str(e)}"}, 400