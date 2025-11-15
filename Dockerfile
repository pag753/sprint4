# Dockerfile (Revisión)
FROM python:3.10-slim

WORKDIR /app

# Copia e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia los archivos de la aplicación y el modelo
COPY main.py .
COPY modelo_bebidas.pkl .
COPY columnas_entrenamiento.pkl . 

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]