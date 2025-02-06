from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Habilitar CORS para permitir solicitudes desde GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nelefrenn.github.io"],  # Cambia esto por tu dominio en GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar datos desde el archivo de oraciones
with open("tokens_oraciones.txt", "r", encoding="utf-8") as file:
    oraciones = file.readlines()

# Limpiar oraciones y eliminar saltos de línea
oraciones = [oracion.strip() for oracion in oraciones if oracion.strip()]

# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(oraciones)

# Función para buscar respuestas con similitud del coseno
def buscar_respuesta(consulta):
    consulta_tfidf = vectorizer.transform([consulta])
    similitudes = cosine_similarity(consulta_tfidf, tfidf_matrix)
    indices_ordenados = np.argsort(similitudes[0])[::-1]  # Ordenar de mayor a menor similitud

    # Seleccionar las mejores respuestas (top 3 con relevancia mayor a 40%)
    respuestas = [oraciones[i] for i in indices_ordenados[:3] if similitudes[0][i] > 0.4]

    return " ".join(respuestas) if respuestas else "No se encontró información específica sobre tu consulta."

# Endpoint para realizar consultas
@app.get("/buscar")
def obtener_respuesta(pregunta: str):
    return {"respuesta": buscar_respuesta(pregunta)}

# Ruta de prueba para verificar si el servidor está activo
@app.get("/")
def home():
    return {"mensaje": "El Profesor Virtual de Salud Pública está en línea."}
