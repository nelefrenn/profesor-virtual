from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Habilitar CORS para permitir conexiones desde GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Aquí puedes especificar solo tu dominio de GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar datos
with open("tokens_oraciones.txt", "r", encoding="utf-8") as file:
    oraciones = file.readlines()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(oraciones)

# Función de búsqueda optimizada
def buscar_respuesta(consulta):
    consulta_tfidf = vectorizer.transform([consulta])
    similitudes = cosine_similarity(consulta_tfidf, tfidf_matrix)
    indices_ordenados = np.argsort(similitudes[0])[::-1]
    respuestas = [oraciones[i] for i in indices_ordenados[:3] if similitudes[0][i] > 0.4]
    return " ".join(respuestas) if respuestas else "No se encontró información específica."

@app.get("/buscar")
def obtener_respuesta(pregunta: str):
    return {"respuesta": buscar_respuesta(pregunta)}

