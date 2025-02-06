from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import random

nltk.download('punkt')

app = FastAPI()

# Habilitar CORS para permitir solicitudes desde GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nelefrenn.github.io"],  # Reemplázalo con tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar datos desde el archivo de oraciones
with open("tokens_oraciones.txt", "r", encoding="utf-8") as file:
    oraciones = file.readlines()

# Limpiar oraciones y eliminar saltos de línea
oraciones = [oracion.strip() for oracion in oraciones if oracion.strip()]

# Tokenización en palabras y frases
tokens_palabras = [nltk.word_tokenize(oracion) for oracion in oraciones]

# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(oraciones)

# Función para generar una respuesta coherente
def generar_respuesta(consulta):
    consulta_tfidf = vectorizer.transform([consulta])
    similitudes = cosine_similarity(consulta_tfidf, tfidf_matrix)
    indices_ordenados = np.argsort(similitudes[0])[::-1]  # Ordenar de mayor a menor similitud

    # Obtener las oraciones más relevantes
    respuestas = [oraciones[i] for i in indices_ordenados[:3] if similitudes[0][i] > 0.4]

    if not respuestas:
        return "No se encontró información específica sobre tu consulta."

    # Generar respuesta coherente combinando la información encontrada
    texto_unido = " ".join(respuestas)

    # Dividir en frases para mejorar fluidez
    frases = nltk.sent_tokenize(texto_unido)
    random.shuffle(frases)  # Mezclar un poco para evitar repeticiones

    respuesta_final = " ".join(frases[:2])  # Seleccionamos dos frases clave para dar coherencia

    return respuesta_final

# Endpoint para consultas
@app.get("/buscar")
def obtener_respuesta(pregunta: str):
    return {"respuesta": generar_respuesta(pregunta)}

# Ruta de prueba para verificar si el servidor está activo
@app.get("/")
def home():
    return {"mensaje": "El Profesor Virtual de Salud Pública está en línea."}

