from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import random

# 🔹 **Solución definitiva para descargar 'punkt' en Render**
nltk_data_path = "/opt/render/nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Configurar NLTK para usar esa carpeta
nltk.data.path.append(nltk_data_path)

# Descargar los recursos necesarios de NLTK en Render
nltk.download('punkt', download_dir=nltk_data_path)

app = FastAPI()

# 🔹 **Corrección definitiva de CORS**
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir TODAS las conexiones (necesario para GitHub Pages)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, OPTIONS)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# 🔹 **CATEGORÍAS DISPONIBLES**
categorias = {
    "Historia de la gestión en la salud pública": [],
    "Contexto, tendencias y retos de la Salud Pública": [],
    "Causalidad y prevención": [],
    "Proceso salud-enfermedad-atención": [],
    "Sistema general de seguridad social integral": [],
}

# 🔹 **Cargar datos y clasificarlos en categorías**
with open("tokens_oraciones.txt", "r", encoding="utf-8") as file:
    oraciones = file.readlines()

oraciones = [oracion.strip() for oracion in oraciones if oracion.strip()]

for oracion in oraciones:
    for categoria in categorias.keys():
        if categoria.lower() in oracion.lower():
            categorias[categoria].append(oracion)

# 🔹 **Crear un vectorizador y matrices TF-IDF por categoría**
vectorizadores = {}
tfidf_matrices = {}

for categoria, textos in categorias.items():
    if textos:
        vectorizador = TfidfVectorizer()
        matriz_tfidf = vectorizador.fit_transform(textos)
        vectorizadores[categoria] = vectorizador
        tfidf_matrices[categoria] = matriz_tfidf

# 🔹 **Función para generar respuestas basadas en la categoría seleccionada**
def generar_respuesta(consulta, categoria):
    if categoria not in categorias or not categorias[categoria]:
        return "No se encontraron datos en la categoría seleccionada."

    vectorizador = vectorizadores[categoria]
    matriz_tfidf = tfidf_matrices[categoria]
    consulta_tfidf = vectorizador.transform([consulta])
    similitudes = cosine_similarity(consulta_tfidf, matriz_tfidf)
    indices_ordenados = np.argsort(similitudes[0])[::-1]

    # Obtener respuestas más relevantes dentro de la categoría
    respuestas = [categorias[categoria][i] for i in indices_ordenados[:3] if similitudes[0][i] > 0.4]

    if not respuestas:
        return "No se encontró información específica sobre tu consulta en esta categoría."

    texto_unido = " ".join(respuestas)
    frases = nltk.sent_tokenize(texto_unido)
    random.shuffle(frases)
    respuesta_final = " ".join(frases[:2])

    return respuesta_final

# 🔹 **Endpoint de consulta con selección de categoría**
@app.get("/buscar")
def obtener_respuesta(pregunta: str, categoria: str = Query(..., description="Selecciona una categoría")):
    respuesta = generar_respuesta(pregunta, categoria)
    return JSONResponse(content={"respuesta": respuesta}, headers={"Access-Control-Allow-Origin": "*"})

# 🔹 **Ruta de prueba para verificar que el servidor está activo**
@app.get("/")
def home():
    return JSONResponse(content={"mensaje": "El Profesor Virtual de Salud Pública está en línea."}, headers={"Access-Control-Allow-Origin": "*"})

    return {"mensaje": "El Profesor Virtual de Salud Pública está en línea."}


