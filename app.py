from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import random
import traceback  # Para capturar errores en los logs

# 🔹 **Solución definitiva para descargar 'punkt' y otros modelos en Render**
nltk_data_path = "/opt/render/nltk_data"
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Configurar NLTK para usar esa carpeta
nltk.data.path.append(nltk_data_path)

# 🔹 Descargar explícitamente los modelos necesarios en español
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

app = FastAPI()

# 🔹 **Corrección definitiva de CORS**
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir TODAS las conexiones
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin"],  # Asegura que el navegador reciba la cabecera
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

    respuestas = [categorias[categoria][i] for i in indices_ordenados[:3] if similitudes[0][i] > 0.4]

    if not respuestas:
        return "No se encontró información específica sobre tu consulta en esta categoría."

    texto_unido = " ".join(respuestas)

    # 🔹 **Corrección: Especificar idioma en sent_tokenize**
    frases = nltk.tokenize.sent_tokenize(texto_unido, language="spanish")
    
    random.shuffle(frases)
    respuesta_final = " ".join(frases[:2])

    return respuesta_final

# 🔹 **Endpoint de consulta con selección de categoría y detección de errores**
@app.get("/buscar")
def obtener_respuesta(pregunta: str, categoria: str = Query(..., description="Selecciona una categoría")):
    try:
        respuesta = generar_respuesta(pregunta, categoria)
        return JSONResponse(content={"respuesta": respuesta}, headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        error_detalles = traceback.format_exc()
        print(error_detalles)  # 🔹 Esto imprimirá el error en los logs de Render
        return JSONResponse(content={"error": str(e), "detalles": error_detalles}, status_code=500, headers={"Access-Control-Allow-Origin": "*"})

# 🔹 **Ruta de prueba para verificar que el servidor está activo**
@app.get("/")
def home():
    return JSONResponse(content={"mensaje": "El Profesor Virtual de Salud Pública está en línea."}, headers={"Access-Control-Allow-Origin": "*"})



