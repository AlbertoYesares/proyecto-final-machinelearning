import pandas as pd
from dataset import load_dataset, load_images_from_folders
from embeddings import get_dataset_embeddings
from utils import translate_text, show_image
from models import load_clip_model
from database import initialize_csv, add_embeddings_to_csv, query_csv
from ikea_search import *
import gradio as gr
import torch
from PIL import Image
import json
import os

# Cargar modelo y procesador
model, processor = load_clip_model()

# Cargar dataset
dataset_path = load_dataset()
image_paths = load_images_from_folders(dataset_path)

# Inicializar CSV
file_name = "embeddings.csv"
initialize_csv(file_name)

df = pd.read_csv(file_name)

if df.empty:
    print("No se encontraron embeddings en el archivo CSV. Procesando...")
    image_embeddings = get_dataset_embeddings(image_paths, processor, model)
    if image_embeddings:
        add_embeddings_to_csv(file_name, image_embeddings, image_paths)
else:
    print("Embeddings cargados desde el archivo CSV.")

# Función principal para la interfaz
def search_furniture(description):
    # Traducir entrada del usuario
    user_input = translate_text(description)
    if not user_input:
        return "Error en la traducción.", None, None, None, None

    # Generar el embedding del texto
    inputs = processor(text=user_input, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).squeeze(0)

    # Consultar el archivo CSV
    results = query_csv(file_name, text_embedding, n_results=1)
    if results.empty:
        return "No se encontró una imagen relevante.", None, None, None, None

    # Procesar resultados
    best_match_metadata = json.loads(results.iloc[0]["metadata"])
    best_match_path = best_match_metadata["path"]

    # Verificar si la imagen existe
    if not os.path.exists(best_match_path):
        return f"No se encontró la imagen en la ruta: {best_match_path}", None, None, None, None

    # Detectar mueble en la imagen
    image = Image.open(best_match_path)
    furniture = detect_furniture(image)
    if not furniture:
        return "No se detectó ningún mueble.", None, None, None, None

    # Buscar mueble en IKEA
    mueble_encontrado = search_furniture_ikea(furniture)
    if not mueble_encontrado:
        return "No se encontró información sobre el mueble en IKEA.", None, None, None, None

    # Extraer detalles del mueble encontrado
    ikea_data = mueble_encontrado[0]
    ikea_image = ikea_data["image"]
    ikea_name = ikea_data["name"]
    ikea_price = f"{ikea_data['price']['currency']} {ikea_data['price']['currentPrice']}"
    ikea_url = ikea_data["url"]

    # Devolver resultados
    return ikea_name, ikea_price, ikea_url, best_match_path, ikea_image

# Crear interfaz con Gradio
def app_interface(description):
    ikea_name, ikea_price, ikea_url, local_image_path, ikea_image_url = search_furniture(description)
    if not ikea_name:
        return "No se encontraron resultados.", None, None, None
    return ikea_name, ikea_price, ikea_url, (local_image_path, ikea_image_url)

interface = gr.Interface(
    fn=app_interface,
    inputs=gr.Textbox(label="Describe la habitación que buscas:"),
    outputs=[
        gr.Textbox(label="Nombre del mueble"),
        gr.Textbox(label="Precio"),
        gr.Textbox(label="Link a IKEA"),
        gr.Gallery(label="Imágenes (Local vs IKEA)"),
    ],
    title="Búsqueda de Muebles en IKEA",
    description="Introduce una descripción de tu habitación ideal y encuentra muebles relacionados en IKEA.",
)

# Ejecutar la app
if __name__ == "__main__":
    interface.launch(allowed_paths=["/Users/albertoyesaresmartinez/.cache/kagglehub/datasets"])

