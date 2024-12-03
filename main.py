import pandas as pd
from dataset import load_dataset, load_images_from_folders
from embeddings import get_dataset_embeddings
from utils import translate_text
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

# Función para procesar descripción (texto)
def search_furniture(description):
    user_input = translate_text(description)
    if not user_input:
        return "Error en la traducción.", None  # Devuelve solo dos valores

    inputs = processor(text=user_input, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).squeeze(0)

    results = query_csv(file_name, text_embedding, n_results=1)
    if results.empty:
        return "No se encontró una imagen relevante.", None  # Devuelve solo dos valores

    base_path = os.getcwd()
    best_match_metadata = json.loads(results.iloc[0]["metadata"])
    relative_path = best_match_metadata["path"]
    best_match_path = os.path.join(base_path, relative_path)

    if not os.path.exists(best_match_path):
        return f"No se encontró la imagen en la ruta: {best_match_path}", None  # Devuelve solo dos valores

    image = Image.open(best_match_path)
    furniture = detect_furniture(image)
    if not furniture:
        return "No se detectó ningún mueble.", None  # Devuelve solo dos valores

    muebles_encontrados = search_furniture_ikea(furniture, n_results=5)
    if not muebles_encontrados or "error" in muebles_encontrados:
        return "No se encontraron muebles en IKEA.", None  # Devuelve solo dos valores

    muebles_tabla = []
    for mueble in muebles_encontrados:
        muebles_tabla.append({
            "name": mueble["name"],
            "price": f"{mueble['price']['currency']} {mueble['price']['currentPrice']}",
            "url": mueble["url"]
        })

    return best_match_path, muebles_tabla  # Siempre dos valores



# Función para procesar imagen
def process_uploaded_image(image):
    # Detectar mueble en la imagen
    furniture = detect_furniture(image)
    if not furniture:
        return "No se detectó ningún mueble en la imagen.", None

    # Buscar muebles similares en IKEA
    muebles_encontrados = search_furniture_ikea(furniture, n_results=5)
    if not muebles_encontrados or "error" in muebles_encontrados:
        return "No se encontraron muebles en IKEA.", None

    muebles_tabla = []
    for mueble in muebles_encontrados:
        muebles_tabla.append({
            "name": mueble["name"],
            "price": f"{mueble['price']['currency']} {mueble['price']['currentPrice']}",
            "url": mueble["url"]
        })

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs).squeeze(0)

    results = query_csv(file_name, image_embedding, n_results=1)
    if results.empty:
        return None, muebles_tabla

    base_path = os.getcwd()
    best_match_metadata = json.loads(results.iloc[0]["metadata"])
    relative_path = best_match_metadata["path"]
    best_match_path = os.path.join(base_path, relative_path)

    return best_match_path, muebles_tabla



# Crear interfaz con Gradio
def app_interface(description, uploaded_image):
    if description and uploaded_image is None:
        local_image_path, muebles_tabla = search_furniture(description)  # Espera dos valores
        print(f"Resultados de search_furniture: {search_furniture(description)}")
    elif uploaded_image is not None:
        local_image_path, muebles_tabla = process_uploaded_image(uploaded_image)  # También espera dos valores
    else:
        return "Por favor, proporciona una descripción o sube una imagen.", None, None

    if not local_image_path:
        return "No se encontraron resultados.", None, None

    muebles_table_html = """
    <table style="width:100%; text-align:left; border-collapse: collapse; border: 1px solid black;">
        <tr>
            <th style="border: 1px solid black; padding: 8px;">Nombre</th>
            <th style="border: 1px solid black; padding: 8px;">Precio</th>
            <th style="border: 1px solid black; padding: 8px;">Enlace</th>
        </tr>
    """
    for mueble in muebles_tabla:
        muebles_table_html += f"""
        <tr>
            <td style="border: 1px solid black; padding: 8px;">{mueble['name']}</td>
            <td style="border: 1px solid black; padding: 8px;">{mueble['price']}</td>
            <td style="border: 1px solid black; padding: 8px;"><a href="{mueble['url']}" target="_blank">Ver en IKEA</a></td>
        </tr>
        """
    muebles_table_html += "</table>"

    return local_image_path, muebles_table_html





# Diseño personalizado
css = """
#title {
    font-size: 2.5em;
    color: #4A90E2;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}

#description {
    font-size: 1.2em;
    color: #555;
    text-align: center;
    margin-bottom: 40px;
}
"""

interface = gr.Interface(
    fn=app_interface,
    inputs=[
        gr.Textbox(
            label="Describe la habitación que buscas:",
            placeholder="Ejemplo: Un dormitorio acogedor con muebles minimalistas"
        ),
        gr.Image(
            label="Sube una imagen de referencia (opcional):",
            type="pil"  # Asegura que la imagen sea procesada como un objeto PIL
        ),
    ],
    outputs=[
        gr.Image(label="Imagen más relevante encontrada:"),
        gr.HTML(label="Muebles recomendados:"),
    ],
    title="<div id='title'>Búsqueda Inteligente de Muebles IKEA</div>",
    description="<div id='description'>Proporciona una descripción o sube una imagen para encontrar muebles relacionados en IKEA.</div>",
    css=css,
)



# Ejecutar la app
if __name__ == "__main__":
    interface.launch(
        allowed_paths=["/Users/albertoyesaresmartinez/.cache/kagglehub/datasets"]
    )
