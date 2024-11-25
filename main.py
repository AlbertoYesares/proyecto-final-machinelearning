import pandas as pd
from dataset import load_dataset, load_images_from_folders
from embeddings import get_dataset_embeddings
from utils import translate_text, show_image
from models import load_clip_model
from database import initialize_csv, add_embeddings_to_csv, query_csv
from ikea_search import *
import torch
from PIL import Image
import json
import os

model, processor = load_clip_model()

dataset_path = load_dataset()
image_paths = load_images_from_folders(dataset_path)

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

# Bucle principal
while True:
    user_input_espanol = input("Describe la habitación que buscas (o escribe 'salir' para terminar): ")
    if user_input_espanol.lower() == "salir":
        break

    # Traducir entrada del usuario
    user_input = translate_text(user_input_espanol)
    if not user_input:
        continue

    # Generar el embedding del texto
    inputs = processor(text=user_input, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).squeeze(0)

    # Consultar el archivo CSV
    results = query_csv(file_name, text_embedding, n_results=1)
    if not results.empty:
        # Asegúrate de cargar el valor de 'metadata' correctamente
        best_match_metadata = json.loads(results.iloc[0]["metadata"])
        best_match_path = best_match_metadata["path"]
        
        # Verificar si la imagen existe antes de intentar abrirla
        if os.path.exists(best_match_path):
            print(f"La imagen más relevante es: {best_match_path}")
            image = Image.open(best_match_path)
            show_image(image)
            furniture = detect_furniture(image)
            # label_detected_furniture = furniture[0]["label"]
            mueble_encontrado = search_furniture_ikea(furniture)
            # print(label_detected_furniture)
            print(mueble_encontrado)
            
        else:
            print(f"No se encontró la imagen en la ruta: {best_match_path}")
    else:
        print("No se encontró una imagen relevante.")

