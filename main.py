from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from googletrans import Translator
import kagglehub
import os

# Crear el traductor
translator = Translator()

# Cargar el modelo y procesador CLIP
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Cargar el dataset de Kaggle usando kagglehub
dataset = kagglehub.dataset_download("robinreni/house-rooms-image-dataset")
print("Path to dataset files:", dataset)

# Función para cargar imágenes desde las carpetas dentro del dataset
def load_images_from_folders(dataset_path):
    image_paths = []
    print(f"Listando contenido del directorio {dataset_path}:")  # Añadido para depuración
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        print(f"Revisando carpeta: {folder_path}")  # Añadido para depuración
        if os.path.isdir(folder_path):
            # Ahora recorremos las subcarpetas dentro de cada categoría
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                print(f"Revisando subcarpeta: {subfolder_path}")  # Añadido para depuración
                if os.path.isdir(subfolder_path):
                    for image_name in os.listdir(subfolder_path):
                        if image_name.lower().endswith((".jpg", ".jpeg", ".png")):  # Aceptar más formatos
                            image_paths.append(os.path.join(subfolder_path, image_name))
    return image_paths


# Obtener las rutas de las imágenes
image_paths = load_images_from_folders(dataset)

# Obtener embeddings de las imágenes del dataset
def get_dataset_embeddings(image_paths):
    image_embeddings = []
    images = []

    for image_path in image_paths:
        try:
            # Abrir la imagen
            image = Image.open(image_path)

            # Procesar la imagen con CLIP
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)

            image_embeddings.append(embedding)
            images.append(image)  # Guardar la imagen en vez de 'unknown'
            print(f"Imagen procesada correctamente: {image_path}")  # Mensaje de depuración
        except Exception as e:
            print(f"Error procesando la imagen {image_path}: {e}")
            continue
    
    if len(image_embeddings) == 0:
        print("No se generaron embeddings para ninguna imagen.")
    return torch.cat(image_embeddings) if image_embeddings else None, images

# Función para buscar la imagen más relevante para un texto
def find_best_match(text, image_embeddings, images):
    # Generar el embedding del texto
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)

    # Calcular similitudes (producto punto)
    similarities = torch.matmul(text_embedding, image_embeddings.T)
    best_match_idx = similarities.argmax().item()
    
    return images[best_match_idx]  # Devolver la imagen directamente

# Función para mostrar la imagen (en terminal, usa PIL)
def show_image(image):
    try:
        image.show()  # Esto abrirá la imagen en el visor de imágenes predeterminado
    except Exception as e:
        print(f"Error al mostrar la imagen: {e}")

# Generar embeddings para el dataset
print("Procesando imágenes del dataset...")
image_embeddings, images = get_dataset_embeddings(image_paths)

if image_embeddings is None:
    print("Error: No se pudieron generar los embeddings.")
else:
    # Bucle para ingresar texto por terminal
    while True:
        user_input_espanol = input("Describe la habitación que buscas (o escribe 'salir' para terminar): ")
        if user_input_espanol.lower() == "salir":
            break

        try:
            # Traducir la entrada del usuario al inglés
            user_input = translator.translate(user_input_espanol, src='es', dest='en').text
        except Exception as e:
            print(f"Error al traducir: {e}")
            continue

        try:
            # Buscar la mejor imagen
            best_match = find_best_match(user_input, image_embeddings, images)
            print(f"La imagen más relevante es:")

            # Mostrar la imagen
            show_image(best_match)
        except Exception as e:
            print(f"Error al procesar la solicitud: {e}")
