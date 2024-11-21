from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from googletrans import Translator

# Crear el traductor
translator = Translator()

# Cargar el modelo y procesador CLIP
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Cargar el dataset de Hugging Face
dataset_name = "straxico/rooms"
dataset = load_dataset(dataset_name, split="train")

# Obtener embeddings de las imágenes del dataset
def get_dataset_embeddings(dataset):
    image_embeddings = []
    images = []

    for example in dataset:
        try:
            # Usar directamente el objeto de imagen desde la columna 'image'
            image = example["image"]  # Imagen cargada como objeto PIL
            
            # Procesar la imagen con CLIP
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)

            image_embeddings.append(embedding)
            images.append(image)  # Guardar la imagen en vez de 'unknown'
        except Exception as e:
            print(f"Error procesando la imagen: {e}")
            continue
    
    return torch.cat(image_embeddings), images

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
image_embeddings, images = get_dataset_embeddings(dataset)

# Bucle para ingresar texto por terminal
while True:
    user_input_espanol = input("Describe la habitación que buscas (o escribe 'salir' para terminar): ")
    if user_input_espanol.lower() == "salir":
        break
    # Traducir la entrada del usuario al inglés
    user_input = translator.translate(user_input_espanol, src='es', dest='en').text

    try:
        # Buscar la mejor imagen
        best_match = find_best_match(user_input, image_embeddings, images)
        print(f"La imagen más relevante es:")

        # Mostrar la imagen
        show_image(best_match)
    except Exception as e:
        print(f"Error al procesar la solicitud: {e}")
