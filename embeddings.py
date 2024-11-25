import torch
from PIL import Image

def get_dataset_embeddings(image_paths, processor, model):
    image_embeddings = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)

            # embedding
            image_embeddings.append(embedding.squeeze(0))  # Convertir a 1D tensor
        except Exception as e:
            print(f"Error procesando la imagen {image_path}: {e}")
            continue
    return image_embeddings


def find_best_match(text, image_embeddings, images, processor, model):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)

    # Calcular similitudes
    similarities = torch.matmul(text_embedding, image_embeddings.T)
    best_match_idx = similarities.argmax().item()
    return images[best_match_idx]
