import pandas as pd
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ruta del directorio `models` relativo a este archivo
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Asegúrate de que el directorio `models` exista
os.makedirs(MODELS_DIR, exist_ok=True)

def initialize_csv(file_name="embeddings.csv"):
    # Ruta completa para guardar el archivo en `models`
    file_path = os.path.join(MODELS_DIR, file_name)
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["id", "embedding", "metadata"])
        df.to_csv(file_path, index=False)
    return file_path

def add_embeddings_to_csv(file_name="embeddings.csv", embeddings=None, image_paths=None):
    # Ruta completa al archivo CSV en el directorio `models`
    file_path = os.path.join(MODELS_DIR, file_name)
    df = pd.read_csv(file_path)
    base_path = os.getcwd()  # Directorio base del proyecto

    for idx, embedding in enumerate(embeddings):
        embedding_list = embedding.tolist()
        relative_path = os.path.relpath(image_paths[idx], start=base_path)  # Convertir a ruta relativa
        metadata = {"path": relative_path}
        new_row = pd.DataFrame({
            "id": [f"image_{idx}"],
            "embedding": [json.dumps(embedding_list)],
            "metadata": [json.dumps(metadata)]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv(file_path, index=False)
    print(f"Se han guardado {len(embeddings)} embeddings en el archivo {file_path}.")

def query_csv(file_name="embeddings.csv", text_embedding=None, n_results=1):
    # Ruta completa al archivo CSV en el directorio `models`
    file_path = os.path.join(MODELS_DIR, file_name)
    df = pd.read_csv(file_path)

    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x)))

    embeddings_matrix = np.stack(df["embedding"].to_numpy())
    text_embedding = text_embedding.numpy().reshape(1, -1)  
    similarities = cosine_similarity(text_embedding, embeddings_matrix).flatten()

    df["similarity"] = similarities
    df = df.sort_values(by="similarity", ascending=False)

    return df.head(n_results)
