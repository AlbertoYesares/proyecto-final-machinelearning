import pandas as pd
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def initialize_csv(file_name="embeddings.csv"):
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=["id", "embedding", "metadata"])
        df.to_csv(file_name, index=False)
    return file_name

def add_embeddings_to_csv(file_name, embeddings, image_paths):
    df = pd.read_csv(file_name)
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
    
    df.to_csv(file_name, index=False)
    print(f"Se han guardado {len(embeddings)} embeddings en el archivo CSV.")


def query_csv(file_name, text_embedding, n_results=1):
    df = pd.read_csv(file_name)

    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x)))

    embeddings_matrix = np.stack(df["embedding"].to_numpy())
    text_embedding = text_embedding.numpy().reshape(1, -1)  
    similarities = cosine_similarity(text_embedding, embeddings_matrix).flatten()

    df["similarity"] = similarities
    df = df.sort_values(by="similarity", ascending=False)

    return df.head(n_results)
