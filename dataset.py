import os
import kagglehub

def load_dataset():
    # Descarga el dataset y retorna la ruta local
    dataset = kagglehub.dataset_download("robinreni/house-rooms-image-dataset")
    print("Path to dataset files:", dataset)
    return dataset

def load_images_from_folders(dataset_path):
    image_paths = []
    print(f"Listando contenido del directorio {dataset_path}:")
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        print(f"Revisando carpeta: {folder_path}")
        if os.path.isdir(folder_path):
            # Ahora recorremos las subcarpetas dentro de cada categoría
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                print(f"Revisando subcarpeta: {subfolder_path}")
                if os.path.isdir(subfolder_path):
                    for image_name in os.listdir(subfolder_path):
                        if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                            image_paths.append(os.path.join(subfolder_path, image_name))
    print(f"Total de imágenes encontradas: {len(image_paths)}")
    return image_paths
