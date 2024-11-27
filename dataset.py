import os

def load_dataset():
    # Ruta relativa al dataset
    dataset_path = os.path.join(os.getcwd(), "House_Room_Dataset")
    print("Ruta del dataset:", dataset_path)
    return dataset_path


def load_images_from_folders(dataset_path):
    image_paths = []
    print(f"Listando contenido del directorio {dataset_path}:")
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        print(f"Revisando carpeta: {folder_path}")
        if os.path.isdir(folder_path):  # Confirmamos que es una carpeta
            # Ahora recorremos los archivos directamente dentro de la carpeta
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if image_name.lower().endswith((".jpg", ".jpeg", ".png")):  # Confirmamos que es una imagen
                    image_paths.append(image_path)
    print(f"Total de imágenes encontradas: {len(image_paths)}")
    return image_paths

