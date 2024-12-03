import os
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

def load_clip_model():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def load_yolo_model():
    # Ruta al archivo yolov8m.pt dentro de la carpeta models
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolov8m.pt")
    detector = YOLO(model_path)
    return detector
