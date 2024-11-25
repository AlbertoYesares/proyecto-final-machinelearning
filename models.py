from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

def load_clip_model():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def load_yolo_model():
    detector = YOLO("yolov8m.pt")
    return detector