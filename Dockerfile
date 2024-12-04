# Usa la imagen base de Python
FROM python:3.10-slim

# Define el directorio de trabajo
WORKDIR /usr/src/app

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Instala dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias de Python necesarias para el proyecto
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir gradio ultralytics transformers scikit-learn translate opencv-python-headless

# Expone el puerto 7860 para Gradio
EXPOSE 7860

# Establece la dirección del servidor Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Comando por defecto para ejecutar la aplicación
CMD ["python", "app/app.py"]
