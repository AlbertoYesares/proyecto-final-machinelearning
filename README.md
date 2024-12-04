# Proyecto Final Machine Learning - Buscador y Comparador de Imágenes de Interiores

Este proyecto es un sistema inteligente que utiliza modelos de Machine Learning para buscar y comparar muebles de interiores basándose en descripciones o imágenes. Ofrece recomendaciones de IKEA y genera imágenes de inspiración visual.

---

## Descripción del Proyecto

Este sistema permite:
1. **Búsqueda por texto:** Describe tu espacio ideal y el sistema encontrará muebles que coincidan con tu descripción.
2. **Búsqueda por imagen:** Sube una imagen y el sistema detectará muebles en ella, sugiriendo alternativas similares en IKEA.
3. **Inspiración visual:** Encuentra imágenes similares en un dataset local para complementar tu búsqueda.

**Modelos utilizados:**
- **YOLOv8:** Para detección de objetos en imágenes.
- **CLIP:** Para análisis de características visuales y textuales.
- **API de IKEA:** Para obtener información detallada de los muebles.

---

## Instalación

### Requisitos Previos
- **Python 3.8 o superior.**
- **Acceso a internet** para instalar las dependencias y usar la API de IKEA.

---

### Paso a Paso

1. **Clona este repositorio:**
   ```bash
   git clone https://github.com/AlbertoYesares/proyecto-final-machinelearning.git
   cd proyecto-final-machinelearning
   ```

2. **Instala las dependencias necesarias:**
   Instala todas las bibliotecas requeridas ejecutando:
   ```bash
   pip install -r requirements.txt
   ```

3. **Instala YOLOv8 (si no se instala con `requirements.txt`):**
   ```bash
   pip install ultralytics
   ```

4. **Configura la API de IKEA:**
   - Regístrate en [RapidAPI](https://rapidapi.com/) y obtén una API Key para IKEA.
    - Para ver todos los detalles de la API, consulta este link: (https://rapidapi.com/Octapi/api/ikea-api).
   - Agrega tu API Key al archivo `ikea_search.py` en la variable `x-rapidapi-key`:
     ```python
     headers = {
         "x-rapidapi-key": "TU_API_KEY",
         "x-rapidapi-host": "ikea-api.p.rapidapi.com"
     }
     ```

5. **Ejecuta el programa:**
   Dirígete al directorio del proyecto y ejecuta:
   ```bash
   python app/app.py
   ```

---

## Dependencias

El proyecto utiliza las siguientes bibliotecas y frameworks:

- `pandas`: Para la manipulación de datos.
- `numpy`: Para operaciones matemáticas avanzadas.
- `torch`: Framework de aprendizaje profundo.
- `transformers`: Para trabajar con el modelo CLIP.
- `torchvision`: Para la manipulación de imágenes.
- `pillow`: Para la gestión de imágenes.
- `datasets`: Para cargar y procesar datos.
- `ultralytics`: Para trabajar con YOLOv8.
- `gradio`: Para la interfaz gráfica.
- `scikit-learn`: Para cálculo de similitud de coseno.
- `requests`: Para consumir la API de IKEA.
- `translate`: Para traducción de texto (español a inglés).

Asegúrate de que estas bibliotecas están instaladas. Todas están listadas en el archivo `requirements.txt`.

### Contenido del archivo `requirements.txt`:
```
pandas
numpy
torch
transformers
torchvision
pillow
datasets
ultralytics
gradio
scikit-learn
requests
translate
```

Instálalas con:
```bash
pip install -r requirements.txt
```

---

## Uso del Proyecto

1. **Ejecución:**

   En local: 
      - Ve al directorio del proyecto y ejecuta:
        ```bash
        python app/app.py
        ```
      - Esto abrirá una interfaz de usuario en tu navegador a través de Gradio.

   Desde Docker:
      - Ve al directorio del proyecto y ejecuta:
        ```bash
        docker build -t gradio-app .
        docker run -p 7860:7860 gradio-app
        ```
      - Esto abrirá una interfaz de usuario en tu navegador a través de Docker en el puerto http://0.0.0.0:7860.
  
3. **Opciones de búsqueda:**
   - **Por descripción:** Introduce una descripción textual del espacio o mueble que buscas.
   - **Por imagen:** Sube una imagen para detectar los muebles presentes y recibir recomendaciones.

---

## Funcionalidades Principales

1. **Detección de muebles:**
   - Utiliza YOLOv8 para identificar muebles en las imágenes proporcionadas por el usuario.

2. **Recomendaciones inteligentes:**
   - Basado en características visuales (imágenes) o textuales (descripciones), el sistema sugiere muebles disponibles en IKEA.

3. **Inspiración visual:**
   - Busca imágenes similares en un dataset local para ofrecer inspiración adicional al usuario.

---

## Tecnologías Utilizadas

- **Python 3.8+**
- **YOLOv8:** Para detección de objetos en imágenes.
- **CLIP:** Para análisis visual y textual.
- **Gradio:** Para la interfaz gráfica de usuario.
- **API de IKEA:** Para obtener información de muebles.
- **Scikit-learn:** Para cálculo de similitudes.

---

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
```

### **Puntos importantes:**
- He listado todas las dependencias necesarias en el archivo `requirements.txt`.
- Incluí instrucciones detalladas para instalar y configurar todo, incluida la API de IKEA.
- Este archivo es completo y no deja nada fuera del flujo de instalación y uso.

Guarda este contenido como `README.md` en el directorio del proyecto. 😊
