
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
bash
   git clone https://github.com/AlbertoYesares/proyecto-final-machinelearning.git
   cd proyecto-final-machinelearning
   
2. **Instala las dependencias necesarias:**
   Ejecuta el siguiente comando para instalar las bibliotecas requeridas:
bash
   pip install -r requirements.txt
   
3. **Instala YOLOv8:**
   YOLOv8 es necesario para la detección de muebles en las imágenes. Instálalo ejecutando:
bash
   pip install ultralytics
   
4. **Configura la API de IKEA:**
   - Regístrate en [RapidAPI](https://rapidapi.com/) y obtén una API Key para IKEA.
   - Agrega tu API Key al archivo `ikea_search.py` en la variable `x-rapidapi-key`:
python
     headers = {
         "x-rapidapi-key": "TU_API_KEY",
         "x-rapidapi-host": "ikea-api.p.rapidapi.com"
     }
     
5. **Ejecuta el programa:**
   Dirígete al directorio del proyecto y ejecuta:
bash
   python main.py
   
---

## Dependencias

Asegúrate de que estas bibliotecas están instaladas antes de ejecutar el proyecto. Todas están listadas en el archivo `requirements.txt` y se pueden instalar con:
bash
pip install -r requirements.txt
### Contenido del archivo `requirements.txt`:

transformers
torchvision
pillow
datasets
ultralytics
gradio
- **transformers:** Para trabajar con el modelo CLIP.
- **torchvision:** Para manipular imágenes.
- **pillow:** Para la gestión de imágenes.
- **datasets:** Para cargar y procesar datos.
- **ultralytics:** Para YOLOv8.
- **gradio:** Para la interfaz gráfica de usuario.

---

## Uso del Proyecto

1. **Ejecución:**
   - Ve al directorio del proyecto y ejecuta:
bash
     python main.py
     
- Esto abrirá una interfaz de usuario en tu navegador a través de Gradio.

2. **Opciones de búsqueda:**
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

---

## Contribución

Si deseas contribuir al proyecto:
1. Haz un fork del repositorio.
2. Crea una nueva rama:
bash
   git checkout -b feature/nueva-funcionalidad
   
3. Realiza tus cambios y haz un commit:
bash
   git commit -m "Añadida nueva funcionalidad"
   
4. Haz un push de tus cambios:
bash
   git push origin feature/nueva-funcionalidad
   
5. Abre un Pull Request en el repositorio principal.

---

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---