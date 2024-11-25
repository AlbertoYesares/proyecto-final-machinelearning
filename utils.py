

from translate import Translator

def translate_text(text, src="es", dest="en"):
    try:
        translator = Translator(from_lang=src, to_lang=dest)
        return translator.translate(text)
    except Exception as e:
        print(f"Error al traducir: {e}")
        return None


def show_image(image):
    try:
        image.show()
    except Exception as e:
        print(f"Error al mostrar la imagen: {e}")
