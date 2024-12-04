import requests
from models import load_yolo_model

detector = load_yolo_model()
url = "https://ikea-api.p.rapidapi.com/keywordSearch"

# querystring = {"keyword":"chair","countryCode":"us","languageCode":"en"}

headers = {
    "x-rapidapi-key": "TU_API_KEY",
    "x-rapidapi-host": "ikea-api.p.rapidapi.com"
}

# response = requests.get(url, headers=headers, params=querystring)

# print(response.json())

def search_furniture_ikea(keyword, country="us", language="en", n_results=5):
    params = {
        "keyword": keyword,
        "countryCode": country,
        "languageCode": language,
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            # Devuelve los primeros n resultados
            return response.json()[:n_results]
        else:
            return {"error": f"Error en la API: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}

    

def detect_furniture(image):
    results = detector(image)
    for result in results[0].boxes:
        label = result.cls
        label_number = int(label.item()) 
    print(results[0])
    print("-------------------")
    print(results)
    print("-------------------")
    print(label_number)
    print(f"el tipo de results es {type(results[0])}")
    print(f"el tipo de results es {type(results)}")
    names = results[0].names
    print(names)
    label_name = names[label_number]
    
    print(label_name)
    return label_name