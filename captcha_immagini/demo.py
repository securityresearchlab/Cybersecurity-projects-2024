# Sonda Selenium che naviga alla pagina web demo, carica l'immagine 
# e, utilizzando il modello creato e addestrato da noi, ne riconosce il contenuto. 
# Scrive la previsione nella textbox e clicca il bottone per verificarne la correttezza

import time
import base64
import io
import numpy as np
import requests
import cv2
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from tensorflow.keras.models import load_model

counter = 0

while counter <= 2:

    # Funzione per configurare Selenium WebDriver
    def configure_webdriver(url):
        driver = webdriver.Chrome()
        driver.get(url)
        time.sleep(3)  # Attendere il caricamento della pagina
        return driver

    # Funzione per caricare il modello di deep learning
    def load_trained_model(model_path):
        model = load_model(model_path)
        model.compile(metrics=['accuracy'])
        return model

    # Funzione per preprocessare l'immagine per il modello
    def preprocess_image(img):
        # Converti l'immagine in formato array NumPy (OpenCV la carica in BGR)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converti BGR a RGB
        img = cv2.resize(img, (120, 120))  # Ridimensiona l'immagine a 120x120
        img = np.expand_dims(img, axis=0)  # Aggiungi una dimensione batch
        return img

    # Funzione per caricare un'immagine da una URL
    def load_image_from_url(url):
        response = requests.get(url)
        img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return img

    # Funzione per acquisire e decodificare l'immagine del CAPTCHA
    def get_captcha_image(driver):
        captcha_element = driver.find_element(By.TAG_NAME, "img")
        captcha_src = captcha_element.get_attribute("src")
        if "base64," in captcha_src:
            img_data = captcha_src.split("base64,")[1]
            image = Image.open(io.BytesIO(base64.b64decode(img_data)))
            return image
        else:
            return load_image_from_url(captcha_src)

    # Funzione per fare una previsione con il modello
    def predict_captcha(model, input_image, class_labels):
        prediction = model.predict(input_image)
        predicted_label_index = np.argmax(prediction)
        predicted_label_text = class_labels[predicted_label_index]
        return predicted_label_text

    # Funzione per inserire il risultato nella textbox e inviare il CAPTCHA
    def submit_captcha_prediction(driver, predicted_text):
        try:
            textbox = driver.find_element(By.ID, "captcha_input")
            textbox.clear()
            textbox.send_keys(predicted_text)
            submit_button = driver.find_element(By.ID, "submit_button")
            submit_button.click()
        except Exception as e:
            print(f"Errore nell'inserimento del testo o nella pressione del pulsante: {e}")

    def main():
        url = "http://localhost:8000/captcha_immagini/demo.html" 
        model_path = "captcha_immagini/modello_immagini.h5"
        class_labels = ["car", "bus", "crosswalk", "palm", "hydrant", "bicycle", "traffic light", "motorcycle", "bridge", "chimney", "stair"]

        # Configurare il WebDriver
        driver = configure_webdriver(url)
    
        try:
            # Caricare il modello di deep learning
            model = load_trained_model(model_path)
            
            # Acquisire l'immagine del CAPTCHA
            image = get_captcha_image(driver)
            
            # Preprocessare l'immagine
            input_image = preprocess_image(image)
            
            # Fare una previsione con il modello
            predicted_label_text = predict_captcha(model, input_image, class_labels)
            
            # Inserire la previsione e inviare il modulo
            submit_captcha_prediction(driver, predicted_label_text)
            
            # Attendere per osservare il risultato
            time.sleep(3)
        
        except Exception as e:
            print(f"Errore: {e}")
        
        finally:
            driver.quit()  

    if __name__ == "__main__":
        main()
    
    counter += 1