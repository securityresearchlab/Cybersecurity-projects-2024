from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import time
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import skimage.filters as filters

counter = 0

while counter <= 2:

    # Carica il modello
    model_caricato = load_model('captcha_logici/modello_logici.h5')

    # Definizione delle etichette per cifre e operatori
    LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    OPERATORS = ['+', '-', 'x', ':']
    INPUT_SHAPE = (50, 200, 1)

    # Configurazione di Selenium WebDriver
    def initialize_webdriver(url):
        driver = webdriver.Chrome()
        driver.get(url)
        return driver

    # Acquisisce l'immagine dalla pagina web e la converte in un array NumPy
    def capture_captcha_image(driver):
        captcha_image_element = driver.find_element(By.TAG_NAME, "img")
        captcha_image_bytes = captcha_image_element.screenshot_as_png
        image = Image.open(BytesIO(captcha_image_bytes)).convert("RGB")
        img = np.array(image)
        return img

    # Preprocessing dell'immagine
    def prepare_image(img, input_shape):
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = 255 - cv2.fastNlMeansDenoising(img)
        smooth = cv2.medianBlur(img, 5)
        smooth = cv2.GaussianBlur(smooth, (5, 5), 0)
        division = cv2.divide(img, smooth, scale=255)
        sharp = filters.unsharp_mask(division, radius=1.5, amount=1.5)
        sharp = (255 * sharp).clip(0, 255).astype(np.uint8)
        return np.array(sharp / 255.0, dtype=np.float32).reshape(input_shape)

    # Calcola il risultato dell'espressione
    def calculate_expression(digit1, digit2, operator):
        if operator == '+':
            return digit1 + digit2
        elif operator == '-':
            return digit1 - digit2
        elif operator in ['x', 'X']:
            return digit1 * digit2
        elif operator == ':':
            return digit1 / digit2 if digit2 != 0 else 'Errore: divisione per zero'
        else:
            return 'Operatore sconosciuto'
        
    # Prevede il contenuto dell'immagine e invia la risposta 
    def predict_and_submit_captcha(model, driver, img, input_shape):
        preprocessed_img = prepare_image(img, input_shape)
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        predictions = model.predict(preprocessed_img)

        digit1 = np.argmax(predictions[0])
        digit2 = np.argmax(predictions[1])
        operator = np.argmax(predictions[2])

        predicted_digit1 = LABELS[digit1]
        predicted_digit2 = LABELS[digit2]
        predicted_operator = OPERATORS[operator]
        
        result = calculate_expression(int(predicted_digit1), int(predicted_digit2), predicted_operator)

        textbox = driver.find_element(By.ID, "captcha_input")
        textbox.clear()
        textbox.send_keys(result)

        verify_button = driver.find_element(By.ID, "submit_button")
        verify_button.click()

    def main():
        url = "http://localhost:8000/captcha_logici/demo.html"
        driver = initialize_webdriver(url)

        try:
            img = capture_captcha_image(driver)
            predict_and_submit_captcha(model_caricato, driver, img, INPUT_SHAPE)
            time.sleep(3)
        except Exception as e:
            print(f"Errore: {e}")
        finally:
            driver.quit()

    if __name__ == "__main__":
        main()
        
    counter += 1