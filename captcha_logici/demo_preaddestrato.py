# Sonda Selenium che naviga alla pagina web demo, carica l'immagine 
# e, utilizzando il modello preaddestrato, ne riconosce il contenuto. 
# Scrive la previsione nella textbox e clicca il bottone per verificarne la correttezza

from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import time

counter = 0

# Calcola il risultato dell'espressione
def calculate_expression(digit1, digit2, operator):
    if operator == '+':
        return digit1 + digit2
    elif operator == '-':
        return digit1 - digit2
    elif operator == 'x' or operator == 'X':
        return digit1 * digit2
    elif operator == ':':
        return digit1 / digit2 if digit2 != 0 else 'Errore: divisione per zero'
    else:
        return 'Operatore sconosciuto'

# Carica il modello e il processore
model = VisionEncoderDecoderModel.from_pretrained('captcha_logici/modello_preaddestrato')
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    
while counter <= 2:

    # Inizializza Selenium
    driver = webdriver.Chrome() 
    driver.get("http://localhost:8000/captcha_logici/demo.html")

    # Trova l'immagine CAPTCHA sulla pagina
    captcha_image = driver.find_element(By.TAG_NAME, "img")

    # Ottieni l'immagine come byte array
    captcha_image_bytes = captcha_image.screenshot_as_png
    image = Image.open(BytesIO(captcha_image_bytes)).convert("RGB")

    # Prepara l'immagine per il modello
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Genera la predizione
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    result = calculate_expression(int(generated_text[0]), int(generated_text[2]), generated_text[1])

    # Inserisce il testo previsto nella textbox 
    textbox = driver.find_element(By.ID, "captcha_input") 
    textbox.send_keys(result)

    # Trova il pulsante di invio e lo clicca
    verify_button = driver.find_element(By.ID, "submit_button")  
    verify_button.click()

    time.sleep(3)
    driver.quit()
    
    counter += 1