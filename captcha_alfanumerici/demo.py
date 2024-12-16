# Sonda Selenium che naviga alla pagina web demo, carica l'immagine 
# e, utilizzando il modello creato e addestrato da noi, ne riconosce il contenuto. 
# Scrive la previsione nella textbox e clicca il bottone per verificarne la correttezza

import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
from pathlib import Path
from tensorflow.keras import layers

counter = 0

while counter <= 2:

    # Definizione del layer CTC
    class LayerCTC(tf.keras.layers.Layer):
        def __init__(self, name=None, **kwargs):
            super().__init__(name=name, **kwargs)
            self.loss_fn = tf.keras.backend.ctc_batch_cost

        def call(self, y_true, y_pred):
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

            loss = self.loss_fn(y_true, y_pred, input_length, label_length)
            self.add_loss(loss)

            return y_pred

    # Preprocessing dell'immagine
    def preprocess_image(image):
        image = image.convert("L")  # Converte in scala di grigi
        image = image.resize((200, 50))  # Ridimensiona l'immagine
        image_array = np.array(image) / 255.0  # Normalizza i valori dei pixel
        image_array = np.expand_dims(image_array, axis=-1)  # Aggiunge un canale
        image_array = np.expand_dims(image_array, axis=0)  # Aggiunge una dimensione batch
        image_array = np.transpose(image_array, (0, 2, 1, 3))  # Traspone l'immagine per il modello
        return image_array

    # Funzione per decodificare le predizioni
    def decode_single_prediction(pred, max_length, num_to_char):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    # Caricamento e configurazione del modello
    def load_model_and_char_mappings(model_path, samples_path, max_length):
        # Carica le immagini per definire le funzioni di codifica e decodifica delle etichette
        dir_img = sorted([str(img) for img in Path(samples_path).glob("*.png") if len(os.path.basename(img).split(".png")[0]) == max_length])
        img_labels = [os.path.basename(img).split(".png")[0] for img in dir_img]
        char_set = sorted(set(char for label in img_labels for char in label))

        # Crea le funzioni conversione
        char_to_num = layers.StringLookup(vocabulary=list(char_set), mask_token=None)
        num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

        # Carica il modello
        model = keras.models.load_model(model_path, custom_objects={'LayerCTC': LayerCTC})
        prediction_model = keras.models.Model(inputs=model.input[0], outputs=model.get_layer(name="dense2").output)
        
        return prediction_model, num_to_char

    # Configurazione di Selenium WebDriver
    def configure_webdriver(url):
        driver = webdriver.Chrome()
        driver.get(url)
        return driver

    def main():
        url = "http://localhost:8000/captcha_alfanumerici/demo.html"
        model_path = 'captcha_alfanumerici/modello_alfanumerici.h5'
        samples_path = "captcha_alfanumerici/samples"
        max_length = 5

        # Carica il modello e le mappature dei caratteri
        prediction_model, num_to_char = load_model_and_char_mappings(model_path, samples_path, max_length)
        
        # Configura il WebDriver
        driver = configure_webdriver(url)

        try:
            # Trova l'immagine CAPTCHA
            captcha_image = driver.find_element(By.TAG_NAME, "img")
            captcha_image_bytes = captcha_image.screenshot_as_png
            image = Image.open(BytesIO(captcha_image_bytes))

            # Preprocessa l'immagine
            image_array = preprocess_image(image)

            # Effettua la previsione
            pred = prediction_model.predict(image_array)
            pred_text = decode_single_prediction(pred, max_length, num_to_char)

            # Inserisce la previsione nella textbox e invia
            textbox = driver.find_element(By.ID, "captcha_input")
            textbox.clear()
            textbox.send_keys(pred_text[0])
            driver.find_element(By.ID, "submit_button").click()

            time.sleep(3)
        except Exception as e:
            print(f"Errore: {e}")
        finally:
            driver.quit()

    if __name__ == "__main__":
        main()
    
    counter += 1