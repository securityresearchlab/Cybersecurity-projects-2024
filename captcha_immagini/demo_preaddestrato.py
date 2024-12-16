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
import numpy as np
import cv2

counter = 0

# costanti
url = "http://localhost:8000/captcha_immagini/demo_yolo.html"
cfg_path = 'captcha_immagini/yolo/yolov3.cfg'
weights_path = 'captcha_immagini/yolo/yolov3.weights'
classes_path = 'captcha_immagini/yolo/coco.names'

#Carica il modello YOLOv3, i pesi e le classi
def load_yolo_model(cfg_path, weights_path, classes_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    with open(classes_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return net, classes

#Esegue la rilevazione degli oggetti utilizzando YOLO
def detect_objects_yolo(img, net, classes):
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(ln)

    h, w = img.shape[:2]
    boxes, confidences, classIDs = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    return classIDs, classes
    
while counter <= 2:

    # Inizializza Selenium
    driver = webdriver.Chrome() 
    driver.get(url)

    time.sleep(1)
    
    # Trova l'immagine CAPTCHA sulla pagina
    captcha_image = driver.find_element(By.TAG_NAME, "img")

    # Ottiene l'immagine come byte array
    captcha_image = captcha_image.screenshot_as_png

    # Usa PIL per aprire l'immagine in memoria
    image = Image.open(BytesIO(captcha_image)).convert("RGB")

    # Converte l'immagine da PIL a formato numpy array leggibile da OpenCV
    img = np.array(image)

    # Cambia il formato del colore da RGB a BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Prepara l'immagine per il modello
    classes = open(classes_path).read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # Carica il modello YOLO
    net, classes = load_yolo_model(cfg_path, weights_path, classes_path)

    # Rileva gli oggetti nell'immagine
    classIDs, classes = detect_objects_yolo(img, net, classes)
            
    # Assicura che almeno un'immagina è stata trovata
    if len(classIDs) > 0:
        first_object_name = classes[classIDs[0]]
        
        # adatta l'etichetta al modello in caso di "idrante"
        if first_object_name == "fire hydrant":
            first_object_name = "hydrant"

        # Inserisce la previsione nella textbox
        textbox = driver.find_element(By.ID, "captcha_input") 
        textbox.send_keys(first_object_name)

        # Trova il pulsante di invio e clicca
        verify_button = driver.find_element(By.ID, "submit_button")  
        verify_button.click()

    
    time.sleep(3)
    driver.quit()
    
    counter += 1
