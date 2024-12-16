# Script per la costruzione di un modello 
# in grado di analizzare e decifrare captcha logici
# con equazioni matematiche

import numpy as np
import cv2
import skimage.filters as filters
import os
import random
import re
import glob
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset_path = "captcha_logici/equation"
files = os.listdir(dataset_path)

file_to_remove = "9+9+_339b26e88adab3e3181b70d780647017.jpg"
if file_to_remove in files:
    files.remove(file_to_remove)
    print(f"File rimosso: {file_to_remove}")  
else:
    print(f"Il file {file_to_remove} non è presente nella lista.")  

print(files[:5])

INPUT_SHAPE = (50, 200, 1)
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
OPERATORS = ['+', '-', 'x', ':']
OPERATOR_SIZE = len(OPERATORS)
OUTPUT_SIZE = len(LABELS)

def prepare_image(img):

    img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    # se l'immagine è a colori, viene convertita in scala di grigi
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #riduzione del rumore
    img = 255 - cv2.fastNlMeansDenoising(img)

    smooth = cv2.medianBlur(img, 5)
    smooth = cv2.GaussianBlur(smooth, (5, 5), 0)
    #aumento del contrasto e della nitidezza
    division = cv2.divide(img, smooth, scale=255)
    sharp = filters.unsharp_mask(division, radius=1.5, amount=1.5, channel_axis=False, preserve_range=False)
    sharp = (255 * sharp).clip(0, 255).astype(np.uint8)

    # normalizza e ridimensionamento per l'input del modello
    return np.array(sharp / 255.0, dtype=np.float32).reshape(INPUT_SHAPE)



def build_model():
    input = Input(INPUT_SHAPE)
    out = input

    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPool2D(pool_size=(2, 2))(out)

    out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPool2D(pool_size=(2, 2))(out)

    out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPool2D(pool_size=(2, 2))(out)

    out = Flatten()(out)
    out = Dense(1024, activation='relu')(out)
    out = Dropout(0.5)(out)  
    out = [
        Dense(OUTPUT_SIZE, name='digit1', activation='softmax')(out),
        Dense(OUTPUT_SIZE, name='digit2', activation='softmax')(out),
        Dense(OPERATOR_SIZE, name='operator', activation='softmax')(out)
    ]

    model = Model(inputs=input, outputs=out)

    return model


def get_model():
      return build_model()
  

if __name__ == '__main__':
    model = get_model()
    summary = model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print("Il diagramma del modello è stato salvato come 'model_plot.png'")
    print(summary)


def load_images(files, dataset_path):
    images = []
    labels_digit1 = []  
    labels_digit2 = []  
    labels_operator = []  

    for file in files:
        file_path = os.path.join(dataset_path, file)

        img = cv2.imread(file_path)
        if img is None:
            print(f"Immagine non valida: {file_path}")  
            continue 

        preprocessed_img = prepare_image(img)  # preprocessing personalizzato
        images.append(preprocessed_img)

        # estrazione espressione
        match = re.match(r'([^=]+)', file)  
        if match:
            equation = match.group(1)  
            digit1 = equation[0]
            operator = equation[1].lower()  
            digit2 = equation[2]
            if digit1 in LABELS and digit2 in LABELS and operator in OPERATORS:
                labels_digit1.append(LABELS.index(digit1))
                labels_digit2.append(LABELS.index(digit2))
                labels_operator.append(OPERATORS.index(operator))
            else:
                print(f"Caratteri non validi nell'immagine {file_path}: {digit1}, {operator}, {digit2}")  
        else:
            print(f"Immagine {file_path} non ha una corrispondenza con l'equazione.")  

    if len(images) != len(labels_digit1) or len(images) != len(labels_digit2) or len(images) != len(labels_operator):
        raise ValueError("Le lunghezze degli array delle immagini e delle etichette non coincidono.")

    images = np.array(images)
    labels_digit1 = np.array(labels_digit1)
    labels_digit2 = np.array(labels_digit2)
    labels_operator = np.array(labels_operator)
    labels_digit1 = to_categorical(labels_digit1, num_classes=OUTPUT_SIZE)
    labels_digit2 = to_categorical(labels_digit2, num_classes=OUTPUT_SIZE)
    labels_operator = to_categorical(labels_operator, num_classes=OPERATOR_SIZE)

    # divisione in train, validation e test (80% training, 10% validation, 10% test)
    # training (80%) e resto X_temp(20%)
    X_train, X_temp, y_train_digit1, y_temp_digit1 = train_test_split(images, labels_digit1, test_size=0.2, random_state=42)
    X_train, X_temp, y_train_digit2, y_temp_digit2 = train_test_split(images, labels_digit2, test_size=0.2, random_state=42)
    X_train, X_temp, y_train_operator, y_temp_operator = train_test_split(images, labels_operator, test_size=0.2, random_state=42)

    # 20% rimanente (X_temp) in validazione (50% del 20%, quindi 10% del totale) e test (50% del 20%, quindi 10% del totale)
    X_val, X_test, y_val_digit1, y_test_digit1 = train_test_split(X_temp, y_temp_digit1, test_size=0.5, random_state=42)
    X_val, X_test, y_val_digit2, y_test_digit2 = train_test_split(X_temp, y_temp_digit2, test_size=0.5, random_state=42)
    X_val, X_test, y_val_operator, y_test_operator = train_test_split(X_temp, y_temp_operator, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train_digit1, y_val_digit1, y_test_digit1, y_train_digit2, y_val_digit2, y_test_digit2, y_train_operator, y_val_operator, y_test_operator


def train(x_train, y_train_digit1, y_train_digit2, y_train_operator, x_val, y_val_digit1, y_val_digit2, y_val_operator, batch_size=32, epochs=10):
    
    model = get_model()  

    model.compile(
        optimizer='adam',
        loss={
            'digit1': 'categorical_crossentropy',
            'digit2': 'categorical_crossentropy',
            'operator': 'categorical_crossentropy'
        },
        metrics={
            'digit1': 'accuracy',
            'digit2': 'accuracy',
            'operator': 'accuracy'
        }
    )

    history = model.fit(
        x_train,
        [y_train_digit1, y_train_digit2, y_train_operator],  
        validation_data=(x_val, [y_val_digit1, y_val_digit2, y_val_operator]),  
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=1,
    )

    return history

# caricamento immagini e etichette
X_train, X_val, X_test, y_train_digit1, y_val_digit1, y_test_digit1, y_train_digit2, y_val_digit2, y_test_digit2, y_train_operator, y_val_operator, y_test_operator = load_images(files, dataset_path)
history = train(X_train, y_train_digit1, y_train_digit2, y_train_operator, X_val, y_val_digit1, y_val_digit2, y_val_operator, batch_size=32, epochs=10)


def plot_history(history):
    plt.figure(figsize=(15, 8))

    plt.subplot(3, 2, 1)
    plt.plot(history.history['digit1_accuracy'], label='Accuracy (train digit1)')
    plt.plot(history.history['val_digit1_accuracy'], label='Accuracy (validation digit1)')
    plt.title('Accuracy per Digit1 durante l\'addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.subplot(3, 2, 2)
    plt.plot(history.history['digit2_accuracy'], label='Accuracy (train digit2)')
    plt.plot(history.history['val_digit2_accuracy'], label='Accuracy (validation digit2)')
    plt.title('Accuracy per Digit2 durante l\'addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(history.history['operator_accuracy'], label='Accuracy (train operator)')
    plt.plot(history.history['val_operator_accuracy'], label='Accuracy (validation operator)')
    plt.title('Accuracy per Operator durante l\'addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(history.history['loss'], label='Loss (train totale)')
    plt.plot(history.history['val_loss'], label='Loss (validation totale)')
    plt.title('Loss totale durante l\'addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout() 
plot_history(history)

#predizione test set

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
OPERATORS = ['+', '-', 'x', ':']
INPUT_SHAPE = (50, 200, 1)  

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

def predict_on_test(model, X_test):
    for idx, img in enumerate(X_test):

        plt.figure()
        plt.imshow(img.squeeze(), cmap='gray')  
        plt.title(f"Immagine {idx + 1}")
        plt.axis('off')
        plt.show()

        # previsione sull'immagine
        preprocessed_img = np.expand_dims(img, axis=0)  
        predictions = model.predict(preprocessed_img)

        pred_digit1 = np.argmax(predictions[0])
        pred_digit2 = np.argmax(predictions[1])
        pred_operator = np.argmax(predictions[2]) 

        print(f"Previsione per Digit1: {LABELS[pred_digit1]}")
        print(f"Previsione per Digit2: {LABELS[pred_digit2]}")
        print(f"Previsione per Operatore: {OPERATORS[pred_operator]}")

        # calcolo il risultato dell'espressione
        result = calculate_expression(int(LABELS[pred_digit1]), int(LABELS[pred_digit2]), OPERATORS[pred_operator])
        print(f"Risultato dell'espressione: {result}")
        print("-" * 50) 



predict_on_test(history.model, X_test)

def evaluate_model_on_test(model, X_test, y_test_digit1, y_test_digit2, y_test_operator):
    predictions = model.predict(X_test)

    accuracy_digit1 = accuracy_score(y_test_digit1.argmax(axis=1), predictions[0].argmax(axis=1))
    accuracy_digit2 = accuracy_score(y_test_digit2.argmax(axis=1), predictions[1].argmax(axis=1))
    accuracy_operator = accuracy_score(y_test_operator.argmax(axis=1), predictions[2].argmax(axis=1))
    
    # media delle accuratezze
    mean_accuracy = (accuracy_digit1 + accuracy_digit2 + accuracy_operator) / 3
    
    print(f'Accuracy: {mean_accuracy * 100:.2f}%')


evaluate_model_on_test(history.model, X_test, y_test_digit1, y_test_digit2, y_test_operator)
