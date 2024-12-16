#Script per la costruzione di un modello 
# in grado di analizzare e decifrare captcha alfanumerici 

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# costanti
IMG_WIDTH = 200
IMG_HEIGHT = 50
BATCH_SIZE = 16
DOWNSAMPLE_FACTOR = 4

# setup delle directory
dataset_path = Path("captcha_alfanumerici/samples")
image_paths = sorted([str(img) for img in dataset_path.glob("*.png") if len(img.stem) == 5])
labels = [Path(img).stem for img in image_paths]

# processing dei caratteri
unique_characters = sorted(list(set(char for label in labels for char in label)))
char_to_num = layers.StringLookup(vocabulary=unique_characters, mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
max_label_length = max([len(label) for label in labels])

# codifica immagini e etichette
def encode_image_and_label(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label}

# splitting
def split_data(images, labels, train_ratio=0.8, val_ratio=0.1):
    size = len(images)
    indices = np.arange(size)
    np.random.shuffle(indices)
    train_size = int(size * train_ratio)
    val_size = int(size * val_ratio)
    x_train, y_train = images[indices[:train_size]], labels[indices[:train_size]]
    x_val, y_val = images[indices[train_size:train_size + val_size]], labels[indices[train_size:train_size + val_size]]
    x_test, y_test = images[indices[train_size + val_size:]], labels[indices[train_size + val_size:]]
    return x_train, y_train, x_val, y_val, x_test, y_test

x_train, y_train, x_val, y_val, x_test, y_test = split_data(np.array(image_paths), np.array(labels))

# creazione oggetti dataset
def create_dataset(images, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(encode_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(x_train, y_train, BATCH_SIZE)
val_dataset = create_dataset(x_val, y_val, BATCH_SIZE)
test_dataset = create_dataset(x_test, y_test, BATCH_SIZE)

# funzione per la costruzione del modello
def build_ocr_model():
    input_img = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    new_shape = ((IMG_WIDTH // 4), (IMG_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)
    
    output = LayerCTC(name="ctc_loss")(labels, x)
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
    model.compile(optimizer=keras.optimizers.Adam())
    return model

# Layer CTC Loss 
class LayerCTC(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

# costruzione e rappresentazione del modello
model = build_ocr_model()
model.summary()

# Training
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[early_stopping])

# modello di predizione
prediction_model = keras.models.Model(inputs=model.input[0], outputs=model.get_layer(name="dense2").output)
prediction_model.summary()

# funzione per la decodifica delle predizioni
def decode_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_label_length]
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

# test del modello
def evaluate_model(test_data):
    for batch in test_data.take(1):
        images = batch["image"]
        labels = batch["label"]
        predictions = prediction_model.predict(images)
        pred_texts = decode_predictions(predictions)

        orig_texts = []
        for label in labels:
            text = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(text)

        accuracy = sum(1 for pred, orig in zip(pred_texts, orig_texts) if pred == orig) / len(orig_texts)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        
        for i in range(min(len(pred_texts), 16)):
            img = (batch["image"][i, :, :, 0] * 255).numpy().astype(np.uint8)  # Converti l'immagine per la visualizzazione
            img = img.T  # Trasponi per visualizzare correttamente
            
            plt.figure(figsize=(5, 5))
            plt.imshow(img, cmap="gray")
            plt.title(f"Prediction: {pred_texts[i]}\nOriginal: {orig_texts[i]}\nAccuracy: {'Correct' if pred_texts[i] == orig_texts[i] else 'Incorrect'}")
            plt.axis("off")
            plt.show()


evaluate_model(test_dataset)

# salvataggio modello allenato
model.save('modello_alfanumerici.h5')

# grafico andamento loss e val_loss
def plot_training_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

plot_training_history(history)