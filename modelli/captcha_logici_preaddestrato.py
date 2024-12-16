# Script per l'allenamento di un modello preaddestrato
# in grado di analizzare e decifrare captcha logici
# con equazioni matematiche

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
from transformers import default_data_collator
import logging
import matplotlib.pyplot as plt

dataset_path = "captcha_logici/equation"

files = os.listdir(dataset_path)

file_to_remove = "9+9+_339b26e88adab3e3181b70d780647017.jpg"
if file_to_remove in files:
    files.remove(file_to_remove)
    print(f"File rimosso: {file_to_remove}") 
else:
    print(f"Il file {file_to_remove} non è presente nella lista.")  


print(files[:5])

df = pd.DataFrame(files, columns=['file_name'])
df['text'] = df['file_name'].str.split('=').str[0]
df

# divisione del dataset
train_df, temp_df = train_test_split(df, test_size=0.2)
train_df.reset_index(drop=True, inplace=True)
temp_df.reset_index(drop=True, inplace=True)
validation_df, test_df = train_test_split(temp_df, test_size=0.5)
validation_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

class CAPTCHADataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=20):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # gestione dei file non validi
        if file_name == "samples" or not file_name.endswith(('.png', '.jpg')):
            return None
        if not isinstance(text, str):
            return None

        try:
            # caricamento e preparazione dell'immagine
            image_path = os.path.join(self.root_dir, file_name)
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            # tokenizzazione del testo
            labels = self.processor.tokenizer(text,
                                              padding="max_length",
                                              max_length=self.max_target_length).input_ids
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
            return encoding
        except Exception as e:
            print(f"Errore durante il caricamento dell'immagine o tokenizzazione del testo per {file_name}: {e}")
            return None


root_dir = os.path.join(path, 'equation')+'/'

print(f"Percorso al dataset: {root_dir}")
if not os.path.exists(root_dir):
    print(f"ERRORE: Il percorso {root_dir} non esiste!")
else:
    print(f"Il percorso {root_dir} è stato trovato.")


try:
    files = os.listdir(root_dir)
    print("Contenuti del dataset:", files[:10]) 
except Exception as e:
    print(f"Impossibile leggere il contenuto del percorso {root_dir}: {e}")

# inizializzazione di modello e processore
model_checkpoint = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_checkpoint)

try:
    train_dataset = CAPTCHADataset(root_dir=root_dir, df=train_df, processor=processor)
    eval_dataset = CAPTCHADataset(root_dir=root_dir, df=validation_df, processor=processor)
    test_dataset = CAPTCHADataset(root_dir=root_dir, df=test_df, processor=processor)
    print("I dataset sono stati creati correttamente.")
except Exception as e:
    print(f"Errore nella creazione del dataset: {e}")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# impostazione dei token speciali e dei parametri del modello
model.config.update({
    'decoder_start_token_id': processor.tokenizer.cls_token_id,
    'pad_token_id': processor.tokenizer.pad_token_id,
    'eos_token_id': processor.tokenizer.sep_token_id,
    'vocab_size': model.config.decoder.vocab_size,
    'max_length': 20,
    'early_stopping': True,
    'length_penalty': 2.0,
    'num_beams': 2
})

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=True,
    logging_steps=2,
    save_strategy="epoch",
    num_train_epochs=3,
    report_to="none",
)

cer_metric = evaluate.load("cer", trust_remote_code=True)

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    accuracy = 1 - cer

    return {"cer": cer, "accuracy": accuracy}

logging.getLogger("transformers").setLevel(logging.ERROR)


def custom_collate_fn(batch):

    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {"pixel_values": pixel_values, "labels": labels}

#addestramento
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
     data_collator=custom_collate_fn

)
training_args.per_device_train_batch_size = 4
trainer.train()

model_save_path = "/content/modello_OCR_num"  
# salvataggio del modello
trainer.model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)

model = VisionEncoderDecoderModel.from_pretrained(model_save_path)

pred_list = []

for i in range(len(test_df)):

    image = Image.open(test_dataset.root_dir + test_df['file_name'][i]).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # genera la predizione
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pred_list.append(generated_text)


    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.axis('off')  
    plt.title(f"True Text: {test_df['text'][i]}\nPredicted Text: {generated_text}")
    plt.show()


    print(i, test_df['text'][i], generated_text)

accuracy = (test_df['text'] == pd.Series(pred_list)).mean()
print(f'Accuracy: {accuracy * 100:.2f}%')

