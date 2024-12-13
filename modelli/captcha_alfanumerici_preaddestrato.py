# Script per l'allenamento di un modello preaddestrato
# in grado di analizzare e decifrare captcha alfanumerici 

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import matplotlib.pyplot as plt
from transformers import default_data_collator
import logging
import evaluate


dataset_path = "captcha_alfanumerici/samples"
files = os.listdir(dataset_path)

print(files[:5])

df = pd.DataFrame(files, columns=['file_name'])
df['text'] = df['file_name'].str.split('.').str[0]
df

# divisione in train, validation e test
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
            # caricamento e preparazione immagine
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

root_dir = 'captcha_alfanumerici/samples/'

print(f"Percorso al dataset: {root_dir}")
if not os.path.exists(root_dir):
    print(f"ERRORE: Il percorso {root_dir} non esiste!")
else:
    print(f"Il percorso {root_dir} è stato trovato.")

try:
    files = os.listdir(root_dir)
    print("Contenuti del dataset:", files[:10])  # Mostra solo i primi 10 file per evitare troppi dati
except Exception as e:
    print(f"Impossibile leggere il contenuto del percorso {root_dir}: {e}")

# inizializzazione di processore e modello
model_checkpoint = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_checkpoint)

# creazione dataset
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
    'max_length': 5,
    'early_stopping': True,
    'length_penalty': 2.0,
    'num_beams': 5
})

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    logging_steps=2,
    save_strategy="epoch",
    num_train_epochs=7,
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

# addestramento
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
     data_collator=custom_collate_fn

)

trainer.train()

model_save_path = "/content/modello_OCR"  
#salvataggio 
trainer.model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)

model = VisionEncoderDecoderModel.from_pretrained(model_save_path)

pred_list = []
accuracy_valid_indices = []  
filtered_temp_df = test_df[~test_df['file_name'].str.contains("samples")]
for i in range(len(filtered_temp_df)):

    image = Image.open(test_dataset.root_dir + filtered_temp_df['file_name'].iloc[i]).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    true_text = filtered_temp_df['text'].iloc[i]
    if not generated_text.strip() or true_text[:-1] == generated_text: 
        continue 
    pred_list.append(generated_text)
    accuracy_valid_indices.append(filtered_temp_df.index[i])

    # visualizzazzione immagine e testo
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off') 
    plt.title(
        f"True Text: {true_text}\nPredicted Text: {generated_text}"
    )
    plt.show()

    print(i, true_text, generated_text)

pred_series = pd.Series(pred_list, index=accuracy_valid_indices)
accuracy = (filtered_temp_df.loc[accuracy_valid_indices, 'text'] == pred_series).mean()
print(f'Accuracy: {accuracy * 100:.2f}%')
