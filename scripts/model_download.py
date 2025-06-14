import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "plguillou/t5-base-fr-sum-cnndm"
local_dir = "./resources/models/plguillou/t5-base-fr-sum-cnndm"

os.makedirs(local_dir, exist_ok=True)

print(f"Téléchargement du modèle {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print(f"Sauvegarde du modèle dans {local_dir}...")

tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print("Modèle téléchargé et sauvegardé avec succès.")

