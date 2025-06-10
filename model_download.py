import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "moussaKam/barthez-orangesum-abstract"
local_dir = "./model/barthez-orangesum-abstract"

os.makedirs(local_dir, exist_ok=True)

print(f"Téléchargement du modèle {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Correction manuelle du GenerationConfig AVANT la sauvegarde
gen_config = model.generation_config

# Patch si nécessaire
if getattr(gen_config, "early_stopping", False) and getattr(gen_config, "num_beams", 1) <= 1:
    print("Correction de la configuration de génération invalide...")
    gen_config.early_stopping = False

# Supprimer early_stopping de la config principale pour éviter warning
if hasattr(model.config, "early_stopping"):
    delattr(model.config, "early_stopping")

print(f"Sauvegarde du modèle dans {local_dir}...")

# Sauvegardes manuelles
tokenizer.save_pretrained(local_dir)
model.config.save_pretrained(local_dir)
model.model.save_pretrained(local_dir)  # <-- important : sauvegarde uniquement les poids

# Sauvegarde manuelle de la generation config
with open(os.path.join(local_dir, "generation_config.json"), "w") as f:
    json.dump(gen_config.to_dict(), f, indent=2)

print("Modèle téléchargé et sauvegardé avec succès.")
