import os
import json
import sqlite3
import hashlib
import faiss
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
import Levenshtein
from langdetect import detect


# === INITIALISATION ===

# --- Chargement de la configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
with open(os.path.join(PROJECT_ROOT, "resources", "config.json")) as f:
    raw_config = json.load(f)
def normalize_path(value):
    if isinstance(value, str):
        if value.startswith("~") or os.path.isabs(value):
            return os.path.normpath(os.path.expanduser(value))
        else:
            return os.path.normpath(os.path.join(PROJECT_ROOT, value))
    return value

config = {key: normalize_path(value) for key, value in raw_config.items()}

DB_PATH = config["db_path"]
FOLDER_PATH = os.path.expanduser(config["lmstudio_folder_path"])
EXTENSIONS = ['.json']
stopwords_path = config["stopwords_file_path"]
with open(stopwords_path, "r", encoding="utf-8") as f:
    french_stopwords = set(json.load(f))

def clock():
    return datetime.now().strftime("[%H:%M:%S]")

# === Modèle et connection à la base ===
kw_model = KeyBERT()
nlp_fr = spacy.load("fr_core_news_lg")
nlp_en = spacy.load("en_core_web_lg")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Index vectoriel
VECTOR_DIM = 384
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# Création de la table si elle n'existe pas avec timestamp ajouté
cur.execute('''
    CREATE TABLE IF NOT EXISTS hash_index (
        hash TEXT PRIMARY KEY
    )
''')

cur.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT NOT NULL,
        llm_model TEXT NOT NULL,
        llm_output TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

cur.execute('''
    CREATE TABLE IF NOT EXISTS vectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER,
        keyword TEXT,
        vector BLOB,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    )
''')

conn.commit()

# === EXTRACTION MOTS-CLÉS ET VECTEURS ===
combined_stopwords = ENGLISH_STOP_WORDS.union(french_stopwords)

def lemmatize_spacy(word, lang="fr"):
    if lang == "fr":
        doc = nlp_fr(word)
    elif lang == "en":
        doc = nlp_en(word)
    else:
        # Par défaut, français
        doc = nlp_fr(word)
    return doc[0].lemma_.lower()


def lemmatize_auto(word):
    lang = detect(word)
    if lang.startswith("fr"):
        return lemmatize_spacy(word, lang="fr")
    elif lang.startswith("en"):
        return lemmatize_spacy(word, lang="en")
    else:
        return word.lower()


def extract_keywords(text, top_n=25, similarity_threshold=0.50):
    raw_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words=list(combined_stopwords),
        top_n=top_n * 3
    )
    seen = set()
    filtered_keywords = []
    for kw, _ in raw_keywords:
        kw_clean = kw.lower().strip()
        kw_clean = lemmatize_spacy(kw_clean)
        if (
            kw_clean not in seen and
            kw_clean not in combined_stopwords and
            len(kw_clean) > 2 and
            re.match(r'^[a-zA-Z\-]+$', kw_clean)
        ):
             # filtrage manuel pluriels
            if kw_clean.endswith('s') and kw_clean[:-1] in seen:
                continue
            if any(Levenshtein.distance(kw_clean, k) <= 1 for k in seen):
                continue
            seen.add(kw_clean)
            filtered_keywords.append(kw_clean)

    embeddings = embedding_model.encode(filtered_keywords, convert_to_tensor=True)

    selected = []
    for i, emb in enumerate(embeddings):
        if len(selected) == 0:
            selected.append((filtered_keywords[i], emb))
        else:
            similarities = util.cos_sim(emb, torch.stack([e[1] for e in selected]))
            if torch.max(similarities).item() < similarity_threshold:
                selected.append((filtered_keywords[i], emb))
        if len(selected) >= top_n:
            break

    return [(kw, emb.cpu().numpy().tolist()) for kw, emb in selected]

# === INSÉRER CONVERSATION AVEC DÉDOUBLONNAGE ===
def insert_conversation_if_new(user_input, llm_output, llm_model):
    combined = user_input + llm_output
    hash_digest = hashlib.md5(combined.encode('utf-8')).hexdigest()

    cur.execute("SELECT 1 FROM hash_index WHERE hash = ?", (hash_digest,))
    if cur.fetchone():
        return False

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO conversations (user_input, llm_output, llm_model, timestamp) VALUES (?, ?, ?, ?)",
        (user_input, llm_output, llm_model, now)
    )
    conversation_id = cur.lastrowid

    vectors = extract_keywords(combined)
    for kw, vec in vectors:
        vec_np = np.array(vec, dtype='float32')
        cur.execute("INSERT INTO vectors (conversation_id, keyword, vector) VALUES (?, ?, ?)",
                    (conversation_id, kw, vec_np.tobytes()))
        faiss_index.add(vec_np.reshape(1, -1))

    cur.execute("INSERT INTO hash_index (hash) VALUES (?)", (hash_digest,))
    conn.commit()
    print(f"{clock()} Nouvelle conversation insérée (id={conversation_id})")
    return True

# === PARSE LES FICHIERS JSON ===
def parse_lmstudio_file(filepath: str) -> list[tuple[str, str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages")
    if not isinstance(messages, list):
        return []

    pairs = []
    current_question = None
    for message in messages:
        versions = message.get("versions", [])
        if not versions:
            continue

        selected_version = versions[message.get("currentlySelected", 0)]
        role = selected_version.get("role")

        if role == "user":
            parts = selected_version.get("content", [])
            texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            current_question = "\n".join(texts).strip()

        elif role == "assistant" and current_question:
            sender_info = selected_version.get("senderInfo", {})  # Correction ici
            current_model = sender_info.get("senderName", "unknown")
            steps = selected_version.get("steps", [])
            response_parts = []
            for step in steps:
                content_blocks = step.get("content", [])
                texts = [cb.get("text", "") for cb in content_blocks if cb.get("type") == "text"]
                response_parts.extend(texts)

            llm_response = "\n".join(response_parts).strip()
            if llm_response:
                pairs.append((current_question, llm_response, current_model))
                current_question = None

    return pairs

# === PARCOURS DU DOSSIER ===
def import_all():
    new_count = 0
    for root, dirs, files in os.walk(FOLDER_PATH):
        if "Unsync" in dirs:
            dirs.remove("Unsync")
        for file in files:
            if any(file.endswith(ext) for ext in EXTENSIONS):
                full_path = os.path.join(root, file)
                pairs = parse_lmstudio_file(full_path)
                for user_input, llm_output, llm_model in pairs:
                    result = insert_conversation_if_new(user_input, llm_output, llm_model)
                    if result:
                        new_count += 1
    print(f"{clock()} Import terminé, {new_count} nouvelles conversations ajoutées.")

if __name__ == '__main__':
    import_all()
