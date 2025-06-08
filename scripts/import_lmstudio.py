import os
import json
import sqlite3
import hashlib
from keybert import KeyBERT
from datetime import datetime

# === CONFIGURATION ===
DB_PATH = '/Users/victorcarre/Code/Projects/llm-memorization/datas/conversations.db'
FOLDER_PATH = '/Users/victorcarre/.lmstudio/conversations'  # ← À modifier
EXTENSIONS = ['.json']
TOP_K = 5

# === INITIALISATIONS ===
kw_model = KeyBERT()
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# === TABLES SI BESOIN ===
cur.execute('''
    CREATE TABLE IF NOT EXISTS hash_index (
        hash TEXT PRIMARY KEY
    )
''')
conn.commit()

# === EXTRACTION MOTS-CLÉS ===
def extract_keywords(text, top_n=TOP_K):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw for kw, _ in keywords]

# === INSÉRER CONVERSATION AVEC DÉDOUBLONNAGE ===
def insert_conversation_if_new(user_input, llm_output):
    combined = user_input + llm_output
    hash_digest = hashlib.sha256(combined.encode('utf-8')).hexdigest()

    cur.execute("SELECT 1 FROM hash_index WHERE hash = ?", (hash_digest,))
    if cur.fetchone():
        return False  # Déjà présent

    # Insertion
    cur.execute("INSERT INTO conversations (user_input, llm_output) VALUES (?, ?)", (user_input, llm_output))
    conversation_id = cur.lastrowid

    keywords = extract_keywords(combined)
    for kw in keywords:
        cur.execute("INSERT INTO keywords (conversation_id, keyword) VALUES (?, ?)", (conversation_id, kw))

    cur.execute("INSERT INTO hash_index (hash) VALUES (?)", (hash_digest,))
    conn.commit()
    return True

# === PARSE LES FICHIERS JSON ===
def parse_lmstudio_file(filepath: str) -> list[tuple[str, str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages")
    if not isinstance(messages, list):
        print(f"⚠️ Format inattendu (pas une liste) dans : {filepath}")
        return []

    pairs = []
    current_question = None
    for message in messages:
        versions = message.get("versions", [])
        if not versions:
            continue

        selected_version = versions[message.get("currentlySelected", 0)]
        role = selected_version.get("role")

        # Pour les questions de l'utilisateur
        if role == "user":
            parts = selected_version.get("content", [])
            texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            current_question = "\n".join(texts).strip()

        # Pour les réponses de l'assistant
        elif role == "assistant" and current_question:
            steps = selected_version.get("steps", [])
            response_parts = []
            for step in steps:
                content_blocks = step.get("content", [])
                texts = [cb.get("text", "") for cb in content_blocks if cb.get("type") == "text"]
                response_parts.extend(texts)

            llm_response = "\n".join(response_parts).strip()
            if llm_response:
                pairs.append((current_question, llm_response))
                current_question = None  # reset pour le prochain échange

    return pairs



# === PARCOURS DU DOSSIER ===
def import_all():
    new_count = 0
    for root, _, files in os.walk(FOLDER_PATH):
        for file in files:
            if any(file.endswith(ext) for ext in EXTENSIONS):
                full_path = os.path.join(root, file)
                pairs = parse_lmstudio_file(full_path)
                for user_input, llm_output in pairs:
                    if insert_conversation_if_new(user_input, llm_output):
                        new_count += 1
    print(f"✅ Import terminé. {new_count} nouvelles paires insérées.")

if __name__ == '__main__':
    import_all()
