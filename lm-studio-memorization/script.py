import sys
from memory_utils import get_relevant_context
import sqlite3
import json
from keybert import KeyBERT
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Chargement configuration
import os
import json as js

with open("config.json", "r") as f:
    config = js.load(f)

DB_PATH = config["memory_db"]
SUMMARY_MODEL = config["summary_model"]

# Chargement des modèles
kw_model = KeyBERT()
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model=SUMMARY_MODEL)

def fetch_memory():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM memory ORDER BY timestamp DESC")
    data = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "content": row[1]} for row in data]

def get_relevant_context(question, top_n=3):
    memory = fetch_memory()
    corpus = [m["content"] for m in memory]
    if not corpus:
        return "Pas de mémoire disponible."

    keywords = kw_model.extract_keywords(question, top_n=5, stop_words='french')
    kw_list = [kw for kw, score in keywords]

    scores = []
    for entry in corpus:
        entry_kw = kw_model.extract_keywords(entry, top_n=5, stop_words='french')
        entry_kw_list = [kw for kw, score in entry_kw]
        score = len(set(kw_list) & set(entry_kw_list))
        scores.append(score)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    top_texts = [corpus[i] for i in top_indices]
    summaries = [summarizer(text, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
                 for text in top_texts]

    return "\n".join(summaries)

def generate_prompt_paragraph(context, question):
    return f"""Tu es un assistant intelligent. Voici un contexte issu de conversations passées :
{context}

En te basant sur ce contexte, réponds à la question suivante :
{question}"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erreur : question manquante")
        sys.exit(1)

    question = sys.argv[1]
    context = get_relevant_context(question)
    prompt = generate_prompt_paragraph(context, question)

    print(prompt)
