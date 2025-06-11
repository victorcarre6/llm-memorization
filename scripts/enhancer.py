import os
import re
import json
import sqlite3
import subprocess
import webbrowser
from collections import Counter
import tkinter as tk
from tkinter import scrolledtext, ttk, Canvas
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

import spacy
from langdetect import detect
import torch
import faiss
import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
import warnings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as transformers_logging
from keybert import KeyBERT
import pyperclip
from datetime import datetime


# === INITIALISATION ===

# Chargement des variables de configuration

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(PROJECT_ROOT, "config.json")

def expand_path(value):
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = os.path.normpath(os.path.join(PROJECT_ROOT, expanded))
    return expanded

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    path_keys = {
        "venv_activate_path",
        "lmstudio_folder_path",
        "sync_script_path",
        "project_script_path",
        "db_path",
        "stopwords_file_path"
    }

    config = {}
    for key, value in raw_config.items():
        if isinstance(value, str):
            if key == "summarizing_model" and value == "model/barthez-orangesum-abstract":
                config[key] = expand_path(value)
            elif key in path_keys:
                config[key] = expand_path(value)
            else:
                config[key] = value
        else:
            config[key] = value
    return config

config = load_config(config_path)

stopwords_path = config.get("stopwords_file_path", "stopwords_fr.json")
with open(stopwords_path, "r", encoding="utf-8") as f:
    french_stop_words = set(json.load(f))

combined_stopwords = ENGLISH_STOP_WORDS.union(french_stop_words)

# Masquage des avertissements
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#logging.getLogger("transformers").setLevel(logging.ERROR)
#logging.getLogger("torch").setLevel(logging.ERROR)
#torch._C._log_api_usage_once = lambda *args, **kwargs: None
#warnings.filterwarnings("ignore", message="Unfeasible length constraints", category=UserWarning, module="transformers.generation.utils")

# Connexion à la base SQLite
conn = sqlite3.connect(config["db_path"])
cur = conn.cursor()

# Initialisation des modèles
nlp_fr = spacy.load("fr_core_news_lg")
nlp_en = spacy.load("en_core_web_lg")
kw_model = KeyBERT()
summarizing_model = config.get("summarizing_model", "model/barthez-orangesum-abstract")
summarizing_pipeline = pipeline(task="summarization", model=summarizing_model, framework="pt")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Index vectoriel
VECTOR_DIM = 384
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# === FONCTIONS PRINCIPALES ===

# Synchronisation des conversations
def sync_conversations():
    sync_path = config.get("sync_script_path")

    update_status("⚙️ Synchronisation en cours...")
    root.update()

    if not sync_path:
        label_status.config(text="❌ Erreur lors de la synchronisation (sync_script_path).", foreground='#ff6b6b')
        return False
    try:
        subprocess.run(["python3", sync_path], check=True)
        label_status.config(text="✅ Synchronisation terminée.", foreground='#599258')
        return True
    except subprocess.CalledProcessError:
        label_status.config(text="❌ Erreur lors de la synchronisation (subprocess).", foreground='#ff6b6b')
        return False
    except FileNotFoundError:
        label_status.config(text="❌ Script de synchronisation introuvable.", foreground='#ff6b6b')
        return False

# Pércuteur
def on_ask():
    question = entry_question.get("1.0", "end-1c")
    if not question.strip():
        update_status("⚠️ Saisissez une question.", error=True)
        return
    
    update_status("⚙️ Traitement en cours...")
    root.update()
    
    try:
        context = get_relevant_context(question, limit=context_count_var.get()) #", limit=context_count_var.get()" ajoutée slider contexte
        prompt = generate_prompt_paragraph(context, question)
        pyperclip.copy(prompt)
        text_output.delete('1.0', tk.END)
        text_output.insert(tk.END, prompt)
        
        # Calcul des métriques
        context_count = len(context)
        token_count = len(prompt.split())
        
        update_status(
        f"✅ Prompt généré ({token_count} tokens) | Contexte utilisé : {context_count} élément{'s' if context_count > 1 else ''}",
        success=True
)
    except Exception as e:
        update_status(f"❌ Erreur : {str(e)}", error=True)

# === CONTEXTE ===

# Choix NLP selon langue
def get_nlp_model(text):
    try:
        lang = detect(text)
    except:
        lang = "fr"  # défaut français si détection impossible
    
    if lang.startswith("en"):
        return nlp_en
    else:
        return nlp_fr


# Récupération des mots-clés de la question initiale
root = tk.Tk()
keyword_count_var = tk.IntVar(value=5)
context_count_var = tk.IntVar(value=3)
multiplier = config.get("keyword_multiplier", 2)


def lemmatize_spacy(word, lang="fr"):
    if lang == "fr":
        doc = nlp_fr(word)
    elif lang == "en":
        doc = nlp_en(word)
    else:
        doc = nlp_fr(word)
    return doc[0].lemma_.lower()

def extract_keywords(text, top_n=None):
    if top_n is None:
        top_n = keyword_count_var.get()

    # Extraction brute avec KeyBERT
    raw_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words=list(combined_stopwords),
        top_n=top_n * multiplier)


    stopwords_set = set(combined_stopwords)

    tokens = re.findall(r'\b[a-zA-Z\-]{3,}\b', text.lower())
    token_freq = Counter([tok for tok in tokens if tok not in stopwords_set])

    # Fonction de validation rapide des mots clés
    def is_valid_kw(kw):
        return (
            kw not in stopwords_set and
            len(kw) > 2 and
            kw.isalpha() or '-' in kw
        )

    # Tri par fréquence dans le texte #  /!\ ==peu entrainer des doublons== /!\
    filtered_raw = []
    seen = set()
    for kw, weight in raw_keywords:
        kw_clean = kw.lower().strip()
        if is_valid_kw(kw_clean):
            kw_lemma = lemmatize_spacy(kw_clean)
            # Filtrage manuel pluriels : on saute si singulier déjà vu
            if kw_lemma.endswith('s') and kw_lemma[:-1] in seen:
                continue
            if kw_lemma in seen:
                continue
            freq = token_freq.get(kw_clean, 0)
            filtered_raw.append((freq, kw_lemma, weight))
            seen.add(kw_lemma)

    top_filtered = heapq.nlargest(top_n, filtered_raw, key=lambda x: x[0])

    filtered_keywords = []
    seen = set()
    for freq, kw_lemma, weight in top_filtered:
        if kw_lemma not in seen:
            seen.add(kw_lemma)
            filtered_keywords.append((kw_lemma, weight, freq))

    return filtered_keywords

def get_vector_for_text(text):
    vec = embedding_model.encode([text])
    return np.array(vec[0], dtype='float32')

# Récupération des anciennes conversations pertinentes

def get_relevant_context(user_question, limit=None, similarity_threshold=0.1):
    if limit is None:
        limit = context_count_var.get()

    keywords = extract_keywords(user_question)
    if not keywords or not isinstance(keywords, (list, tuple)):
        print("Warning: keywords non valides ou vides:", keywords)
        return []

    try:
        keyword_strings = [kw[0] for kw in keywords if isinstance(kw, (list, tuple)) and len(kw) > 0]
    except Exception as e:
        print(f"Erreur extraction keywords: {e}")
        return []

    if not keyword_strings:
        print("Warning: liste keyword_strings vide après extraction")
        return []

    query_vectors = '''
        SELECT conversation_id, keyword, vector
        FROM vectors
    '''
    cur.execute(query_vectors)
    vector_rows = cur.fetchall()

    if not vector_rows:
        return []

    convo_ids = []
    vectors = []
    keywords_by_convo = {}

    for convo_id, kw, vec_str in vector_rows:
        if not vec_str or not vec_str.strip():
            continue
        try:
            vec_str_clean = vec_str.strip().replace('\n', '').replace('\r', '').replace('[', '').replace(']', '')
            vec = np.fromstring(vec_str_clean, sep=',').astype('float32')
        except Exception as e:
            print(f"Erreur conversion vecteur pour convo {convo_id}: {e}")
            continue
        convo_ids.append(convo_id)
        vectors.append(vec)
        keywords_by_convo.setdefault(convo_id, set()).add(kw)

    if not vectors:
        return []

    vectors = np.array(vectors).astype('float32')

    # Normalisation pour similarité par cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, a_min=1e-10, a_max=None)

    question_vec = get_vector_for_text(user_question).astype('float32').reshape(1, -1)
    question_norm = np.linalg.norm(question_vec, axis=1, keepdims=True)
    question_vec = question_vec / np.clip(question_norm, a_min=1e-10, a_max=None)

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)

    distances, indices = faiss_index.search(question_vec, limit)

    found_convo_ids = []
    found_scores = {}
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        score = distances[0][i]
        if score >= similarity_threshold:
            convo_id = convo_ids[idx]
            found_convo_ids.append(convo_id)
            found_scores[convo_id] = score


    if not found_convo_ids:
        return []

    placeholders_ids = ', '.join(['?'] * len(found_convo_ids))

    query_contexts = f'''
        SELECT id, user_input, llm_output, timestamp
        FROM conversations
        WHERE id IN ({placeholders_ids})
    '''
    cur.execute(query_contexts, found_convo_ids)
    context_rows = cur.fetchall()

    filtered_context = []
    for convo_id, user_input, llm_output, timestamp in context_rows:
        kws = list(keywords_by_convo.get(convo_id, []))
        score = found_scores.get(convo_id, 0.0)
        filtered_context.append((user_input, llm_output, timestamp, kws, score))

    return filtered_context


# Nettoyage du texte
def nlp_clean_text(text, max_chunk_size=500):
    text = re.sub(r'```(?:python)?\s*.*?```', '', text, flags=re.DOTALL)
    nlp = get_nlp_model(text)
    chunks, current_chunk, current_length = [], [], 0

    for sent in nlp(text).sents:
        s = sent.text.strip()
        if len(s) < 20:
            continue
        if current_length + len(s) < max_chunk_size:
            current_chunk.append(s)
            current_length += len(s)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_length = len(s)
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return " ".join(chunks[:3])  # limite à 3 blocs maximum

# === CONSTRUCTION DU PROMPT ===

# Compression du contexte extrait
def summarize(text, focus_terms=None, max_length=1024):
    transformers_logging.set_verbosity_error()
    update_status("⚙️ Compression du contexte extrait ...")
    root.update()
    try:
        # Filtrage des phrases importantes si focus_terms donné
        if focus_terms:
            sentences = [s for s in text.split('.') 
                        if any(term.lower() in s.lower() for term in focus_terms)]
            text = '. '.join(sentences)[:2000] or text[:2000]

        # Résumé avec le texte filtré
        result = summarizing_pipeline(
            text,
            max_length=max_length,
            min_length=max_length // 2,
            no_repeat_ngram_size=3,
            do_sample=False,
            truncation=True
        )
        return nlp_clean_text(result[0]['summary_text'])

    except Exception as e:
        print(f"Erreur summarization : {e}")
        return text[:max_length] + "... [résumé tronqué]"
    
# Construction du prompt


def generate_prompt_paragraph(context, question, target_tokens=1000):
    update_status("⚙️ Génération du prompt ...")
    root.update()
    if not context:
        return f"{question}"
    # 1. Prétraitement
    processed_items = []
    limit=context_count_var.get() # Nombre max d'éléments dans le contexte
    
    root.update()
    for item in context[:limit]:  
        try:
            # Extraction sécurisée

            user_input = str(item[0])[:300]  # Troncature des questions longues
            llm_output = str(item[1])
            keyword = str(item[5]) if len(item) > 5 and str(item[3]).strip() not in {"", "none", "null", "1", "2", "3"} else None


            # Summarization, netooyage, segmentation
            summary = nlp_clean_text(summarize(llm_output))

            processed_items.append({
                'question': user_input,
                'summary': summary,
                'keyword': keyword if keyword else None
            })
        except Exception as e:
            print(f"Erreur traitement item : {e}")
            continue

    if not processed_items:
        return question

    # 2. Construction du prompt
    parts = []
    # Partie questions
    if processed_items:
        questions = [f"'{item['question']}'" for item in processed_items]
        if len(questions) == 1:
            parts.append(f"Tes discussions avec l'utilisateur t'ont amené à répondre à cette question : {questions[0]}")
        else:
            *init, last = questions
            parts.append(f"Tes discussions avec l'utilisateur t'ont amené à répondre à ces questions :  {', '.join(init)}, et enfin {last}")

    # Partie mots-clés
    keywords = {item['keyword'] for item in processed_items if item['keyword']}
    if keywords:
        parts.append(f"Mots-clés pertinents : {', '.join(sorted(keywords))}")

    # Partie résumés
    if processed_items:
        summaries = [f"- {item['summary']}" for item in processed_items]
        parts.append("Ces intéractions vous ont amené à discuter de ces sujets :\n" + "\n".join(summaries))

    # Question actuelle
    parts.append(f"Réponds maintenant à cette question, dans le contexte de vos discussions précédentes : {question}")

    return "\n".join(parts)

# === INTERFACE TKINTER ===

def update_status(message, error=False, success=False):
    label_status.config(text=message)
    if error:
        label_status.config(foreground='#ff6b6b')
    elif success:
        label_status.config(foreground='#599258')
    else:
        label_status.config(foreground='white')

def open_github(event):
    webbrowser.open_new("https://github.com/victorcarre6")

def show_help():
    help_text = (
        "LLM Memorization and Prompt Enhancer — Aide\n\n"
        "• Synchroniser les conversations : ajoute les nouveaux échanges depuis LM Studio à la base de données.\n\n"
        "• Générer prompt : extrait les mots-clés de votre question, cherche des conversations similaires dans votre base SQL, puis compresse les informations avec un LLM local.\n\n"
        "Le prompt final est affiché puis automatiquement copié dans votre presse-papier !\n\n"
        "Pour en savoir plus, obtenir plus d'informations à propos d'un éventuel bloquage des scripts, ou me contacter, visitez : github.com/victorcarre6/llm-memorization"
    )
    help_window = tk.Toplevel(root)
    help_window.title("Aide")
    help_window.geometry("500x300")
    help_window.configure(bg="#323232")

    text_widget = tk.Text(help_window, wrap=tk.WORD, font=("Segoe UI", 8))
    text_widget.insert(tk.END, help_text)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    ok_button = tk.Button(help_window, text="Fermer", command=help_window.destroy)
    ok_button.pack(pady=10)

# === CONFIGURATION DE L'INTERFACE ===
root.title("LLM Memorization and Prompt Enhancer")
root.geometry("1000x325")
root.configure(bg="#323232")

# Style global unique
style = ttk.Style(root)
style.theme_use('clam')

# Configuration du style
style_config = {
    'Green.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 11),
        'padding': 2
    },
    'Bottom.TButton': {
        'background': '#599258',
        'foreground': 'white',
        'font': ('Segoe UI', 9),
        'padding': 2
    },
    'Blue.TLabel': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 8, 'italic underline'),
        'padding': 0
    },
    'TLabel': {
        'background': '#323232',
        'foreground': 'white',
        'font': ('Segoe UI', 11)
    },
    'TEntry': {
        'fieldbackground': '#FDF6EE',
        'foreground': 'black',
        'font': ('Segoe UI', 11)
    },
    'TFrame': {
        'background': '#323232'
    },
    'Status.TLabel': {
        'background': '#323232',
        'font': ('Segoe UI', 11)
    },
    'TNotebook': {
        'background': '#323232',
        'borderwidth': 0
    },
    'TNotebook.Tab': {
        'background': '#2a2a2a',
        'foreground': 'white',
        'padding': (10, 4)
    },
    'Custom.Treeview': {
        'background': '#2a2a2a',
        'foreground': 'white',
        'fieldbackground': '#2a2a2a',
        'font': ('Segoe UI', 10),
        'bordercolor': '#323232',
        'borderwidth': 0,
    },
    'Custom.Treeview.Heading': {
        'background': '#323232',
        'foreground': '#599258',
        'font': ('Segoe UI', 11, 'bold'),
        'relief': 'flat'
    }
}

for style_name, app_config in style_config.items():
    style.configure(style_name, **app_config)

style.map('Green.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

style.map("TNotebook.Tab",
          background=[("selected", "#323232"), ("active", "#2a2a2a")],
          foreground=[("selected", "white"), ("active", "white")])


style.map('Bottom.TButton',
          background=[('active', '#457a3a'), ('pressed', '#2e4a20')],
          foreground=[('disabled', '#d9d9d9')])

# Widgets principaux
main_frame = ttk.Frame(root, style='TFrame')
main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Section question - Centrée
question_header = ttk.Frame(main_frame, style='TFrame')
question_header.pack(fill='x', pady=(0, 1))
ttk.Label(question_header, text="Poser la question :").pack(expand=True)

question_frame = tk.Frame(main_frame, bg="#323232")
question_frame.pack(pady=(0, 5), fill='x', expand=True)

entry_question = tk.Text(question_frame, height=4, width=80, wrap="word", font=('Segoe UI', 11))
entry_question.pack(side="left", fill="both", expand=True)

# Configuration du style pour la scrollbar
style = ttk.Style()
style.configure("Vertical.TScrollbar",
    troughcolor='#FDF6EE',
    background='#C0C0C0',
    darkcolor='#C0C0C0',
    lightcolor='#C0C0C0',
    bordercolor='#FDF6EE',
    arrowcolor='black',
    relief='flat')

scrollbar = ttk.Scrollbar(
    question_frame,
    orient="vertical",
    command=entry_question.yview,
    style="Vertical.TScrollbar"  # Application du style
)
scrollbar.pack(side="right", fill="y")

entry_question.config(yscrollcommand=scrollbar.set)
entry_question.bind("<Return>", lambda event: on_ask())

# Frame horizontale principale
control_frame = ttk.Frame(main_frame, style='TFrame')
control_frame.pack(fill='x', pady=(0, 10), padx=5)

# Sliders
slider_keywords_frame = ttk.Frame(control_frame, style='TFrame')
slider_keywords_frame.grid(row=0, column=0, sticky='w')

label_keyword_count = ttk.Label(slider_keywords_frame, text=f"Nombre de mots-clés : {keyword_count_var.get()}", style='TLabel')
label_keyword_count.pack(anchor='w')

slider_keywords = ttk.Scale(
    slider_keywords_frame,
    from_=1,
    to=15,
    orient="horizontal",
    variable=keyword_count_var,
    length=180,
    command=lambda val: label_keyword_count.config(text=f"Nombre de mots-clés : {int(float(val))}")
)
slider_keywords.pack(anchor='w')

slider_context_frame = ttk.Frame(control_frame, style='TFrame')
slider_context_frame.grid(row=0, column=1, padx=20, sticky='w')

label_contexts_count = ttk.Label(slider_context_frame, text=f"Nombre de contextes : {context_count_var.get()}", style='TLabel')
label_contexts_count.pack(anchor='w')

slider_contexts = ttk.Scale(
    slider_context_frame,
    from_=1,
    to=10,
    orient=tk.HORIZONTAL,
    variable=context_count_var,
    length=180,
    command=lambda val: label_contexts_count.config(text=f"Nombre de contextes : {int(float(val))}")
)
slider_contexts.pack(anchor='w')

# Boutons synchronisation et percuteur
button_frame = ttk.Frame(control_frame, style='TFrame')
button_frame.grid(row=0, column=2, sticky='e')

sync_button = ttk.Button(button_frame, text="Synchroniser les conversations", 
                         command=sync_conversations, style='Green.TButton')
sync_button.pack(side='left', padx=5)

btn_ask = ttk.Button(button_frame, text="Générer prompt", command=on_ask, style='Green.TButton')
btn_ask.pack(side='left', padx=5)
control_frame.grid_columnconfigure(2, weight=1)

# Zone de sortie étendable
output_expanded = tk.BooleanVar(value=False)

def toggle_output():
    """Basculer l'affichage de la zone de sortie et ajuster la taille de la fenêtre"""
    if output_expanded.get():
        text_output.pack_forget()
        toggle_btn.config(text="▼ Afficher le résultat")
        output_expanded.set(False)
        root.geometry("1000x325")
    else:
        text_output.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        toggle_btn.config(text="▲ Masquer le résultat")
        output_expanded.set(True)
        root.geometry("1000x750")

output_frame = ttk.Frame(main_frame, style='TFrame')
output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))



# Bouton pour étendre/cacher
toggle_btn = ttk.Button(
    output_frame,
    text="▼ Afficher le résultat",
    command=toggle_output,
    style='Green.TButton'  # Utilise le même style que tes autres boutons
)
toggle_btn.pack(fill=tk.X, pady=(0, 5))

text_output = scrolledtext.ScrolledText(
    output_frame, 
    width=100, 
    height=20,
    font=('Segoe UI', 11), 
    wrap=tk.WORD, 
    bg="#FDF6EE", 
    fg="black", 
    insertbackground="black"
)

# Clics droits sur les zones de texte

context_menu = tk.Menu(text_output, tearoff=0)
context_menu.add_command(label="Copier", command=lambda: text_output.event_generate("<<Copy>>"))

def show_context_menu(event):
    context_menu.tk_popup(event.x_root, event.y_root)

text_output.bind("<Button-3>", show_context_menu)


question_menu = tk.Menu(entry_question, tearoff=0)
question_menu.add_command(label="Coller", command=lambda: entry_question.event_generate("<<Paste>>"))

def show_question_menu(event):
    question_menu.tk_popup(event.x_root, event.y_root)

entry_question.bind("<Button-3>", show_question_menu)



#text_output.bind("<Button-3>", show_context_menu)

    #for (convo_id, user_input, llm_output, timestamp, kws, score) in filtered_context
    #for (kw_lemma, weight, freq) in filtered_keywords

    # On récupère les infos de la question initiales
    #question_vector = np.array(vec[0], dtype='float32')



def show_infos():
    global notebook
    info_window = tk.Toplevel(root)
    info_window.title("Détails sur le prompt généré et sur la base de donnée")
    info_window.geometry("900x750")
    container = tk.Frame(info_window, bg="#323232")
    container.pack(fill="both", expand=True)

    info_window.transient(root)
    info_window.grab_set()

    question = entry_question.get("1.0", tk.END).strip()
    if not question:
        update_status("⚠️ Saisissez une question avant de continuer.", error=True)
        info_window.destroy()
        return

    filtered_keywords = extract_keywords(question)
    filtered_context = get_relevant_context(question)

# -- Onglet Input

    notebook = ttk.Notebook(container, style="TNotebook")
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    single_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(single_tab, text="Input")

    if filtered_keywords:
        kw_lemmas = [kw for kw, _, _ in filtered_keywords]
        freqs = [freq for _, _, freq in filtered_keywords]
        weights = [weight for _, weight, _ in filtered_keywords]

        # --- Barplot fréquence + poids ---

        sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)

        kw_lemmas_sorted = [kw_lemmas[i] for i in sorted_indices]
        freqs_sorted = [freqs[i] for i in sorted_indices]
        weights_sorted = [weights[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        bar_width = 0.4
        x = range(len(kw_lemmas_sorted))

        bars1 = ax.bar(x, freqs_sorted, width=bar_width, label="Occurrences", color="#599258")
        bars2 = ax.bar([i + bar_width for i in x], weights_sorted, width=bar_width, label="Poids", color="#a2d149")

        ax.set_facecolor("#323232")
        fig.patch.set_facecolor("#323232")
        ax.set_xticks([i + bar_width / 2 for i in x])
        ax.set_xticklabels(kw_lemmas_sorted, rotation=45, ha='right', color="white", fontsize=10)
        ax.tick_params(axis='y', colors="white", labelsize=10)
        ax.set_title("Fréquence et poids des mots-clés de la question", color="white", fontsize=10, fontweight='bold')
        ax.legend(facecolor="#323232", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_color('white')

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=single_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))
        plt.close(fig)


        # --- Tableau amélioré ---

        max_scores = {}
        for item in filtered_context:
            user_input = item[1]
            score = item[4]
            kws = set(kw for kw, _, _ in extract_keywords(user_input))
            for kw in kws:
                if kw not in max_scores or score > max_scores[kw]:
                    max_scores[kw] = score

        cols = ('Mot-clé', 'Occurrences', 'Poids', 'Score max')
        tree = ttk.Treeview(single_tab, columns=cols, show='headings', height=12, style='Custom.Treeview')
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=150 if col != 'Mot-clé' else 200, anchor=tk.CENTER)
        tree.pack(expand=False, fill='both', padx=10, pady=(10, 0))

        tree.tag_configure('oddrow', background='#2a2a2a')
        tree.tag_configure('evenrow', background='#383838')

        for i, (kw, weight, freq) in enumerate(filtered_keywords):
            score = max_scores.get(kw, 0.0)
            tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            tree.insert('', tk.END, values=(kw, freq, f"{weight:.3f}", f"{score:.3f}"), tags=(tag,))

    # --- Onglet Contextes ---
    context_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(context_tab, text="Output")
    lbl_container = tk.Frame(context_tab, bg="#323232")
    lbl_container.pack(fill="both", expand=True)

        # Affichage des questions colorées

    tk.Label(lbl_container, text="Questions contextuelles :", fg="white", bg="#323232", font=("Segoe UI", 10, "bold")).pack(pady=5)
    tk.Label(lbl_container,
        text="Légende :",
        fg="white",
        bg="#323232",
        justify="left",
        wraplength=700,
        font=("Segoe UI", 8, "italic")
    ).pack(anchor="w", padx=10)

    legend_texts = [
        ("- 1 mot clef", "#ff6b6b"),
        ("- 2 ou 3 mots clefs", "#ffb347"),
        ("- 4 mots clefs ou plus", "#599258"),
    ]

    for text, color in legend_texts:
        label = tk.Label(
            lbl_container,
            text=text,
            fg=color,
            bg="#323232",
            font=("Segoe UI", 8, "italic")
        )
        label.pack(anchor="w", padx=30)

    tk.Label(lbl_container, text="", bg="#323232").pack(pady=5)
    base_keywords = set(kw for kw, _, _ in filtered_keywords) #user_input, timestamp, kws
    for item in filtered_context:
        q_text = item[0]  # ici user_input
        timestamp = item[2] # changer nombre si timestamp non donné
        extracted = set(kw for kw, _, _ in extract_keywords(q_text))
        shared = len(extracted & base_keywords)
        color = "#ff6b6b" if shared <= 1 else "#ffb347" if shared <= 3 else "#7CFC00"
        display_text = f"{timestamp} - {q_text[:250]}{'...' if len(q_text) > 250 else ''}"
        tk.Label(lbl_container, text=display_text, fg=color, bg="#323232", wraplength=700, justify="left").pack(anchor="w", padx=10)



    # Nuage de mots des mots clefs des questions extraites

    tk.Label(lbl_container, text="Nuage de mots clefs des contextes :", fg="white", bg="#323232", font=("Segoe UI", 10, "bold")).pack(pady=2)
    tk.Label(lbl_container, text="🚧WIP🚧 Taille : fréquence, couleur : poids", fg="white", bg="#323232", font=("Segoe UI", 7, "italic")).pack(pady=2)

        ## Comptage des mots-clés
    kw_counter = Counter()
    for item in filtered_context:
        if len(item) > 3 and isinstance(item[3], list):
            kw_counter.update(item[3])

        ## Création du nuage de mots basé sur les fréquences
    wc = WordCloud(
        width=700,
        height=400,
        background_color="#323232",
        mode="RGBA",
        colormap="YlGnBu",
        prefer_horizontal=0.8,
        min_font_size=10,
        max_font_size=80
    ).generate_from_frequencies(kw_counter)

            ### Définition d'un colormap personnalisé du vert clair au vert foncé
    custom_cmap = LinearSegmentedColormap.from_list(
            "custom_green",
            ["#599238", "#323232"]
        )

        ### Fonction de recolorisation selon la fréquence relative (entre 0 et 1)
    def color_func(word, **kwargs):
        freq = kwargs.get('frequency', 0)
        max_freq = max(kw_counter.values()) if kw_counter else 1
        norm_freq = freq / max_freq if max_freq > 0 else 0
        norm_freq = 0.1 + 0.9 * norm_freq
        rgba = custom_cmap(norm_freq)
        r, g, b = [int(255 * x) for x in rgba[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"

            ### Appliquer la recolorisation
    wc = wc.recolor(color_func=color_func)

        ## Affichage dans l'interface Tkinter
    fig3, ax3 = plt.subplots(figsize=(7, 4), dpi=100)
    ax3.imshow(wc, interpolation='bilinear')
    ax3.axis("off")
    fig3.patch.set_facecolor("#323232")
    ax3.set_facecolor("#323232")

    canvas3 = FigureCanvasTkAgg(fig3, master=lbl_container)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))
    plt.close(fig3)

    # --- Onglet Carte Mentale ---
    mindmap_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(mindmap_tab, text="Carte mentale")

    canvas = tk.Canvas(mindmap_tab, bg="#323232", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    questions_list = [item[0] for item in filtered_context]
    keywords_per_question = [set(kw for kw, _, _ in extract_keywords(q)) for q in questions_list]
    center_x, center_y, radius = 350, 300, 250
    radius_node = 15
    positions = [(center_x + radius * math.cos(2 * math.pi * i / len(questions_list)),
                  center_y + radius * math.sin(2 * math.pi * i / len(questions_list)))
                 for i in range(len(questions_list))]

    for i in range(len(questions_list)):
        for j in range(i + 1, len(questions_list)):
            if keywords_per_question[i] & keywords_per_question[j]:
                canvas.create_line(*positions[i], *positions[j], fill="#7CFC00", width=1)

    tooltip = None

    def show_tooltip(event, text):
        nonlocal tooltip
        if tooltip:
            canvas.delete(tooltip)
        tooltip = canvas.create_text(event.x + 10, event.y + 10, text=text, anchor="nw", fill="white", font=("Segoe UI", 9), width=300)

    def hide_tooltip(event):
        nonlocal tooltip
        if tooltip:
            canvas.delete(tooltip)
            tooltip = None

    for i, (x, y) in enumerate(positions):
        node = canvas.create_oval(x - radius_node, y - radius_node, x + radius_node, y + radius_node, fill="#599258", outline="")
        canvas.tag_bind(node, "<Enter>", lambda e, q=questions_list[i]: show_tooltip(e, q))
        canvas.tag_bind(node, "<Leave>", hide_tooltip)

    # --- Database ---

    stats_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(stats_tab, text="Base de donnée")

    frame_stats = tk.Frame(stats_tab, bg="#323232")
    frame_stats.pack(fill="both", expand=True, padx=20, pady=20)

    # Reconnexion à la base SQLite ici
    conn = sqlite3.connect(config["db_path"])
    cur = conn.cursor()

    # Requêtes SQL pour les stats globales
    cur.execute("SELECT COUNT(*) FROM conversations")
    nb_conversations = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM vectors")
    nb_mots_clefs = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT keyword) FROM vectors")
    nb_mots_clefs_uniques = cur.fetchone()[0]

    conn.close()

    # Affichage
    tk.Label(frame_stats, text="Statistiques globales de la base", fg="white", bg="#323232", font=("Segoe UI", 10, "bold")).pack(pady=(0, 20))
    tk.Label(frame_stats, text=f"Nombre de conversations : {nb_conversations}", fg="white", bg="#323232", font=("Segoe UI", 10)).pack(anchor="w", pady=2)
    tk.Label(frame_stats, text=f"Nombre total de mots clefs : {nb_mots_clefs}", fg="white", bg="#323232", font=("Segoe UI", 10)).pack(anchor="w", pady=2)
    tk.Label(frame_stats, text=f"Nombre de mots clefs uniques : {nb_mots_clefs_uniques}", fg="white", bg="#323232", font=("Segoe UI", 10)).pack(anchor="w", pady=2)

    tk.Label(frame_stats, text="🚧WIP🚧 Visualisation des données conversationnelles globales", fg="white", bg="#323232", font=("Segoe UI", 10, "bold")).pack(pady=2)

# Barre de statut et boutons
status_buttons_frame = ttk.Frame(main_frame, style='TFrame')
status_buttons_frame.pack(fill=tk.X, pady=(5, 2))

# Barre de statut
label_status = ttk.Label(
    status_buttons_frame,
    text="Prêt",
    style='Status.TLabel',
    foreground='white',
    anchor='w'
)
label_status.pack(side=tk.LEFT, anchor='w')

# Boutons Info et Aide
right_buttons = ttk.Frame(status_buttons_frame, style='TFrame')
right_buttons.pack(side=tk.RIGHT, anchor='e')

btn_info = ttk.Button(right_buttons, text="Détails", style='Bottom.TButton', command=show_infos, width=8)
btn_info.pack(side=tk.TOP, pady=(0, 3))

btn_help = ttk.Button(right_buttons, text="Aide", style='Bottom.TButton', command=show_help, width=8)
btn_help.pack(side=tk.TOP)

# Footer - inchangé
footer_frame = ttk.Frame(root, style='TFrame')
footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

dev_label = ttk.Label(footer_frame, text="Développé par Victor Carré —", style='TLabel', font=('Segoe UI', 8))
dev_label.pack(side=tk.LEFT)

github_link = ttk.Label(footer_frame, text="GitHub", style='Blue.TLabel', cursor="hand2")
github_link.pack(side=tk.LEFT)
github_link.bind("<Button-1>", open_github)

root.mainloop()