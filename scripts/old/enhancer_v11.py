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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
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
def sync_conversations(config, label_status):
    sync_path = config.get("sync_script_path")
    if not sync_path:
        label_status.config(text="sync_script_path introuvable.")
        return False
    try:
        subprocess.run(["python3", sync_path], check=True)
        label_status.config(text="Synchronisation terminée.")
        return True
    except subprocess.CalledProcessError:
        label_status.config(text="Erreur lors de la synchronisation.")
        return False
    except FileNotFoundError:
        label_status.config(text="Script de synchronisation introuvable.")
        return False

# Pércuteur
def on_ask():
    question = entry_question.get("1.0", "end-1c")
    if not question.strip():
        update_status("⚠️ Merci de saisir une question.", error=True)
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
        f"Prompt généré ({token_count} tokens) | Contexte utilisé : {context_count} élément{'s' if context_count > 1 else ''}",
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

def get_relevant_context(user_question, limit=None, similarity_thresold=0.75):
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

    # Normalisation pour simulrité par cosine
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, a_min=1e-10, a_max=None)

    question_vec = get_vector_for_text(user_question).astype('float32').reshape(1, -1)
    question_norm = np.linalg.norm(question_vec, axis=1, keepdims=True)
    question_vec = question_vec / np.clip(question_norm, a_min=1e-10, a_max=None)

    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)

    distances, indices = faiss_index.search(question_vec, limit)

    found_convo_ids = [convo_ids[idx] for idx in indices[0] if idx != -1]

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
    for idx, (convo_id, user_input, llm_output, timestamp) in enumerate(context_rows):
        kws = list(keywords_by_convo.get(convo_id, []))
        score = distances[0][idx] if idx < len(distances[0]) else 0.0
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
    if not context:
        return f"{question}"

    # 1. Prétraitement
    processed_items = []
    for item in context[:3]:  # Nombre max d'éléments dans le contexte
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
    """Met à jour le label de statut avec style approprié"""
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
        'padding': 4
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
    to=5,
    orient=tk.HORIZONTAL,
    variable=context_count_var,
    length=170,
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
def show_infos():
    global notebook
    context_tab = ttk.Frame(notebook, style="TFrame")
    info_window = tk.Toplevel(root)
    info_window.title("Détails sur le prompt généré")
    info_window.geometry("750x725")
    container = tk.Frame(info_window, bg="#323232")
    container.pack(fill="both", expand=True)

    info_window.transient(root)
    info_window.grab_set()

    question = entry_question.get("1.0", tk.END).strip()
    if not question:
        update_status("⚠️ Posez une question d'abord.", error=True)
        return

    # On récupère le vecteur d'embedding de la question
    question_vector = get_vector_for_text(question)
    if question_vector is None:
        update_status("⚠️ Impossible de générer l'embedding pour la question.", error=True)
        return

    # Récupération des contextes similaires via vecteurs
    filtered_context = get_relevant_context(question, limit=5)
    if not filtered_context:
        update_status("⚠️ Aucun contexte pertinent trouvé.", error=True)
        return

    #filtered_keywords = extract_keywords(texte, top_n=20)

    # Extraction de mots-clés uniquement pour affichage, mais on ne s'en sert plus pour la recherche
    res = extract_keywords(question)
    if isinstance(res, tuple) and len(res) == 2:
        keywords = res[0]
    else:
        keywords = res
        update_status("⚠️ Aucun mot-clé extrait pour affichage.", error=True)
        return

    # --- Affichage debug console ---
    print("=== Keywords (premiers éléments) ===")
    print(keywords[:5])
    print("=== Tuples dans keywords (type et longueur) ===")
    for kw_tuple in keywords[:5]:
        print(type(kw_tuple), len(kw_tuple) if hasattr(kw_tuple, '__len__') else "N/A", kw_tuple)
    print("=== Tokens extraits des contextes (premiers 20) ===")

    # On rassemble le texte complet des contextes pour compter les mots-clés extraits
    full_text_context = " ".join(
        (item[0] + " " + item[1]) if len(item) >= 2 else item[0]
        for item in filtered_context
    ).lower()
    token_list = re.findall(r'\b[a-zA-Z\-]{3,}\b', full_text_context)
    token_list_clean = [str(t).lower() for t in token_list]

    print(token_list_clean[:20])

    # Comptage occurrences mots-clés dans les tokens
    word_counts = {}
    for kw_tuple in keywords:
        try:
            if isinstance(kw_tuple, (list, tuple)):
                kw = kw_tuple[0]
            else:
                kw = kw_tuple
            kw = str(kw).lower()
            word_counts[kw] = token_list_clean.count(kw)
        except Exception as e:
            print(f"Erreur avec le mot-clé : {kw_tuple} -> {e}")

    # --- Nouveau graphique double barres ---
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)

    question_keywords = []
    question_weights = []
    context_freqs = []

    for kw_tuple in keywords:
        if isinstance(kw_tuple, (tuple, list)) and len(kw_tuple) >= 2:
            kw = kw_tuple[0]
            try:
                weight = float(kw_tuple[1])
            except Exception:
                weight = 0.0
            freq = word_counts.get(str(kw).lower(), 0)
            question_keywords.append(kw)
            question_weights.append(weight)
            context_freqs.append(freq)

    x = np.arange(len(question_keywords))
    width = 0.35

    bars1 = ax.bar(x - width/2, question_weights, width, label='Poids TF-IDF', color='#599258')
    bars2 = ax.bar(x + width/2, context_freqs, width, label='Occurrences dans contextes', color='#ad5e5e')

    ax.set_xticks(x)
    ax.set_xticklabels(question_keywords, rotation=45, ha='right', fontsize=9)
    ax.set_facecolor("#323232")
    fig.patch.set_facecolor("#323232")
    ax.set_title("Comparaison poids TF-IDF vs fréquence contextuelle", color="white", fontsize=11)
    ax.legend(facecolor='#323232', edgecolor='white', labelcolor='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=container)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))
    plt.close(fig)

    # Tableau : mot-clé, poids, fréquence dans contextes
    cols = ('Mot-clé', 'Poids (TF-IDF)', 'Occurrences dans les contextes')
    tree = ttk.Treeview(container, columns=cols, show='headings', height=10, style='Custom.Treeview')
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=180)
    tree.pack(expand=False, fill='both', padx=10, pady=(10, 0))

    tree.tag_configure('oddrow', background='#2a2a2a')
    tree.tag_configure('evenrow', background='#383838')
    tree.tag_configure('highlight', background='#455a40')

    for i, kw_tuple in enumerate(keywords):
        if isinstance(kw_tuple, (tuple, list)) and len(kw_tuple) >= 2:
            kw = kw_tuple[0]
            weight_raw = kw_tuple[1]
            try:
                weight = float(weight_raw)
            except Exception:
                weight = 0.0

            kw_str = str(kw).lower()
            freq = word_counts.get(kw_str, 0)
            tag = 'highlight' if freq > 0 else ('evenrow' if i % 2 == 0 else 'oddrow')

            tree.insert('', tk.END, values=(kw, f"{weight:.3f}", freq), tags=(tag,))

    def copy_keywords():
        kws = []
        for kw_tuple in keywords:
            if isinstance(kw_tuple, (tuple, list)) and len(kw_tuple) >= 1:
                kws.append(str(kw_tuple[0]))
            elif isinstance(kw_tuple, str):
                kws.append(kw_tuple)
        root.clipboard_clear()
        root.clipboard_append(",".join(kws))

    ttk.Button(container, text="Copier mots-clés", command=copy_keywords).pack(pady=10)


    # --- Onglet contextes ---
    context_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(context_tab, text="Contextes")
    lbl_container = tk.Frame(context_tab, bg="#323232")
    lbl_container.pack(fill="both", expand=True)

    tk.Label(lbl_container, text="Questions contextuelles :", fg="white", bg="#323232", font=("Segoe UI", 10, "bold")).pack(pady=5)
    tk.Label(lbl_container, text="Légende :\n- 1 mot clef : rouge\n- 2 ou 3 : orange\n- >3 : vert",
             fg="white", bg="#323232", justify="left", wraplength=700, font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=10, pady=(0, 10))

    base_keywords = set()
    for item in keywords:
        if isinstance(item, (list, tuple)):
            if len(item) > 0:
                base_keywords.add(item[0])
        elif isinstance(item, str):
            base_keywords.add(item)
        else:
            pass

    for item in filtered_context:
        q_text = item[0]
        extracted = set(kw for kw, _, _ in extract_keywords(q_text))
        shared = len(extracted & base_keywords)
        color = "#ff6b6b" if shared <= 1 else "#ffb347" if shared <= 3 else "#7CFC00"
        tk.Label(lbl_container, text=q_text[:100] + ("..." if len(q_text) > 100 else ""), fg=color, bg="#323232", wraplength=700).pack(anchor="w", padx=10)

    # --- Onglet carte mentale ---
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

    # --- Onglet Stats Globales ---
    global_tab = ttk.Frame(notebook, style="TFrame")
    notebook.add(global_tab, text="Stats Globales")

    top_n = 20
    top_keywords = filtered_keywords[:top_n]
    labels = [kw for kw, _, _ in top_keywords]
    occurrences = [count for _, count, _ in top_keywords]

    fig2, ax2 = plt.subplots(figsize=(7, 4), dpi=100)
    ax2.bar(labels, occurrences, color='#457b9d')
    ax2.set_facecolor("#323232")
    fig2.patch.set_facecolor("#323232")
    ax2.set_title("Top 20 mots-clés de toute la base", color="white", fontsize=10)
    ax2.set_ylabel("Occurrences totales", color="white", fontsize=10)
    ax2.tick_params(axis='x', labelrotation=45, colors="white", labelsize=10)
    ax2.tick_params(axis='y', colors="white", labelsize=10)
    for spine in ax2.spines.values():
        spine.set_color('white')

    fig2.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=global_tab)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 0))
    plt.close(fig2)

    search_var = tk.StringVar()

    def filter_treeview(*args):
        search_term = search_var.get().lower()
        tree_global.delete(*tree_global.get_children())
        for i, (kw, total, nb_conv) in enumerate(filtered_keywords):
            if search_term in kw.lower():
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                tree_global.insert('', tk.END, values=(kw, total, nb_conv), tags=(tag,))

    search_var.trace_add('write', filter_treeview)

    search_entry = ttk.Entry(global_tab, textvariable=search_var)
    search_entry.pack(fill='x', padx=10, pady=(10, 0))
    search_entry.insert(0, "Rechercher un mot-clé...")

    def on_entry_click(event):
        if search_entry.get() == "Rechercher un mot-clé...":
            search_entry.delete(0, tk.END)
    search_entry.bind('<FocusIn>', on_entry_click)

    cols_global = ('Mot-clé', 'Occurrences totales', 'Nb conversations')
    tree_global = ttk.Treeview(global_tab, columns=cols_global, show='headings', height=15, style='Custom.Treeview')
    for col in cols_global:
        tree_global.heading(col, text=col)
        tree_global.column(col, width=180)
    tree_global.pack(expand=False, fill='both', padx=10, pady=(5, 0))

    tree_global.tag_configure('oddrow', background='#2a2a2a')
    tree_global.tag_configure('evenrow', background='#383838')

    for i, (kw, total, nb_conv) in enumerate(token_freq):
        tag = 'evenrow' if i % 2 == 0 else 'oddrow'
        tree_global.insert('', tk.END, values=(kw, total, nb_conv), tags=(tag,))




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